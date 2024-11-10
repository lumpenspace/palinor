"""
Manager class for pngr operations.
"""

import os
from pathlib import Path
from typing import Any, Optional
import torch
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from huggingface_hub import login
import dotenv

from .ControllableModel import ControllableModel
from .ControlVector import ControlVector
from .message_template import message_template
from . import create_dataset


dotenv.load_dotenv()

console = Console()


class PngrManager:
    """
    Manager class for pngr operations.

    Handles model loading, vector training and storage, and inference operations.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = os.getenv("HF_TOKEN"),
        layer_ids: Optional[list[int]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the manager.

        Args:
            model_name: Name of the model on HuggingFace
            cache_dir: Directory to store models and vectors (default: ~/.pngr)
            hf_token: HuggingFace token for gated models
            layer_ids: Which layers to control (default: [-1, -2, -3])
            device: Device to use for computation (default: auto)
        """
        console.print(f"Initializing PngrManager for model {model_name}")
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.pngr"))
        self.vectors_dir = self.cache_dir / "vectors" / model_name.replace("/", "_")
        self.models_dir = self.cache_dir / "models"

        # Set device - prefer CUDA if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"Using device: {self.device}")

        # Create cache directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Set up HF token if provided
        if hf_token:
            login(token=hf_token)

        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer: PreTrainedTokenizerBase = self._load_tokenizer()

        # Create controllable model
        self.layer_ids = layer_ids or [-1, -2, -3]
        self.controllable_model = ControllableModel(
            self.model, layer_ids=self.layer_ids
        )

        # Load existing vectors
        self.vectors = self._load_existing_vectors()

        console.print(f"Loaded {len(self.vectors)} vectors")

    def _load_model(self) -> PreTrainedModel:
        """Load or download the model."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.models_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cuda":
            model = model.cuda()
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Load or download the tokenizer."""
        return AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.models_dir),
            padding_side="left",
        )

    def _load_existing_vectors(self) -> dict[str, ControlVector]:
        """Load all existing control vectors for this model."""
        vectors = {}
        for vector_path in self.vectors_dir.glob("*.pkl"):
            name = vector_path.stem
            try:
                vectors[name] = ControlVector.from_file(str(vector_path))
            except Exception as e:
                print(f"Failed to load vector {name}: {e}")
        return vectors

    def train_vector(
        self, name: str, a_adjective: str, b_adjective: str, **kwargs: Any
    ) -> ControlVector:
        """
        Train a new control vector.

        Args:
            name: Name to save the vector as
            a_adjective: First personality trait
            b_adjective: Second personality trait
            **kwargs: Additional training parameters

        Returns:
            The trained ControlVector
        """

        # Create dataset
        template_path = os.path.join(
            os.path.dirname(__file__), "..", "dataset_templates/alphapenger.yaml"
        )
        prompts = create_dataset.create_personality_prompts(
            template_path, a_adjective, b_adjective
        )

        console.print(
            f"Training vector [green]{name}[/green] with {len(prompts)} prompts"
        )
        # Train vector
        vector = ControlVector.train(
            model=self.controllable_model,
            tokenizer=self.tokenizer,
            dataset=prompts,
            **kwargs,
        )

        # Save vector
        vector_path: Path = self.vectors_dir / f"{name}.pkl"
        console.print(f"Saving vector to [green]{vector_path}[/green]")
        vector.to_file(str(vector_path))

        # Add to loaded vectors
        self.vectors[name] = vector

        return vector

    def generate(
        self,
        prompt: str,
        vector_name: Optional[str] = None,
        coeff: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with optional control vector.

        Args:
            prompt: Input text
            vector_name: Name of vector to use (if any)
            coeff: Control strength (negative inverts control)
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if vector_name is not None:
            if vector_name not in self.vectors:
                raise ValueError(
                    f"Vector '{vector_name}' not found. Available vectors: "
                    f"{list(self.vectors.keys())}"
                )
            console.print(f"Using vector '{vector_name}' with strength {coeff}")
            self.controllable_model.set_control(self.vectors[vector_name], coeff=coeff)
        # Format prompt using Llama chat format with system message
        formatted_prompt = message_template(
            prompt,
            system_message="you are a helpful AI assistant for travel tips and recommendations",
        )

        # Extract generation parameters
        max_new_tokens = kwargs.pop("max_new_tokens", 50)

        # Create inputs
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )

        try:
            # Generate directly using the model
            outputs = self.controllable_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

            # Decode the output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up the response
            if response.startswith(formatted_prompt):
                response = response[len(formatted_prompt) :].strip()

            # Escape any rich markup characters
            response = response.replace("[", "\\[").replace("]", "\\]")
            self.controllable_model.reset()
            return response

        except Exception as e:
            console.print(f"[red]Error during generation: {str(e)}[/red]")
            return f"Error during generation: {str(e)}"

    def list_vectors(self) -> list[str]:
        """Get names of available vectors."""
        return list(self.vectors.keys())

    def reset(self):
        """Reset the controllable model."""
        self.controllable_model.reset()

    def generate_tweet(
        self,
        prompt: str,
        vector_name: Optional[str] = None,
        coeff: float = 1.0,
        max_length: int = 280,
        **kwargs: Any,
    ) -> str:
        """
        Generate a Twitter-compatible response.

        Args:
            prompt: Input text
            vector_name: Name of vector to use (if any)
            coeff: Control strength
            max_length: Maximum tweet length (default: 280)
            **kwargs: Additional generation parameters

        Returns:
            Generated tweet text
        """
        # Set max_new_tokens based on max_length
        kwargs["max_new_tokens"] = kwargs.get("max_new_tokens", max_length // 4)

        try:
            output = self.generate(prompt, vector_name, coeff, **kwargs)

            # Ensure output fits in a tweet
            if len(output) > max_length:
                output = output[: max_length - 3] + "..."

            return output

        except Exception as e:
            console.print(f"[red]Error generating tweet: {str(e)}[/red]")
            return f"Error generating tweet: {str(e)}"

    def batch_generate(
        self,
        prompts: list[str],
        vector_name: Optional[str] = None,
        coeff: float = 1.0,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generate multiple responses in batch.

        Args:
            prompts: List of input prompts
            vector_name: Name of vector to use (if any)
            coeff: Control strength
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        if vector_name is not None:
            self.controllable_model.set_control(self.vectors[vector_name], coeff=coeff)
        else:
            self.controllable_model.reset()

        # Format prompts using Llama chat format with system message
        formatted_prompts = [message_template(p.content) for p in prompts]

        # Tokenize all prompts
        inputs = self.tokenizer(
            formatted_prompts, padding=True, return_tensors="pt"
        ).to(self.model.device)

        # Generate
        outputs = self.controllable_model.generate(
            **inputs,
            max_new_tokens=kwargs.pop("max_new_tokens", 50),
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        # Decode and clean responses
        responses = []
        for output, prompt in zip(outputs, prompts):
            response = self.tokenizer.decode(output, skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt) :].strip()
            responses.append(response.strip())

        return responses
