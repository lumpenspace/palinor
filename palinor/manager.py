"""
Manager class for palinor operations.
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
from .create_dataset import create_personality_prompts


dotenv.load_dotenv()

console = Console()

hf_token = os.getenv("HF_TOKEN")


class palinorManager:
    """
    Manager class for palinor operations.

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
            cache_dir: Directory to store models and vectors (default: ~/.palinor)
            hf_token: HuggingFace token for gated models
            layer_ids: Which layers to control (default: [-1, -2, -3])
            device: Device to use for computation (default: auto)
        """
        console.print(
            f"[bold]Initializing palinorManager for model [yellow]{model_name}[/yellow][/bold]"
        )
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.palinor"))
        self.vectors_dir = self.cache_dir / "vectors" / model_name.replace("/", "_")
        self.models_dir = self.cache_dir / "models"

        # Set device - prefer CUDA if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[bold]Using device: [yellow]{self.device}[/yellow][/bold]")

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
            token=hf_token,
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
        prompts = create_personality_prompts(template_path, a_adjective, b_adjective)

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
        else:
            console.print("No vector specified, using default model")

        formatted_prompt = message_template(
            prompt,
            system_message="you are a helpful AI assistant",
        )

        # Extract generation parameters
        max_new_tokens = kwargs.pop("max_new_tokens", 50)

        # Create inputs
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )

        try:
            with console.status("[bold green]Generating response...") as status:
                outputs = self.controllable_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    callback=lambda step, total: status.update(
                        f"[bold green]Generating... {int((step/total)*100)}%"
                    ),
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

    def interactive_menu(self):
        """Interactive menu for palinor operations."""
        console.print(
            """[bold magenta]
        ╔═══════════════════════════════════════╗
        ║             [cyan]P A L I N O R[/cyan]             ║
        ╚═══════════════════════════════════════╝[/bold magenta]
        """
        )

        options = {
            "1": ("Train new vector", self._interactive_train_vector),
            "2": ("Generate text", self._interactive_generate),
            "3": ("Generate tweet", self._interactive_generate_tweet),
            "4": (
                "List vectors",
                lambda: console.print(
                    f"[green]Available vectors:[/green] {self.list_vectors()}"
                ),
            ),
            "q": ("Quit", lambda: None),
        }

        while True:
            console.print("\n[yellow]Available commands:[/yellow]")
            for key, (desc, _) in options.items():
                console.print(f"[cyan]{key}[/cyan]: {desc}")

            choice = console.input(
                "\n[bold blue]Choose an option:[/bold blue] "
            ).lower()

            if choice == "q":
                break
            elif choice in options:
                options[choice][1]()
            else:
                console.print("[red]Invalid option![/red]")

    def _interactive_train_vector(self):
        """Interactive vector training."""
        name = console.input("[bold green]Vector name:[/bold green] ")
        a_trait = console.input("[bold green]First personality trait:[/bold green] ")
        b_trait = console.input("[bold green]Second personality trait:[/bold green] ")

        self.train_vector(name, a_trait, b_trait)

    def _interactive_generate(self):
        """Interactive text generation."""
        prompt = console.input("[bold green]Enter prompt:[/bold green] ")

        vectors = self.list_vectors()
        if vectors:
            use_vector = (
                console.input("[bold green]Use a vector? (y/n):[/bold green] ").lower()
                == "y"
            )
            if use_vector:
                console.print(f"[cyan]Available vectors:[/cyan] {vectors}")
                vector_name = console.input("[bold green]Vector name:[/bold green] ")
                coeff = float(
                    console.input(
                        "[bold green]Control strength (-2 to 2):[/bold green] "
                    )
                    or "1.0"
                )
                response = self.generate(prompt, vector_name, coeff)
            else:
                response = self.generate(prompt)
        else:
            response = self.generate(prompt)

        console.print(f"\n[bold cyan]Response:[/bold cyan]\n{response}")

    def _interactive_generate_tweet(self):
        """Interactive tweet generation."""
        prompt = console.input("[bold green]Enter prompt:[/bold green] ")

        vectors = self.list_vectors()
        if vectors:
            use_vector = (
                console.input("[bold green]Use a vector? (y/n):[/bold green] ").lower()
                == "y"
            )
            if use_vector:
                console.print(f"[cyan]Available vectors:[/cyan] {vectors}")
                vector_name = console.input("[bold green]Vector name:[/bold green] ")
                coeff = float(
                    console.input(
                        "[bold green]Control strength (-2 to 2):[/bold green] "
                    )
                    or "1.0"
                )
                response = self.generate_tweet(prompt, vector_name, coeff)
            else:
                response = self.generate_tweet(prompt)
        else:
            response = self.generate_tweet(prompt)

        console.print(f"\n[bold cyan]Tweet:[/bold cyan]\n{response}")