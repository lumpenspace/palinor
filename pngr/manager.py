"""
Manager class for pngr operations.
"""

import os
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from huggingface_hub import login

from .ControllableModel import ControllableModel
from .ControlVector import ControlVector
from . import create_dataset


class PngrManager:
    """
    Manager class for pngr operations.

    Handles model loading, vector training and storage, and inference operations.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        layer_ids: Optional[list[int]] = None,
    ):
        """
        Initialize the manager.

        Args:
            model_name: Name of the model on HuggingFace
            cache_dir: Directory to store models and vectors (default: ~/.pngr)
            hf_token: HuggingFace token for gated models
            layer_ids: Which layers to control (default: [-1, -2, -3])
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.pngr"))
        self.vectors_dir = self.cache_dir / "vectors" / model_name.replace("/", "_")
        self.models_dir = self.cache_dir / "models"

        # Create cache directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Set up HF token if provided
        if hf_token:
            login(token=hf_token)

        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=str(self.models_dir)
        )

        # Create controllable model
        self.layer_ids = layer_ids or [-1, -2, -3]
        self.controllable_model = ControllableModel(
            self.model, layer_ids=self.layer_ids
        )

        # Load existing vectors
        self.vectors = self._load_existing_vectors()

    def _load_model(self) -> PreTrainedModel:
        """Load or download the model."""
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.models_dir),
            torch_dtype=torch.float16,
            device_map="auto",
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

        # Train vector
        vector = ControlVector.train(
            model=self.controllable_model,
            tokenizer=self.tokenizer,
            dataset=prompts,
            **kwargs,
        )

        # Save vector
        vector_path = self.vectors_dir / f"{name}.pkl"
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
        # Set up control if requested
        if vector_name is not None:
            if vector_name not in self.vectors:
                raise ValueError(
                    f"Vector {vector_name} not found. Available vectors: "
                    f"{list(self.vectors.keys())}"
                )
            self.controllable_model.set_control(self.vectors[vector_name], coeff=coeff)
        else:
            self.controllable_model.reset()

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.controllable_model.generate(
            **inputs, max_new_tokens=kwargs.pop("max_new_tokens", 50), **kwargs
        )

        return self.tokenizer.decode(outputs[0])

    def list_vectors(self) -> list[str]:
        """Get names of available vectors."""
        return list(self.vectors.keys())
