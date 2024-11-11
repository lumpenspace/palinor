import dataclasses
import warnings
from typing import Any
import pickle

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from pngr.ControllableModel import ControllableModel
from pngr.Message import DatasetEntry
from pngr.vector_readers import read_representations


@dataclasses.dataclass
class ControlVector:
    """
    A trained control vector.
    """

    model_type: str
    poles: dict[int, np.ndarray[Any, Any]]

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel | ControllableModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
        **kwargs: Any,
    ) -> "ControlVector":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model: The model to train against.
            tokenizer: The tokenizer to tokenize the dataset.
            dataset: The dataset used for training.
            **kwargs: Additional keyword arguments.
                max_batch_size (int): Maximum batch size for training (default: 32).
                method (str): Training method, pca_diff or pca_center
                    (default: `pca_diff`).

        Returns:
            ControlVector: The trained vector.
        """
        with torch.inference_mode():
            poles = read_representations(
                model,
                tokenizer,
                dataset,
                **kwargs,
            )
        return cls(model_type=model.config.model_type, poles=poles)

    def _helper_combine(
        self, other: "ControlVector", other_coeff: float
    ) -> "ControlVector":
        """Helper method for vector arithmetic operations."""
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, "
                + "this may produce unexpected results."
            )

        poles: dict[int, np.ndarray[Any, Any]] = {}
        for layer in self.poles:
            poles[layer] = self.poles[layer]
        for layer in other.poles:
            other_layer = other_coeff * other.poles[layer]
            if layer in poles:
                poles[layer] = poles[layer] + other_layer
            else:
                poles[layer] = other_layer
        return ControlVector(model_type=self.model_type, poles=poles)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ControlVector):
            return NotImplemented
        if self is other:
            return True

        if self.model_type != other.model_type:
            return False
        if self.poles.keys() != other.poles.keys():
            return False
        return all((self.poles[k] == other.poles[k]).all() for k in self.poles)

    def __add__(self, other: "ControlVector") -> "ControlVector":
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        poles = {layer: -self.poles[layer] for layer in self.poles}
        return ControlVector(model_type=self.model_type, poles=poles)

    def __mul__(self, other: int | float | np.int_ | np.float32) -> "ControlVector":
        poles = {layer: other * self.poles[layer] for layer in self.poles}
        return ControlVector(model_type=self.model_type, poles=poles)

    def __rmul__(self, other: int | float | np.int_ | np.float32) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.int_ | np.float32) -> "ControlVector":
        return self.__mul__(1 / other)

    @classmethod
    def from_file(cls, path: str) -> "ControlVector":
        """Load a control vector from a file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(model_type=data["model_type"], poles=data["poles"])

    def to_file(self, path: str) -> None:
        """Save the control vector to a file."""
        with open(path, "wb") as f:
            pickle.dump(
                {"poles": self.poles, "model_type": self.model_type},
                f,
            )
