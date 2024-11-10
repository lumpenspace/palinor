import dataclasses
import warnings
from typing import Any

import numpy as np
import torch
import code
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from pngr.ControllableModel import ControllableModel
from pngr.vector_readers import DatasetEntry, read_representations


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
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_diff".

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
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, "
                + "this may produce unexpected results."
            )

        model_type = self.model_type
        poles: dict[int, np.ndarray[Any, Any]] = {}
        for layer in self.poles:
            poles[layer] = self.poles[layer]
        for layer in other.poles:
            other_layer = other_coeff * other.poles[layer]
            if layer in poles:
                poles[layer] = poles[layer] + other_layer
            else:
                poles[layer] = other_layer
        return ControlVector(model_type=model_type, poles=poles)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ControlVector):
            return NotImplemented
        if self is other:
            return True

        if self.model_type != other.model_type:
            return False
        if self.poles.keys() != other.poles.keys():
            return False
        for k, v in self.poles.items():
            if (v != other.poles[k]).any():
                return False
        return True

    def __add__(self, other: "ControlVector") -> "ControlVector":
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":

        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        poles: dict[int, np.ndarray[Any, Any]] = {}
        for layer in self.poles:
            poles[layer] = -self.poles[layer]
        return ControlVector(model_type=self.model_type, poles=poles)

    def __mul__(self, other: int | float | np.int_ | np.float32) -> "ControlVector":
        poles: dict[int, np.ndarray[Any, Any]] = {}
        for layer in self.poles:
            poles[layer] = other * self.poles[layer]
        return ControlVector(model_type=self.model_type, poles=poles)

    def __rmul__(self, other: int | float | np.int_ | np.float32) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.int_ | np.float32) -> "ControlVector":
        return self.__mul__(1 / other)
