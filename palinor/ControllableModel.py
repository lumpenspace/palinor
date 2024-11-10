"""
Provides the ControllableModel class for wrapping language models with control capabilities.
"""

from typing import Any, Iterable, TYPE_CHECKING
import warnings

import torch
from transformers import PreTrainedModel, PretrainedConfig

from palinor.BlockControlParams import BlockControlParams
from palinor.ControlLayer import ControlLayer


if TYPE_CHECKING:
    from .ControlVector import ControlVector


class ControllableModel(torch.nn.Module):
    """
    A wrapped language model that can have controls set on its layers.

    **Warning**: This mutates the wrapped `model`! Be careful using `model` after wrapping.

    The ControllableModel allows dynamic injection of control vectors into specific layers
    of a language model, enabling fine-grained control over the model's behavior.
    """

    def __init__(self, model: PreTrainedModel, layer_ids: Iterable[int]):
        """
        Initialize a new ControllableModel by wrapping an existing model.

        Args:
            model: The language model to wrap
            layer_ids: Which layers to enable control on. Negative indices count from end.

        **Warning**: This mutates the wrapped `model`! Be careful using `model` after wrapping.
        """
        super().__init__()
        self.model = model

        layers = model_layer_list(model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]
        for layer_id in layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, ControlLayer):
                layers[layer_id] = ControlLayer(layer)
            else:
                warnings.warn(
                    "Trying to rewrap a wrapped model! Probably not what you want! "
                    "Try calling .unwrap first."
                )

    @property
    def config(self) -> PretrainedConfig:
        """Get the model's configuration."""
        return self.model.config

    @property
    def device(self) -> torch.device:
        """Get the model's device."""
        return self.model.device

    def unwrap(self) -> PreTrainedModel:
        """
        Remove control wrappers and return the original model.

        After using this method, `set_control` and `reset` will not work.
        """
        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layers[layer_id] = layers[layer_id].block
        return self.model

    def set_control(
        self, control: "ControlVector", coeff: float = 1.0, **kwargs: Any
    ) -> None:
        """
        Set a control vector for the controlled layers.

        Args:
            control: The control vector to apply
            coeff: Strength of control (negative inverts the control, e.g. happy→sad)
            **kwargs: Additional control parameters
                normalize: Rescale activations after control (default: False)
                operator: How to combine base output and control (default: +)
        """
        raw_control = {}
        for layer_id in self.layer_ids:
            raw_control[layer_id] = torch.tensor(coeff * control.poles[layer_id]).to(
                self.model.device, dtype=self.model.dtype
            )
        self.set_raw_control(raw_control, **kwargs)

    def reset(self) -> None:
        """Reset all layer controls, returning the model to base behavior."""
        self.set_raw_control(None)

    def set_raw_control(
        self, control: dict[int, torch.Tensor] | None, **kwargs: Any
    ) -> None:
        """
        Set or remove raw control parameters for controlled layers.

        Args:
            control: Dict mapping layer IDs to control tensors, or None to reset
            **kwargs: Additional control parameters
                normalize: Rescale activations after control (default: False)
                operator: How to combine base output and control (default: +)
        """
        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer: ControlLayer = layers[layer_id]  # type: ignore
            if control is None:
                layer.reset()
            else:
                layer.set_control(BlockControlParams(control[layer_id], **kwargs))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the model, with control applied."""
        return self.model.forward(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate output from the model, with control applied."""
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the model, with control applied."""
        return self.model(*args, **kwargs)


def model_layer_list(
    model: "ControllableModel | PreTrainedModel",
) -> torch.nn.ModuleList:
    """
    Get the list of transformer layers from a model.

    Args:
        model: The model to extract layers from

    Returns:
        ModuleList containing the model's transformer layers

    Raises:
        ValueError: If the model architecture is not recognized
    """
    if isinstance(model, ControllableModel):
        model = model.model

    if hasattr(model, "model"):  # mistral-like
        return model.model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")