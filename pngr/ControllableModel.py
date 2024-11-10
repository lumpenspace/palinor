"""
Wrap a transformer layer with control logic.
"""
from typing import Any, Tuple, Union
import warnings

import torch
from transformers import PreTrainedModel

from pngr.BlockControlParams import BlockControlParams


class ControlLayer(torch.nn.Module):
    """
    A wrapper around a transformer layer that can apply control vectors.
    """

    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.block = block
        self.control_params: BlockControlParams | None = None

    def set_control(self, params: BlockControlParams) -> None:
        """
        Set the control parameters to use for this layer.
        """
        self.control_params = params

    def reset(self) -> None:
        """
        Reset the control tensor for this layer.
        """
        self.control_params = None

    def forward(
        self,
        *args: Any,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Union[Any, Tuple[Union[Any, torch.Tensor], ...], torch.Tensor]:
        """
        Forward pass through the layer, applying control if set.
        """
        if len(args) != 1:
            warnings.warn(
                f"ControlLayer got {len(args)} args, expected 1. Control might not work."
            )
        out = self.block(*args, output_attentions=output_attentions, **kwargs)
        if self.control_params is not None:
            if isinstance(out, tuple):
                hidden_states = out[0]
            else:
                hidden_states = out

            # this will modify hidden_states in-place
            self.control_params.apply_to_hidden_states(hidden_states)

        return out

    def __getattr__(self, name: str) -> Any:
        """
        Forward any unknown attributes to the wrapped block.
        """
        return getattr(self.block, name)
                warnings.warn(
                    "Trying to rewrap a wrapped model! Probably not what you want! "
                    + "Try calling .unwrap first."
                )

    @property
    def config(self) -> PretrainedConfig:
        """
        The model's configuration.
        """
        return self.model.config

    @property
    def device(self) -> torch.device:
        """
        The model's device.
        """
        return self.model.device

    def unwrap(self) -> PreTrainedModel:
        """
        Removes the mutations done to the wrapped model and returns it.
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
        Set a `ControlVector` for the layers this ControlModel handles, with a strength given
        by `coeff`. (b `coeff` values invert the control vector, e.g. happiness→sadness.)
        `coeff` defaults to `1.0`.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale
          the activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and
          control (default: +)
        """

        raw_control = {}
        for layer_id in self.layer_ids:
            raw_control[layer_id] = torch.tensor(coeff * control.poles[layer_id]).to(
                self.model.device, dtype=self.model.dtype
            )
        self.set_raw_control(raw_control, **kwargs)

    def reset(self) -> None:
        """
        Resets the control for all layer_ids, returning the model to base behavior.
        """
        self.set_raw_control(None)

    def set_raw_control(
        self, control: dict[int, torch.Tensor] | None, **kwargs: Any
    ) -> None:
        """
        Set or remove control parameters to the layers this ControlModel handles.
        The keys of `control` should be equal to or a superset of the `layer_ids` passed to
        __init__.
        Only those layers will be controlled, any others in `control` will be ignored.

        Passing `control=None` will reset the control tensor for all layer_ids, making the
        model act like a non-control model.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale
          the activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and
          control (default: +)
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer: ControlLayer = layers[layer_id]  # type: ignore
            if control is None:
                layer.reset()
            else:
                layer.set_control(BlockControlParams(control[layer_id], **kwargs))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass through the model, with control applied.
        """
        return self.model.forward(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """
        Generate output from the model, with control applied.
        """
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the model, with control applied.
        """
        return self.model(*args, **kwargs)


def model_layer_list(model: ControllableModel | PreTrainedModel) -> torch.nn.ModuleList:
    """
    Get the list of layers from a model.
    """

    if isinstance(model, ControllableModel):
        model = model.model

    if hasattr(model, "model"):  # mistral-like
        return model.model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")
