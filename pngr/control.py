import dataclasses
from typing import Any, Callable, Iterable, TYPE_CHECKING
import warnings

import torch
from transformers import PretrainedConfig, PreTrainedModel

from pngr.ControlLayer import ControlLayer

if TYPE_CHECKING:
    from .ControlVector import ControlVector


class ControllableModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after.**

    A wrapped language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(self, model: PreTrainedModel, layer_ids: Iterable[int]):
        """
        **This mutates the wrapped `model`! Be careful using `model` after.**

        Build a new ControlModel around a model instance, initializing control on
        the layers specified in `layer_ids`.
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


@dataclasses.dataclass
class BlockControlParams:
    """
    Parameters for controlling a model block.
    """

    control: torch.Tensor | None = None
    normalize: bool = False
    operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        """
        The default control parameters.
        """
        return cls()


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
