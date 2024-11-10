"""Parameters for controlling model blocks."""

import dataclasses
from typing import Callable

import torch


@dataclasses.dataclass
class BlockControlParams:
    """
    Parameters for controlling a model block.

    Attributes:
        control: The control tensor to apply
        normalize: Whether to normalize activations after control
        operator: Function to combine base output and control
    """

    control: torch.Tensor | None = None
    normalize: bool = False
    operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        """Create default control parameters."""
        return cls()
