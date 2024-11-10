import torch


import dataclasses
from typing import Callable


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
