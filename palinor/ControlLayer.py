from typing import Any

import torch

from .BlockControlParams import BlockControlParams


class ControlLayer(torch.nn.Module):
    """
    A wrapper around a model block that applies control.
    """

    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.block: torch.nn.Module = block
        self.params: BlockControlParams = BlockControlParams.default()

    def set_control(self, params: BlockControlParams) -> None:
        """
        Set the control parameters.
        """
        self.params = params

    def reset(self) -> None:
        """
        Reset the control parameters to the default.
        """
        self.set_control(BlockControlParams.default())

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass through the model block, with control applied.
        """
        output = self.block(*args, **kwargs)

        control = self.params.control

        if control is None:
            return output

        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output

        # Reshape control for broadcasting
        control = control.to(modified.device)
        if len(control.shape) == 1:
            # Reshape from (hidden_dim,) to (1, 1, hidden_dim)
            control = control.reshape(1, 1, -1)

        # Get target shape for broadcasting
        batch_size, seq_len, hidden_dim = modified.shape

        # Expand control to match all dimensions
        if control.shape[0] == 1:
            control = control.expand(batch_size, -1, -1)
        if control.shape[1] == 1:
            control = control.expand(-1, seq_len, -1)

        # Verify shapes match
        assert control.shape[-1] == hidden_dim, (
            f"Hidden dimension mismatch: control {control.shape[-1]} "
            f"vs modified {hidden_dim}"
        )
        assert control.shape[0] == batch_size, (
            f"Batch dimension mismatch: control {control.shape[0]} "
            f"vs modified {batch_size}"
        )
        assert control.shape[1] == seq_len, (
            f"Sequence dimension mismatch: control {control.shape[1]} "
            f"vs modified {seq_len}"
        )

        # Store original norm for later
        norm_pre = torch.norm(modified, dim=-1, keepdim=True)

        # Handle padding tokens in activation addition
        if "position_ids" in kwargs:
            pos = kwargs["position_ids"]
            zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
            col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
            mask = (col_indices >= zero_indices).float().reshape(batch_size, seq_len, 1)
            mask = mask.to(modified.dtype).to(modified.device)
        else:
            mask = torch.ones_like(modified[..., :1])

        # Apply control
        modified = self.params.operator(modified, control * mask)

        # Renormalize if requested
        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output
