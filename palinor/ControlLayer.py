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
        """Set the control parameters."""
        self.params = params

    def reset(self) -> None:
        """Reset the control parameters to the default."""
        self.set_control(BlockControlParams.default())

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the model block, with stronger control applied."""
        output = self.block(*args, **kwargs)
        control = self.params.control

        if control is None:
            return output

        # Get the hidden states to modify
        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output

        try:
            # Critical: Ensure control tensor matches hidden state dimensions
            control = control.to(modified.device)
            if len(control.shape) == 1:
                control = control.reshape(1, 1, -1)

            # Broadcasting validation
            batch_size, seq_len, hidden_dim = modified.shape
            if control.shape[0] == 1:
                control = control.expand(batch_size, -1, -1)
            if control.shape[1] == 1:
                control = control.expand(-1, seq_len, -1)

            # Handle padding tokens
            if "position_ids" in kwargs:
                pos = kwargs["position_ids"]
                zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
                col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
                mask = (col_indices >= zero_indices).float().reshape(batch_size, seq_len, 1)
                mask = mask.to(modified.dtype).to(modified.device)
            else:
                mask = torch.ones_like(modified[..., :1])

            # Apply control with stronger effect
            if self.params.normalize:
                # Apply control then renormalize more aggressively
                modified = self.params.operator(modified, control * mask * 1.5)  # Boost effect
                norm_pre = torch.norm(modified, dim=-1, keepdim=True)
                norm_post = torch.norm(modified, dim=-1, keepdim=True)
                modified = modified / norm_post * norm_pre
            else:
                # Apply stronger control without normalization
                modified = self.params.operator(modified, control * mask * 1.5)

            # Return the modified output in the correct format
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        except Exception as e:
            print(f"Error in ControlLayer forward pass: {str(e)}")
            return output  # Fallback to original output on error