"""
Utilities for reading and processing control vectors.
"""

from typing import Any, Literal, Sequence, Union, Dict
import numpy as np
import numpy.typing as npt
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .ControllableModel import ControllableModel
from .Message import Message, DatasetEntry
from rich.console import Console

console = Console()


def format_messages(messages: Sequence[Message]) -> str:
    """Format a list of messages into a single string using Llama chat format."""
    formatted: list[str] = []
    for msg in messages:
        if msg.role == "system":
            formatted.append(f"<s>[INST] <<SYS>>\n{msg.content}\n<</SYS>>\n\n")
        elif msg.role == "user":
            formatted.append(f"{msg.content} [/INST]")
        elif msg.role == "assistant":
            formatted.append(f"{msg.content} </s>")
        else:
            formatted.append(msg.content)
    return "".join(formatted)


def get_batch_hidden_states(
    model: Union[PreTrainedModel, ControllableModel],
    tokenizer: PreTrainedTokenizerBase,
    inputs: Sequence[Union[DatasetEntry, Dict[str, Any]]],
    hidden_layers: Sequence[int],
    batch_size: int = 32,
) -> Dict[int, torch.Tensor]:
    """
    Get hidden states for a batch of inputs.

    Args:
        model: The model to get hidden states from
        tokenizer: Tokenizer for the model
        inputs: Sequence of input prompts and responses
        hidden_layers: Which layers to get hidden states from
        batch_size: How many inputs to process at once

    Returns:
        Dictionary mapping layer indices to hidden states tensors
    """
    # Convert dict inputs to DatasetEntry objects if needed
    dataset_entries = [
        DatasetEntry.from_dict(entry) if isinstance(entry, dict) else entry
        for entry in inputs
    ]

    # Extract prompts from the chat format
    console.print("Preparing prompts...")
    a_prompts = []
    b_prompts = []
    for entry in dataset_entries:
        # Format messages into strings
        a_text = format_messages(entry.a)
        b_text = format_messages(entry.b)
        a_prompts.append(a_text)
        b_prompts.append(b_text)

    # Reduce batch size if on CPU
    if model.device.type == "cpu":
        batch_size = min(batch_size, 4)  # Even smaller batches for CPU
        console.print(f"Using CPU mode with batch size {batch_size}")

    tokenizer.padding_side = "left"  # LLMs need left padding
    tokenizer.pad_token = tokenizer.eos_token

    # Get token lengths for all prompts
    console.print("Calculating sequence lengths...")
    a_lengths = [len(tokenizer.encode(p)) for p in a_prompts]
    b_lengths = [len(tokenizer.encode(p)) for p in b_prompts]
    max_length = max(max(a_lengths), max(b_lengths))
    console.print(f"Max sequence length: {max_length}")

    # Initialize hidden states storage
    hidden_states = {layer_idx: [] for layer_idx in hidden_layers}

    # Process all prompts in batches
    total_batches = (len(a_prompts) + batch_size - 1) // batch_size
    console.print(f"\nProcessing {len(a_prompts)} prompts in {total_batches} batches")

    for batch_idx in range(0, len(a_prompts), batch_size):
        batch_end = min(batch_idx + batch_size, len(a_prompts))
        batch_a = a_prompts[batch_idx:batch_end]
        batch_b = b_prompts[batch_idx:batch_end]

        batch_num = (batch_idx // batch_size) + 1
        console.print(
            f"Processing batch {batch_num}/{total_batches} " f"(size: {len(batch_a)})"
        )

        # Tokenize current batch with consistent max_length
        a_tokens = tokenizer(
            batch_a,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)

        b_tokens = tokenizer(
            batch_b,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)

        try:
            with torch.no_grad():
                # Process batch
                a_output = model(
                    **a_tokens, output_hidden_states=True, return_dict=True
                )
                b_output = model(
                    **b_tokens, output_hidden_states=True, return_dict=True
                )

                # Store hidden states for each layer, converting to float32 immediately
                for layer_idx in hidden_layers:
                    a_states = a_output.hidden_states[layer_idx].to(torch.float32)
                    b_states = b_output.hidden_states[layer_idx].to(torch.float32)

                    # Debug info
                    if a_states.shape != b_states.shape:
                        msg = f"Shape mismatch in layer {layer_idx}"
                        console.print(f"[yellow]Warning: {msg}[/yellow]")
                        console.print(
                            f"A shape: {a_states.shape}, B shape: {b_states.shape}"
                        )

                    hidden_states[layer_idx].append(
                        torch.cat([a_states, b_states], dim=0).cpu()
                    )

            console.print("✓ Batch completed")

        except RuntimeError as e:
            if "out of memory" in str(e):
                console.print("[red]Out of memory, trying to recover...[/red]")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise e

        # Clear memory
        del a_tokens, b_tokens, a_output, b_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all batches for each layer
    console.print("\nCombining results...")
    final_states = {}
    for layer_idx in hidden_layers:
        final_states[layer_idx] = torch.cat(hidden_states[layer_idx], dim=0)
        del hidden_states[layer_idx]  # Free memory as we go

    return final_states


def read_representations(
    model: Union[PreTrainedModel, ControllableModel],
    tokenizer: PreTrainedTokenizerBase,
    dataset: Sequence[DatasetEntry],
    max_batch_size: int = 32,
    method: Literal["pca_diff", "pca_center"] = "pca_diff",
    **kwargs: Any,
) -> Dict[int, npt.NDArray[np.float64]]:
    """
    Read and process hidden state representations from the model.
    """
    if isinstance(model, ControllableModel):
        layers = model.layer_ids
    else:
        # Default to last 3 layers if not specified
        layers = [-1, -2, -3]

    hidden_states = get_batch_hidden_states(
        model, tokenizer, dataset, layers, max_batch_size
    )

    # Process hidden states based on method
    control_vectors = {}
    for layer, states in hidden_states.items():
        console.print(f"Processing layer {layer}: {states.shape}")
        # States are already float32, just convert to numpy
        states_np = states.cpu().numpy()

        # Get dimensions
        batch_size, seq_len, hidden_dim = states_np.shape

        if method == "pca_diff":
            # Split into A and B groups while maintaining hidden_dim
            half_batch = batch_size // 2
            a_states = states_np[:half_batch].reshape(
                -1, hidden_dim
            )  # (batch/2 * seq_len, hidden_dim)
            b_states = states_np[half_batch:].reshape(
                -1, hidden_dim
            )  # (batch/2 * seq_len, hidden_dim)

            # Compute difference between A and B examples
            diff = a_states - b_states  # Shape: (batch/2 * seq_len, hidden_dim)

            # Compute SVD on the transposed difference matrix to get hidden_dim components
            u, s, vh = np.linalg.svd(diff.T, full_matrices=False)
            # Take the first component, which will have shape (hidden_dim,)
            control_vector = u[:, 0]

        else:  # pca_center
            # Reshape to (batch * seq_len, hidden_dim)
            states_2d = states_np.reshape(-1, hidden_dim)
            # Compute SVD on the transposed states to get hidden_dim components
            u, s, vh = np.linalg.svd(states_2d.T, full_matrices=False)
            # Take the first component
            control_vector = u[:, 0]

        console.print(f"Control vector shape for layer {layer}: {control_vector.shape}")
        # Verify we got the right dimension
        assert control_vector.shape[0] == hidden_dim, (
            f"Control vector dimension mismatch: got {control_vector.shape[0]}, "
            f"expected {hidden_dim}"
        )
        control_vectors[layer] = control_vector

    return control_vectors
