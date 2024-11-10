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
    """Format a list of messages into a single string."""
    formatted: list[str] = []
    for msg in messages:
        if msg.role == "system":
            formatted.append(f"System: {msg.content}")
        else:
            formatted.append(msg.content)
    return " ".join(formatted)


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

    # Process in smaller chunks to avoid memory issues
    chunk_size = min(32, len(a_prompts))
    total_chunks = (len(a_prompts) + chunk_size - 1) // chunk_size
    hidden_states = {layer_idx: [] for layer_idx in hidden_layers}

    for chunk_idx in range(3):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(a_prompts))
        chunk_a = a_prompts[start_idx:end_idx]
        chunk_b = b_prompts[start_idx:end_idx]

        console.print(f"\nProcessing chunk {chunk_idx + 1}/{total_chunks}")
        console.print(f"Tokenizing inputs {start_idx} to {end_idx}...")

        # Tokenize current chunk
        a_tokens = tokenizer(
            chunk_a,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(model.device)

        b_tokens = tokenizer(
            chunk_b,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(model.device)

        # Process batches within chunk
        total_batches = (len(chunk_a) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(0, len(chunk_a), batch_size):
                batch_end = min(i + batch_size, len(chunk_a))
                batch_a = {k: v[i:batch_end] for k, v in a_tokens.items()}
                batch_b = {k: v[i:batch_end] for k, v in b_tokens.items()}

                batch_num = i // batch_size + 1
                console.print(
                    f"Processing batch {batch_num}/{total_batches} "
                    f"(size: {batch_end - i})"
                )

                try:
                    # Process batches
                    a_output = model(
                        **batch_a, output_hidden_states=True, return_dict=True
                    )
                    b_output = model(
                        **batch_b, output_hidden_states=True, return_dict=True
                    )

                    # Store hidden states for each layer, converting to float32 immediately
                    for layer_idx in hidden_layers:
                        hidden_states[layer_idx].append(
                            torch.cat(
                                [
                                    a_output.hidden_states[layer_idx].to(torch.float32),
                                    b_output.hidden_states[layer_idx].to(torch.float32),
                                ],
                                dim=0,
                            ).cpu()  # Move to CPU after float32 conversion
                        )

                    console.print("✓ Batch completed")

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        console.print("[red]Out of memory, trying to recover...[/red]")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise e
                    raise e

                # Clear some memory
                del a_output, b_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Clear chunk tensors
        del a_tokens, b_tokens
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
        console.print(f"processing layer {layer}: {states.shape}")
        # States are already float32, just convert to numpy
        states_np = states.cpu().numpy()
        if method == "pca_diff":
            # Compute difference between A and B examples
            diff = states_np[::2] - states_np[1::2]
            # Use first principal component as control vector
            u, _, _ = np.linalg.svd(diff, full_matrices=False)
            control_vectors[layer] = u[:, 0]
        else:  # pca_center
            # Use first principal component directly
            u, _, _ = np.linalg.svd(states_np, full_matrices=False)
            control_vectors[layer] = u[:, 0]

    return control_vectors
