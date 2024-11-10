"""
Utilities for reading and processing control vectors.
"""

from typing import Any, Literal, Sequence

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pngr.Message import DatasetEntry

from .ControllableModel import ControllableModel


def get_batch_hidden_states(
    model: PreTrainedModel | ControllableModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: Sequence[DatasetEntry],
    hidden_layers: Sequence[int],
    batch_size: int,
) -> dict[int, np.ndarray[Any, Any]]:
    """
    Get hidden states for a batch of inputs.

    Args:
        model: The model to get hidden states from
        tokenizer: Tokenizer for the model
        inputs: Sequence of dataset entries
        hidden_layers: Which layers to get hidden states from
        batch_size: How many inputs to process at once

    Returns:
        Dictionary mapping layer numbers to hidden states
    """
    device = model.device
    all_hidden_states: dict[int, list[np.ndarray[Any, Any]]] = {
        layer: [] for layer in hidden_layers
    }

    # Process in batches
    for i in range(0, len(inputs), batch_size):
        batch: Sequence[DatasetEntry] = inputs[i : i + batch_size]

        # Convert to Llama format and tokenize
        a_texts, b_texts = zip(*(entry.to_llama_strings() for entry in batch))

        # Tokenize both sets
        a_tokens = tokenizer(
            list(a_texts), return_tensors="pt", padding=True, truncation=True
        ).to(device)

        b_tokens = tokenizer(
            list(b_texts), return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Get hidden states for both sets
        with torch.no_grad():
            a_output = model(**a_tokens, output_hidden_states=True, return_dict=True)
            b_output = model(**b_tokens, output_hidden_states=True, return_dict=True)

        # Extract and store hidden states
        for layer in hidden_layers:
            a_states = a_output.hidden_states[layer]
            b_states = b_output.hidden_states[layer]

            # Average over sequence length
            a_mean = a_states.mean(dim=1).cpu().numpy()
            b_mean = b_states.mean(dim=1).cpu().numpy()

            all_hidden_states[layer].extend(a_mean)
            all_hidden_states[layer].extend(b_mean)

    # Concatenate all batches
    return {
        layer: np.concatenate(states, axis=0)
        for layer, states in all_hidden_states.items()
    }


def read_representations(
    model: PreTrainedModel | ControllableModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Sequence[DatasetEntry],
    max_batch_size: int = 32,
    method: Literal["pca_diff", "pca_center"] = "pca_diff",
    **kwargs: Any,
) -> dict[int, np.ndarray[Any, Any]]:
    """
    Read and process hidden state representations from the model.

    Args:
        model: Model to get representations from
        tokenizer: Tokenizer for the model
        dataset: Dataset to process
        max_batch_size: Maximum batch size for processing
        method: Method for computing control vector ("pca_diff" or "pca_center")
        **kwargs: Additional arguments passed to get_batch_hidden_states

    Returns:
        Dictionary mapping layer numbers to control vectors
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
        if method == "pca_diff":
            # Compute difference between A and B examples
            diff = states[::2] - states[1::2]
            # Use first principal component as control vector
            u, _, _ = np.linalg.svd(diff, full_matrices=False)
            control_vectors[layer] = u[:, 0]
        else:  # pca_center
            # Use first principal component directly
            u, _, _ = np.linalg.svd(states, full_matrices=False)
            control_vectors[layer] = u[:, 0]

    return control_vectors
