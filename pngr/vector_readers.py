"""
Utilities for reading and processing control vectors.
"""

from typing import Any, Literal, Sequence, Union, Dict, Optional
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from rich.console import Console
from tqdm import tqdm

from .ControllableModel import ControllableModel
from .Message import Message, DatasetEntry

console = Console()


def format_messages(messages: Sequence[Message]) -> str:
    """
    Format a list of messages using Llama chat format.
    
    Args:
        messages: Sequence of Message objects to format
        
    Returns:
        Formatted string following Llama chat conventions
    """
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


def compute_attention_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute attention-pooled representation of hidden states.
    
    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Tensor of shape (batch_size, seq_len)
        
    Returns:
        Pooled tensor of shape (batch_size, hidden_dim)
    """
    # Create attention weights from mask
    mask_float = attention_mask.float().unsqueeze(-1)
    # Normalize weights
    weights = mask_float / mask_float.sum(dim=1, keepdim=True).clamp(min=1e-9)
    # Apply weights and sum
    return (hidden_states * weights).sum(dim=1)


def get_batch_hidden_states(
    model: Union[PreTrainedModel, ControllableModel],
    tokenizer: PreTrainedTokenizerBase,
    inputs: Sequence[Union[DatasetEntry, Dict[str, Any]]],
    hidden_layers: Sequence[int],
    batch_size: int = 32,
) -> Dict[int, torch.Tensor]:
    """
    Get hidden states for batched inputs.
    
    Args:
        model: The model to extract hidden states from
        tokenizer: Tokenizer for the model
        inputs: Input sequences to process
        hidden_layers: Which layers to extract from
        batch_size: Number of inputs to process at once
        
    Returns:
        Dictionary mapping layer indices to hidden state tensors
    """
    device = model.device
    use_cuda = device.type == 'cuda'
    
    # Convert inputs to DatasetEntry objects
    dataset_entries = [
        DatasetEntry.from_dict(entry) if isinstance(entry, dict) else entry
        for entry in inputs
    ]

    console.print("Preparing prompts...")
    a_prompts = [format_messages(entry.a) for entry in dataset_entries]
    b_prompts = [format_messages(entry.b) for entry in dataset_entries]

    # Adjust batch size for available memory
    if not use_cuda:
        batch_size = min(batch_size, 4)
        console.print(f"Using CPU mode with batch size {batch_size}")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Pre-calculate sequence lengths
    console.print("Calculating sequence lengths...")
    max_length = max(
        max(len(tokenizer.encode(p)) for p in a_prompts),
        max(len(tokenizer.encode(p)) for p in b_prompts)
    )
    console.print(f"Max sequence length: {max_length}")

    # Initialize storage
    hidden_states = {layer_idx: [] for layer_idx in hidden_layers}
    total_batches = (len(a_prompts) + batch_size - 1) // batch_size

    console.print(f"\nProcessing {len(a_prompts)} prompts in {total_batches} batches")

    for batch_idx in range(0, len(a_prompts), batch_size):
        batch_end = min(batch_idx + batch_size, len(a_prompts))
        batch_a = a_prompts[batch_idx:batch_end]
        batch_b = b_prompts[batch_idx:batch_end]

        console.print(f"Processing batch {(batch_idx//batch_size)+1}/{total_batches}")

        try:
            # Process both A and B sequences together
            tokens = tokenizer(
                batch_a + batch_b,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.cuda.amp.autocast(enabled=use_cuda):
                with torch.no_grad():
                    outputs = model(**tokens, output_hidden_states=True, return_dict=True)
                    
                    for layer_idx in hidden_layers:
                        # Get and pool hidden states
                        states = outputs.hidden_states[layer_idx]
                        pooled_states = compute_attention_pooling(
                            states, 
                            tokens['attention_mask']
                        )
                        hidden_states[layer_idx].append(pooled_states)

            console.print("✓ Batch completed")

        except RuntimeError as e:
            if "out of memory" in str(e):
                console.print("[red]Out of memory, trying to recover...[/red]")
                if use_cuda:
                    torch.cuda.empty_cache()
            raise e

        # Clean up
        del tokens, outputs
        if use_cuda:
            torch.cuda.empty_cache()

    # Combine results
    console.print("\nCombining results...")
    final_states = {}
    for layer_idx in hidden_layers:
        final_states[layer_idx] = torch.cat(hidden_states[layer_idx], dim=0)
        del hidden_states[layer_idx]

    return final_states


def read_representations(
    model: Union[PreTrainedModel, ControllableModel],
    tokenizer: PreTrainedTokenizerBase,
    dataset: Sequence[DatasetEntry],
    max_batch_size: int = 32,
    method: Literal["pca_diff", "pca_center"] = "pca_diff",
    **kwargs: Any,
) -> Dict[int, torch.Tensor]:
    """
    Read and process hidden state representations.
    
    Args:
        model: Model to extract representations from
        tokenizer: Tokenizer for the model
        dataset: Dataset of contrastive examples
        max_batch_size: Maximum batch size for processing
        method: Method for computing control vectors
        **kwargs: Additional arguments
        
    Returns:
        Dictionary mapping layer indices to control vectors
    """
    device = model.device
    use_cuda = device.type == 'cuda'
    
    if isinstance(model, ControllableModel):
        layers = model.layer_ids
    else:
        layers = [-1, -2, -3]

    hidden_states = get_batch_hidden_states(
        model, tokenizer, dataset, layers, max_batch_size
    )

    control_vectors = {}
    for layer, states in hidden_states.items():
        console.print(f"Processing layer {layer}: {states.shape}")
        
        batch_size = states.shape[0] // 2
        if method == "pca_diff":
            # Compute differences between paired examples
            a_states = states[:batch_size]
            b_states = states[batch_size:]
            diff = a_states - b_states
            
            # Compute principal component using SVD
            U, S, V = torch.svd(diff.T)
            control_vector = U[:, 0]  # First principal component
            
            # Determine sign based on projections
            proj_a = torch.matmul(a_states, control_vector)
            proj_b = torch.matmul(b_states, control_vector)
            
            if (proj_a < proj_b).float().mean() > 0.5:
                control_vector = -control_vector

        else:  # pca_center
            # Center the representations
            center = (states[:batch_size] + states[batch_size:]) / 2
            centered_states = torch.cat([
                states[:batch_size] - center,
                states[batch_size:] - center
            ])
            
            # Compute principal component
            U, S, V = torch.svd(centered_states.T)
            control_vector = U[:, 0]

        console.print(f"Control vector shape for layer {layer}: {control_vector.shape}")
        control_vectors[layer] = control_vector

    return control_vectors


def project_onto_direction(H: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Project tensor H onto direction vector.
    
    Args:
        H: Tensor to project
        direction: Direction vector
        
    Returns:
        Projected tensor
    """
    mag = torch.norm(direction)
    assert not torch.isinf(mag)
    return (H @ direction) / mag


def set_device_settings(model: PreTrainedModel) -> tuple[torch.device, bool]:
    """
    Configure device settings for the model.
    
    Args:
        model: Model to configure settings for
        
    Returns:
        Tuple of (device, use_cuda)
    """
    device = model.device
    use_cuda = device.type == 'cuda'
    
    if use_cuda:
        # Enable TF32 for faster processing on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    return device, use_cuda