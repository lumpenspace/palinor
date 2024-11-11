"""
Utilities for reading and processing control vectors.
"""

from typing import Any, Literal, Sequence, Union, Dict
import math
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from rich.console import Console

from .ControllableModel import ControllableModel
from .Message import Message, DatasetEntry

console = Console()


def format_messages(messages: Sequence[Message]) -> str:
    """Format messages without special tokens for training."""
    formatted: list[str] = []
    for msg in messages:
        if msg.role == "user":
            formatted.append(msg.content)  # Just the content, no special tokens
    return " ".join(formatted)


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Message],
    device: str = "cuda",
) -> str:
    """Generate model response for a sequence of messages."""
    input_text = format_messages(messages)
    tokens = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response[len(input_text) :].strip()

    return response


def compute_attention_pooling(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Compute attention-weighted average of hidden states."""
    mask_float = attention_mask.float().unsqueeze(-1)
    weights = mask_float / mask_float.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return (hidden_states * weights).sum(dim=1)


def get_batch_hidden_states(
    model: Union[PreTrainedModel, ControllableModel],
    tokenizer: PreTrainedTokenizerBase,
    inputs: Sequence[Union[DatasetEntry, Dict[str, Any]]],
    hidden_layers: Sequence[int],
    batch_size: int = 32,
) -> Dict[int, torch.Tensor]:
    """Process inputs in batches to get hidden states."""
    device = model.device
    use_cuda = device.type == "cuda"
    dtype = torch.float32 if not use_cuda else torch.float16

    dataset_entries = [
        DatasetEntry.from_dict(entry) if isinstance(entry, dict) else entry
        for entry in inputs
    ]

    console.print("Preparing prompts and responses...")
    a_texts = [format_messages(entry.a) for entry in dataset_entries]
    b_texts = [format_messages(entry.b) for entry in dataset_entries]

    if not use_cuda:
        batch_size = min(batch_size, 4)
        console.print(f"Using CPU mode with batch size {batch_size}")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    max_length = max(
        max(len(tokenizer.encode(p)) for p in a_texts + b_texts),
        512,  # Reasonable default max length
    )
    console.print(f"Max sequence length: {max_length}")

    hidden_states = {layer_idx: [] for layer_idx in hidden_layers}

    for batch_idx in range(0, len(a_texts), batch_size):
        batch_end = min(batch_idx + batch_size, len(a_texts))
        batch_a = a_texts[batch_idx:batch_end]
        batch_b = b_texts[batch_idx:batch_end]

        try:
            tokens = tokenizer(
                batch_a + batch_b,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_cuda):
                with torch.no_grad():
                    outputs = model(
                        **tokens, output_hidden_states=True, return_dict=True
                    )

                    for layer_idx in hidden_layers:
                        states = outputs.hidden_states[layer_idx].to(dtype)
                        pooled_states: torch.Tensor = compute_attention_pooling(
                            states, tokens.attention_mask
                        )
                        hidden_states[layer_idx].append(pooled_states)

            if use_cuda:
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                console.print("[red]Out of memory, trying to recover...[/red]")
                if use_cuda:
                    torch.cuda.empty_cache()
            raise e

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
    amplification_factor: float = 3.0,
    **kwargs: Any,
) -> Dict[int, torch.Tensor]:
    device = model.device
    use_cuda = device.type == "cuda"
    dtype = torch.float32 if not use_cuda else torch.float16

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
        hidden_dim = states.shape[-1]

        states = states.to(device, dtype=dtype)

        # Split states
        a_states = states[:batch_size]
        b_states = states[batch_size:]
        
        console.print(f"Raw A states mean: {a_states.mean().item():.6f}")
        console.print(f"Raw B states mean: {b_states.mean().item():.6f}")
        
        # More aggressive standardization
        def aggressive_standardize(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            x = (x - mean) / (std + 1e-5)
            # Amplify differences
            return torch.tanh(x * 2)  # Force more distinct activations

        a_states = aggressive_standardize(a_states)
        b_states = aggressive_standardize(b_states)

        console.print(f"Standardized A mean: {a_states.mean().item():.6f}")
        console.print(f"Standardized B mean: {b_states.mean().item():.6f}")

        # Compute activation patterns more aggressively
        a_active = (a_states > 0.2).float().mean(dim=0)  # Higher threshold
        b_active = (b_states > 0.2).float().mean(dim=0)
        
        # Direct difference between states
        state_diff = (b_states.mean(dim=0) - a_states.mean(dim=0))
        
        # Compute importance more directly
        importance = state_diff.abs() * (b_active - a_active).abs()
        importance = F.softmax(importance * 10, dim=0)  # Force more contrast
        
        console.print(f"Mean importance: {importance.mean().item():.6f}")
        console.print(f"Max importance: {importance.max().item():.6f}")

        # Select top dimensions more aggressively
        k = int(hidden_dim * 0.1)  # Top 10% only
        top_values, top_indices = torch.topk(importance, k)
        
        # Create control vector with forced magnitude
        control_vector = torch.zeros(hidden_dim, device=device, dtype=dtype)
        control_vector[top_indices] = state_diff[top_indices]
        
        # Force stronger magnitudes
        magnitude = torch.abs(control_vector)
        scaled_magnitude = torch.pow(magnitude.clamp(min=1e-6), 2.0)  # Square for emphasis
        control_vector = control_vector.sign() * scaled_magnitude
        
        # Ensure non-zero magnitude
        if control_vector.abs().max() < 1e-6:
            console.print("[yellow]Warning: Low magnitude, forcing stronger values[/yellow]")
            control_vector[top_indices] = torch.sign(state_diff[top_indices]) * 0.1
        
        # Final normalization
        control_vector = F.normalize(control_vector, p=2, dim=0)
        
        console.print(f"Final vector stats:")
        console.print(f"Non-zero elements: {(control_vector != 0).sum().item()}")
        console.print(f"Max magnitude: {control_vector.abs().max().item():.6f}")
        console.print(f"Mean magnitude: {control_vector.abs().mean().item():.6f}")
        
        control_vectors[layer] = control_vector.cpu()

    return control_vectors