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
    """Format messages while preserving text formatting."""
    formatted: list[str] = []
    for msg in messages:
        if msg.role == "user":
            # Preserve exact formatting, including case and punctuation
            formatted.append(f"{msg.content}")  
            
            # Debug print to verify formatting
            console.print(f"Formatted message: {msg.content[:50]}...")
            
    return "\n".join(formatted)  # Use newline separator to preserve formatting



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
    """Process inputs in batches to get hidden states and responses."""
    device = model.device
    use_cuda = device.type == "cuda"
    dtype = torch.float32 if not use_cuda else torch.float16

    dataset_entries = [
        DatasetEntry.from_dict(entry) if isinstance(entry, dict) else entry
        for entry in inputs
    ]

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    hidden_states = {layer_idx: [] for layer_idx in hidden_layers}
    
    # Process in batches
    for batch_idx in range(0, len(dataset_entries), batch_size):
        batch_entries = dataset_entries[batch_idx:batch_idx + batch_size]
        
        # Prepare both prompts and system contexts for batch
        a_prompts = [f"You are {entry.a_trait}. {entry.a[0].content}" for entry in batch_entries]
        b_prompts = [f"You are {entry.b_trait}. {entry.b[0].content}" for entry in batch_entries]
        
        # Tokenize entire batch at once
        batch_inputs = tokenizer(
            a_prompts + b_prompts,  # Combine both sets
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        # Generate responses and get hidden states in one pass
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_cuda):
            with torch.no_grad():
                outputs = model(
                    **batch_inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

                # Process hidden states for each layer
                for layer_idx in hidden_layers:
                    states = outputs.hidden_states[layer_idx].to(dtype)
                    
                    # Efficient pooling over batch
                    pooled_states = compute_attention_pooling(
                        states, 
                        batch_inputs.attention_mask
                    )
                    hidden_states[layer_idx].append(pooled_states)

        if use_cuda:
            torch.cuda.empty_cache()

    # Combine all batches
    final_states = {}
    for layer_idx in hidden_layers:
        final_states[layer_idx] = torch.cat(hidden_states[layer_idx], dim=0)

    return final_states


def read_representations(
    model: Union[PreTrainedModel, ControllableModel],
    tokenizer: PreTrainedTokenizerBase,
    dataset: Sequence[DatasetEntry],
    max_batch_size: int = 32,
    **kwargs: Any,
) -> Dict[int, torch.Tensor]:
    """Simple representation reading focusing on consistent activation differences."""
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
        batch_size = states.shape[0] // 2
        states = states.to(device, dtype=dtype)

        # Split into A and B states
        a_states = states[:batch_size]
        b_states = states[batch_size:]
        
        # Get mean activation difference
        state_diff = b_states.mean(dim=0) - a_states.mean(dim=0)
        
        # Find neurons with significant differences
        diff_threshold = state_diff.abs().mean()
        significant_neurons = (state_diff.abs() > diff_threshold)
        
        # Create control vector using only significant differences
        control_vector = torch.zeros_like(state_diff)
        control_vector[significant_neurons] = state_diff[significant_neurons]
        
        # Normalize to unit length
        control_vector = F.normalize(control_vector, p=2, dim=0)
        
        control_vectors[layer] = control_vector.cpu()

    return control_vectors