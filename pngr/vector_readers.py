"""
Read the representations from a model.
"""

import dataclasses
from typing import Any, Callable, Iterable, Literal

import numpy as np
import torch
import tqdm
from sklearn.decomposition import PCA
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import EncodingFast

from .control import ControllableModel, model_layer_list

ExtractMethod = Literal["pca_diff", "pca_center", "umap"]


@dataclasses.dataclass
class DatasetEntry:
    """
    A and B are strings that are contrastive examples.
    """

    a: str
    b: str


def read_representations(
    model: "PreTrainedModel | ControllableModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    hidden_layers: Iterable[int] | None = None,
    batch_size: int = 32,
    method: ExtractMethod = "pca_diff",
    transform_hiddens: (
        Callable[
            [dict[int, np.ndarray[Any, Any]]],
            dict[int, np.ndarray[Any, Any]],
        ]
        | None
    ) = None,
) -> dict[int, np.ndarray[Any, Any]]:
    """
    Extract the representations based on the contrast dataset.
    """
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    # normalize the layer indexes if they're b
    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    # the order is [a, b, a, b, ...]
    train_strs = [s for ex in inputs for s in (ex.a, ex.b)]

    hidden_states = get_batch_hidden_states(
        model, tokenizer, train_strs, hidden_layers, batch_size
    )

    if transform_hiddens is not None:
        hidden_states: dict[int, np.ndarray[Any, Any]] = transform_hiddens(
            hidden_states
        )

    # get poles for each layer using PCA
    poles: dict[int, np.ndarray[Any, Any]] = {}
    for layer in tqdm.tqdm(hidden_layers):
        h = hidden_states[layer]
        assert h.shape[0] == len(inputs) * 2

        if method == "pca_diff":
            train = h[::2] - h[1::2]
        elif method == "pca_center":
            center = (h[::2] + h[1::2]) / 2
            train = h
            train[::2] -= center
            train[1::2] -= center
        elif method == "umap":
            train = h
        else:
            raise ValueError("unknown method " + method)

        # shape (1, n_features)
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        # shape (n_features,)
        poles[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)
        # calculate sign
        projected_hiddens = project_onto_direction(h, poles[layer])

        # order is [a, b, a, b, ...]
        a_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        a_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        # if the mean of the a's is greater than the mean of the b's, invert the pole§
        if a_smaller_mean > a_larger_mean:  # type: ignore
            poles[layer] *= -1

    return poles


def get_batch_hidden_states(
    model: PreTrainedModel | ControllableModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
) -> dict[int, np.ndarray[Any, Any]]:
    """
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token.

    Returns a dictionary with a single entry for each `hidden_layer` with a numpy array
    of shape `(n_inputs, hidden_dim)`.
    """
    batched_inputs: list[list[str]] = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}
    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs):
            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt")
            encoded_batch = encoded_batch.to(model.device)
            out = model(**encoded_batch, output_hidden_states=True)
            attention_mask: Any | EncodingFast = encoded_batch["attention_mask"]
            for i in range(len(batch)):
                last_non_padding_index = (
                    attention_mask[i].nonzero(as_tuple=True)[0][-1].item()
                )
                for layer in hidden_layers:
                    hidden_idx = layer + 1 if layer >= 0 else layer
                    hidden_state = (
                        out.hidden_states[hidden_idx][i][last_non_padding_index]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    hidden_states[layer].append(hidden_state)
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}


def project_onto_direction(
    H: np.ndarray[Any, Any], direction: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag: np.floating[Any] = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag
