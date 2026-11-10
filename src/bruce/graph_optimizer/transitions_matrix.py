'''
For every availale (state, action) pair, find the grid states it leads to
and their weights using the model of the system dynamics (``Asset.successors``).
'''

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bruce.graph_optimizer.assets.base import Action, Asset, Params, State
from bruce.graph_optimizer.settings import config_dir, get_logger

logger = get_logger("transitions")

_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            _key, _val = _key.strip(), _val.strip().strip('"').strip("'")
            if _key and _key not in os.environ:
                os.environ[_key] = _val


@dataclass(frozen=True)
class TransitionMatrix:
    """Successor state indices and weights, indexed [action, state(, variant), k].
    Entries past an entry's own number of successors have weight 0.
    """

    next_state_index: np.ndarray
    next_state_weight: np.ndarray

    def successors(self, action_index: int, state_index: int, variant: int = 0) -> list[tuple[int, float]]:
        if self.next_state_index.ndim == 3:
            indices = self.next_state_index[action_index, state_index]
            weights = self.next_state_weight[action_index, state_index]
        else:
            indices = self.next_state_index[action_index, state_index, variant]
            weights = self.next_state_weight[action_index, state_index, variant]
        return [(int(i), float(w)) for i, w in zip(indices, weights, strict=True) if w > 0]


def _transition_matrix_path[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> Path:
    filename = asset.transition_matrix_params().cache_filename(asset.name)
    return config_dir() / filename


def _expected_matrix_shape[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> tuple[int, ...]:
    """Shape without the trailing successor axis."""
    n_actions = len(asset.action_space)
    n_states = len(asset.state_space)
    n_variants = asset.transition_matrix_variant_count()
    if n_variants <= 1:
        return (n_actions, n_states)
    return (n_actions, n_states, n_variants)


def _build_transition_matrix[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> TransitionMatrix:
    shape = _expected_matrix_shape(asset)
    logger.info(f"Building transition matrix with shape: {shape}")
    n_variants = asset.transition_matrix_variant_count()
    entries: dict[tuple[int, ...], list[tuple[int, float]]] = {}
    for action in asset.action_space:
        st = time.time()
        logger.info(f"Computing all transitions for action: {action}")
        action_i = action.index
        variant_range = (
            range(n_variants)
            if asset.action_uses_transition_variants(action)
            else range(1)
        )
        for variant in variant_range:
            for state in asset.state_space:
                successors = [
                    (successor.index, weight)
                    for successor, weight in asset.successors(state, action, transition_variant=variant)
                ]
                key = (action_i, state.index) if len(shape) == 2 else (action_i, state.index, variant)
                entries[key] = successors
        if len(shape) == 3 and not asset.action_uses_transition_variants(action):
            for state in asset.state_space:
                for variant in range(1, n_variants):
                    entries[(action_i, state.index, variant)] = entries[(action_i, state.index, 0)]
        logger.info(f"Done in {round(time.time() - st)} seconds")

    k = max(len(successors) for successors in entries.values())
    indices = np.zeros((*shape, k), dtype=np.int32)
    weights = np.zeros((*shape, k), dtype=np.float32)
    for key, successors in entries.items():
        for j, (index, weight) in enumerate(successors):
            indices[key + (j,)] = index
            weights[key + (j,)] = weight

    path = _transition_matrix_path(asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, next_state_index=indices, next_state_weight=weights)
    logger.info(f"Saved transition matrix to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return TransitionMatrix(indices, weights)


def get_transition_matrix[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> TransitionMatrix:
    """Load or build the transition matrix: successor ``state.index`` values and their weights."""
    path = _transition_matrix_path(asset)
    expected_shape = _expected_matrix_shape(asset)

    if path.exists():
        try:
            with np.load(path) as data:
                indices = np.asarray(data["next_state_index"], dtype=np.int32)
                weights = np.asarray(data["next_state_weight"], dtype=np.float32)
            if tuple(indices.shape[:-1]) != expected_shape or indices.shape != weights.shape:
                raise ValueError(
                    f"Transition matrix shape {indices.shape} != expected {expected_shape} + (k,)"
                )
            logger.info(f"Loading transition matrix from {path}")
            return TransitionMatrix(indices, weights)
        except (ValueError, KeyError, OSError) as exc:
            logger.warning(f"Could not load transition matrix ({exc}); rebuilding")

    if os.environ.get("CAN_COMPUTE_TRANSITION_MATRICES", "false").strip().lower() != "true":
        raise RuntimeError(
            f"No cached transition matrix at {path}; set CAN_COMPUTE_TRANSITION_MATRICES=true to build it."
        )
    return _build_transition_matrix(asset)
