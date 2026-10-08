'''
For every availale (state, action) pair, find the next state
using the model of the system dynamics. Match it to the 
closest available state using the distance metric.
'''

import os
import time
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


def _transition_matrix_path[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> Path:
    filename = asset.transition_matrix_params().cache_filename(asset.name)
    return config_dir() / filename


def _expected_matrix_shape[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> tuple[int, ...]:
    n_actions = len(asset.action_space)
    n_states = len(asset.state_space)
    n_variants = asset.transition_matrix_variant_count()
    if n_variants <= 1:
        return (n_actions, n_states)
    return (n_actions, n_states, n_variants)


def _build_transition_matrix[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> np.ndarray:
    shape = _expected_matrix_shape(asset)
    matrix = np.empty(shape, dtype=np.int32)
    n_variants = asset.transition_matrix_variant_count()
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
                next_state = asset.next_state(state, action, transition_variant=variant)
                closest = asset.closest_state(next_state)
                if len(shape) == 2:
                    matrix[action_i, state.index] = closest.index
                else:
                    matrix[action_i, state.index, variant] = closest.index
        if len(shape) == 3 and not asset.action_uses_transition_variants(action):
            matrix[action_i, :, 1:] = matrix[action_i, :, :1]
        logger.info(f"Done in {round(time.time() - st)} seconds")

    path = _transition_matrix_path(asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, next_state_index=matrix)
    logger.info(f"Saved transition matrix to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return matrix


def get_transition_matrix[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> np.ndarray:
    """Load or build transition matrix -> next ``state.index`` (2D or 3D with transition variants)."""
    path = _transition_matrix_path(asset)
    expected_shape = _expected_matrix_shape(asset)

    if path.exists():
        try:
            with np.load(path) as data:
                matrix = np.asarray(data["next_state_index"], dtype=np.int32)
            if tuple(matrix.shape) != expected_shape:
                raise ValueError(
                    f"Transition matrix shape {matrix.shape} != expected {expected_shape}"
                )
            logger.info(f"Loading transition matrix from {path}")
            return matrix
        except (ValueError, KeyError, OSError) as exc:
            logger.warning(f"Could not load transition matrix ({exc}); rebuilding")

    if os.environ.get("CAN_COMPUTE_TRANSITION_MATRICES", "false").strip().lower() != "true":
        raise RuntimeError(
            f"No cached transition matrix at {path}; set CAN_COMPUTE_TRANSITION_MATRICES=true to build it."
        )
    return _build_transition_matrix(asset)
