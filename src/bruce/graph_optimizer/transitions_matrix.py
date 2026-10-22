'''
For every availale (state, action) pair, find the next state
using the model of the system dynamics. Match it to the 
closest available state using the distance metric.
'''

import math
import os
import time
from pathlib import Path

import numpy as np

from bruce.graph_optimizer.assets.base import Action, Asset, Params, State
from bruce.graph_optimizer.settings import config_dir, get_logger

logger = get_logger("transitions")

_UNCOMPUTED = np.int32(-1)

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


def _load_matrix_for_build(
    path: Path, expected_shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load cache into buffers sized for ``expected_shape``, or None if not resumable."""
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            stored = np.asarray(data["next_state_index"], dtype=np.int32)
            stored_shape = tuple(stored.shape)
            resumable = (
                len(stored_shape) == len(expected_shape)
                and stored_shape[1:] == expected_shape[1:]
                and stored_shape[0] <= expected_shape[0]
            )
            if not resumable:
                if stored_shape[1:] == expected_shape[1:] and stored_shape[0] > expected_shape[0]:
                    logger.warning(
                        f"Transition matrix has {stored_shape[0]} action rows but "
                        f"{expected_shape[0]} expected; rebuilding from scratch"
                    )
                else:
                    logger.warning(
                        f"Transition matrix shape {stored_shape} is not resumable for "
                        f"{expected_shape}; rebuilding from scratch"
                    )
                return None

            matrix = np.full(expected_shape, _UNCOMPUTED, dtype=np.int32)
            n_stored = stored_shape[0]
            matrix[:n_stored] = stored
            complete = np.zeros(expected_shape[0], dtype=bool)
            if "action_row_complete" in data:
                flags = np.asarray(data["action_row_complete"], dtype=bool)
                n_flags = min(flags.shape[0], expected_shape[0])
                complete[:n_flags] = flags[:n_flags]
            if stored.ndim == 2:
                row_complete = ~np.any(stored < 0, axis=1)
            else:
                row_complete = ~np.any(stored.reshape(stored.shape[0], -1) < 0, axis=1)
            complete[:n_stored] |= row_complete
    except (OSError, KeyError, ValueError) as exc:
        logger.warning(f"Could not load transition matrix for resume ({exc}); building from scratch")
        return None
    return matrix, complete


def _build_transition_matrix[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> np.ndarray:
    path = _transition_matrix_path(asset)
    shape = _expected_matrix_shape(asset)
    n_elements = int(np.prod(shape, dtype=np.int64))
    n_states = shape[1]
    bits_per_index = max(1, math.ceil(math.log2(max(n_states, 2))))
    est_mb = n_elements * bits_per_index / 8 * 1.15 / 1e6
    loaded = _load_matrix_for_build(path, shape)
    if loaded is None:
        matrix = np.full(shape, _UNCOMPUTED, dtype=np.int32)
        complete = np.zeros(shape[0], dtype=bool)
        logger.info(f"Building transition matrix with shape: {shape} (~{est_mb:.1f} MB estimated file size)")
    else:
        matrix, complete = loaded
        n_done = int(complete.sum())
        n_total = shape[0]
        if n_done:
            logger.info(
                f"Resuming transition matrix build with shape: {shape} "
                f"({n_done}/{n_total} action rows already complete, ~{est_mb:.1f} MB estimated file size)"
            )
        else:
            logger.info(f"Building transition matrix with shape: {shape} (~{est_mb:.1f} MB estimated file size)")

    n_variants = asset.transition_matrix_variant_count()
    total_actions = len(asset.action_space)
    for actions_computed, action in enumerate(asset.action_space):
        action_i = action.index
        if complete[action_i]:
            logger.info(
                f"Skipping action ({actions_computed}/{total_actions}) (cached): {action}"
            )
            continue
        st = time.time()
        logger.info(f"Computing all transitions for action ({actions_computed}/{total_actions}): {action}")
        variants = asset.transition_variants_to_precompute(action)
        for variant in variants:
            for state in asset.state_space:
                next_state = asset.next_state(state, action, transition_variant=variant)
                closest = asset.closest_state(next_state)
                if len(shape) == 2:
                    matrix[action_i, state.index] = closest.index
                else:
                    matrix[action_i, state.index, variant] = closest.index
        if len(shape) == 3:
            fill_from = variants[0] if variants else 0
            for v in range(n_variants):
                if v not in variants:
                    matrix[action_i, :, v] = matrix[action_i, :, fill_from]
        complete[action_i] = True
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, next_state_index=matrix, action_row_complete=complete)
        logger.info(f"Done in {round(time.time() - st)} seconds")

    logger.info(f"Saved transition matrix to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return matrix


def get_transition_matrix[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> np.ndarray:
    """Load or build transition matrix -> next ``state.index`` (2D or 3D with transition variants)."""
    path = _transition_matrix_path(asset)
    expected_shape = _expected_matrix_shape(asset)
    can_compute = os.environ.get("CAN_COMPUTE_TRANSITION_MATRICES", "false").strip().lower() == "true"

    if path.exists():
        loaded = _load_matrix_for_build(path, expected_shape)
        if loaded is not None:
            matrix, complete = loaded
            if complete.all():
                logger.info(f"Loading transition matrix from {path}")
                return matrix
            if can_compute:
                return _build_transition_matrix(asset)
            n_done = int(complete.sum())
            raise RuntimeError(
                f"Partial transition matrix at {path} "
                f"({n_done}/{expected_shape[0]} action rows complete); "
                "set CAN_COMPUTE_TRANSITION_MATRICES=true to continue building."
            )

    if not can_compute:
        raise RuntimeError(
            f"No cached transition matrix at {path}; set CAN_COMPUTE_TRANSITION_MATRICES=true to build it."
        )
    return _build_transition_matrix(asset)
