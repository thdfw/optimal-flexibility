'''
For every availale (state, action) pair, find the next state
using the model of the system dynamics. Match it to the 
closest available state using the distance metric.
'''

import gzip
import json
import os
import time
from pathlib import Path
from typing import TypeAlias

from bruce.graph_optimizer.assets.base import Action, Asset, Params, State
from bruce.graph_optimizer.settings import config_dir, get_logger

logger = get_logger("transitions")

TransitionsTable: TypeAlias = dict[str, dict[str, str]]

_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            _key, _val = _key.strip(), _val.strip().strip('"').strip("'")
            if _key and _key not in os.environ:
                os.environ[_key] = _val


def _transitions_table_path[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> Path:
    filename = asset.transitions_table_params().cache_filename(asset.name)
    return config_dir() / filename


def _build_transitions_table[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> TransitionsTable:
    data: TransitionsTable = {}
    for action in asset.action_space:
        st = time.time()
        logger.info(f"Computing all transitions for action: {action}")
        action_key = action.to_key()
        state_map: dict[str, str] = {}
        for state in asset.state_space:
            next_state = asset.next_state(state, action)
            closest_state = asset.closest_state(next_state)
            state_map[state.to_key()] = closest_state.to_key()
        data[action_key] = state_map
        logger.info(f"Done in {round(time.time() - st)} seconds")

    path = _transitions_table_path(asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    logger.info(f"Saved transitions table to {path} ({path.stat().st_size / 1e6:.1f} MB)")

    return data


def get_transitions_table[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> TransitionsTable:
    path = _transitions_table_path(asset)
    if not path.exists():
        if os.environ.get("CAN_COMPUTE_TRANSITION_TABLES", "false").strip().lower() != "true":
            raise RuntimeError(f"No cached transition table at {path}; set CAN_COMPUTE_TRANSITION_TABLES=true to build it.")
        return _build_transitions_table(asset)

    logger.info(f"Loading transitions table from {path}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
