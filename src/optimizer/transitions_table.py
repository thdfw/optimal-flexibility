'''
For every availale (state, action) pair, find the next state
using the model of the system dynamics. Match it to the 
closest available state using the distance metric.
'''

import gzip
import json
import time
from pathlib import Path
from typing import TypeAlias

from assets.base import Action, Asset, Params, State
from optimizer.settings import get_logger

logger = get_logger("transitions")

TransitionsTable: TypeAlias = dict[str, dict[str, str]]


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

    path = Path("transition_tables") / f"{asset.name}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    logger.info(f"Saved transitions table to {path} ({path.stat().st_size / 1e6:.1f} MB)")

    return data


def get_transitions_table[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> TransitionsTable:
    path = Path("transition_tables") / f"{asset.name}.json.gz"
    if not path.exists():
        return _build_transitions_table(asset)

    logger.info(f"Loading transitions table from {path}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
