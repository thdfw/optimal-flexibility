'''
For every availale (state, action) pair, find the next state
using the model of the system dynamics. Match it to the 
closest available state using the distance metric.
'''

import gzip
import json
import time
from pathlib import Path

from assets.base import Action, Asset, Params, State


def _build_transitions_table[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> dict[tuple[S, A], S]:
    transitions: dict[tuple[S, A], S] = {}
    for action in asset.action_space:
        st = time.time()
        print(f"Computing all transitions for action: {action}")
        for state in asset.state_space:
            next_state = asset.next_state(state, action)
            closest_state = asset.closest_state(next_state)
            transitions[(state, action)] = closest_state
        print(f"Done in {round(time.time() - st)} seconds")

    path = Path("transition_tables") / f"{asset.name}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, str]] = {}
    for (state, action), next_state in transitions.items():
        data.setdefault(action.to_key(), {})[state.to_key()] = next_state.to_key()
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Saved transitions table to {path} ({path.stat().st_size / 1e6:.1f} MB)")

    return transitions


def get_transitions_table[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> dict[tuple[S, A], S]:
    path = Path("transition_tables") / f"{asset.name}.json.gz"
    if not path.exists():
        return _build_transitions_table(asset)

    actions_by_key = {a.to_key(): a for a in asset.action_space}
    states_by_key = {s.to_key(): s for s in asset.state_space}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data: dict[str, dict[str, str]] = json.load(f)

    transitions: dict[tuple[S, A], S] = {}
    for action_key, state_map in data.items():
        action = actions_by_key[action_key]
        for state_key, next_state_key in state_map.items():
            transitions[(states_by_key[state_key], action)] = states_by_key[next_state_key]
    return transitions
