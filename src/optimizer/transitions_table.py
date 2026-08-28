'''
For every availale (state, action) pair, find the next state
using the model of the system dynamics. Match it to the 
closest available state using the distance metric.
'''

import json
from pathlib import Path

from assets.base import Action, Asset, Params, State


def _transitions_path(asset: Asset) -> Path:
    return Path("transition_tables") / f"{asset.name}.json"


def _load_transitions_table[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> dict[tuple[S, A], S]:
    path = _transitions_path(asset)
    print(f"Loading transitions table from {path}")
    state_type = type(asset.state_space[0])
    action_type = type(asset.action_space[0])
    data = json.loads(path.read_text())
    transitions: dict[tuple[S, A], S] = {}
    for entry in data:
        state = state_type.model_validate(entry["state"])
        action = action_type.model_validate(entry["action"])
        next_state = state_type.model_validate(entry["next_state"])
        transitions[(state, action)] = next_state
    return transitions


def _save_transitions_table[S: State, A: Action](
    transitions: dict[tuple[S, A], S], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "state": state.model_dump(),
            "action": action.model_dump(),
            "next_state": next_state.model_dump(),
        }
        for (state, action), next_state in transitions.items()
    ]
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved transitions table to {path}")


def get_transitions_table[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> dict[tuple[S, A], S]:
    path = _transitions_path(asset)
    if path.exists():
        return _load_transitions_table(asset)
    return _build_transitions_table(asset)


def _build_transitions_table[S: State, A: Action, P: Params](asset: Asset[S, A, P]) -> dict[tuple[S, A], S]:
    transitions: dict[tuple[S, A], S] = {}
    for action in asset.action_space:
        print(f"Computing transitions for action: {action}")
        for state in asset.state_space:
            next_state = asset.next_state(state, action)
            closest_state = asset.closest_state(next_state)
            transitions[(state, action)] = closest_state

    path = _transitions_path(asset)
    _save_transitions_table(transitions, path)
    return transitions
