"""Heat pump mode tracking: off collapses the graph to mode 1.

Hour 0 still uses hp_currently_on for turn-on time. Later hours pay half of
hp_turn_on_minutes, and every edge stays in mode 1.
"""

from bruce.graph_optimizer.assets.heat_pump_water_tank import (
    HeatPumpWaterTankAsset,
    HeatPumpWaterTankParams,
)


def _params(**overrides: object) -> HeatPumpWaterTankParams:
    horizon = 2
    values: dict[str, object] = dict(
        horizon=horizon,
        start_unix_s=1762009200,
        site_id="test",
        num_layers=1,
        elec_price_mwh=[40.0, 40.0],
        rswt_f=[120.0, 120.0],
        load_kwh=[1.0, 1.0],
        oat_f=[20.0, 20.0],
        initial_thermocline1=1,
        initial_thermocline2=1,
        hp_turn_on_minutes=12,
        hp_currently_on=False,
    )
    values.update(overrides)
    return HeatPumpWaterTankParams(**values)  # type: ignore[arg-type]


def _action_indexes(asset: HeatPumpWaterTankAsset, state_index: int, time_step: int, mode: int) -> list[int]:
    state = asset.state_space[state_index]
    return [action.index for action, _ in asset.get_available_actions(state, time_step, mode)]


def test_tracking_off_is_mode_1_and_hour_0_follows_hp_currently_on() -> None:
    tracked = HeatPumpWaterTankAsset(_params())
    cold = HeatPumpWaterTankAsset(_params(heat_pump_mode_tracking_enabled=False))
    running = HeatPumpWaterTankAsset(
        _params(heat_pump_mode_tracking_enabled=False, hp_currently_on=True)
    )

    assert tracked.get_modes() == [0, 1]
    assert tracked.initial_mode() == 0
    assert cold.get_modes() == [1]
    assert cold.initial_mode() == 1
    assert running.get_modes() == [1]
    assert running.initial_mode() == 1

    state_index = next(
        i
        for i in range(len(tracked.state_space))
        if _action_indexes(tracked, i, 0, 0) != _action_indexes(tracked, i, 0, 1)
    )

    assert cold._turn_on_minutes(0, 1) == 12
    assert cold._turn_on_minutes(1, 1) == 6
    assert running._turn_on_minutes(0, 1) == 0
    assert running._turn_on_minutes(1, 1) == 6
    assert tracked._turn_on_minutes(1, 0) == 12
    assert tracked._turn_on_minutes(1, 1) == 0

    assert _action_indexes(cold, state_index, 0, 1) == _action_indexes(tracked, state_index, 0, 0)
    assert _action_indexes(running, state_index, 0, 1) == _action_indexes(tracked, state_index, 0, 1)
    later = _action_indexes(cold, state_index, 1, 1)
    assert later != _action_indexes(tracked, state_index, 1, 0)
    assert later != _action_indexes(tracked, state_index, 1, 1)

    cold_state = cold.state_space[state_index]
    for time_step in (0, 1):
        actions = cold.get_available_actions(cold_state, time_step, 1)
        assert actions
        assert {next_mode for _, next_mode in actions} == {1}
