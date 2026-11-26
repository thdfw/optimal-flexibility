"""A tank colder than the return-water line cannot invent a discharge.

Water at or below its return temperature gives no heat. Asking the storage for
the house load anyway must not be matched to a colder grid state that keeps
the top layer and grows the cold bottom, and the heat the water cannot give
is unmet load.
"""

import pytest

from bruce.graph_optimizer.assets.heat_pump_water_tank import (
    HeatPumpWaterTankAction,
    HeatPumpWaterTankAsset,
    HeatPumpWaterTankParams,
)

HORIZON = 2


@pytest.fixture(scope="module")
def asset() -> HeatPumpWaterTankAsset:
    params = HeatPumpWaterTankParams(
        horizon=HORIZON,
        start_unix_s=1795665600,
        site_id="hw1.isone.me.versant.keene.oak",
        elec_price_mwh=[100.0] * HORIZON,
        num_layers=9,
        storage_volume_gallons=120,
        rswt_f=[96.0] * HORIZON,
        load_kwh=[4.7] * HORIZON,
        oat_f=[23.0] * HORIZON,
        initial_top_temp=90,
        initial_middle_temp=80,
        initial_bottom_temp=70,
        initial_thermocline1=1,
        initial_thermocline2=6,
    )
    return HeatPumpWaterTankAsset(params)


def _state(asset: HeatPumpWaterTankAsset, top: float, middle: float, bottom: float, th1: int, th2: int):
    return next(
        state
        for state in asset.state_space
        if state.top_temp == top
        and state.middle_temp == middle
        and state.bottom_temp == bottom
        and state.thermocline1 == th1
        and state.thermocline2 == th2
    )


def test_a_cold_discharge_stays_near_the_heat_the_water_can_give(asset: HeatPumpWaterTankAsset) -> None:
    start = _state(asset, 90, 80, 70, 1, 6)
    action = HeatPumpWaterTankAction(heat_to_store_kwh=-4.7, index=0)
    predicted = asset.next_state(start, action, transition_variant=0)
    deliverable = asset.deliverable_discharge_kwh(start, 4.7, 0)

    assert deliverable < 0.5
    assert start.energy - predicted.energy == pytest.approx(deliverable, abs=0.05)

    heads = asset.successors(start, action, transition_variant=0)
    weighted = sum(weight * head.energy for head, weight in heads)
    assert weighted == pytest.approx(predicted.energy, abs=0.05)
    assert all(head.energy > start.energy - 1.0 for head, _ in heads)
    # Growing the 70°F bottom while the 90°F cap stays put is how the grid used to shed the load.
    assert all(
        not (head.top_temp == 90 and head.thermocline1 == 1 and head.thermocline2 < 4)
        for head, _ in heads
    )


def test_the_heat_cold_water_cannot_give_is_unmet_load_on_the_whole_edge(asset: HeatPumpWaterTankAsset) -> None:
    """Both heads carry the full undeliverable charge, so the weighted edge cost does too."""
    start = _state(asset, 90, 80, 70, 1, 6)
    action = HeatPumpWaterTankAction(heat_to_store_kwh=-4.7, index=0)
    heads = asset.successors(start, action, transition_variant=0)
    assert len(heads) == 2
    losses = asset._losses_kwh(start, 0)
    deliverable = asset.deliverable_discharge_kwh(start, 4.7, 0)
    expected = asset.params.unmet_load_penalty_weight * asset._max_heat_price_kwh * (4.7 - losses - deliverable)

    penalties = [asset.unmet_load_penalty(start, head, action, 0) for head, _ in heads]
    assert penalties == pytest.approx([expected, expected], rel=1e-6)
    edge_penalty = sum(weight * penalty for (_, weight), penalty in zip(heads, penalties, strict=True))
    assert edge_penalty == pytest.approx(expected, rel=1e-6)
    assert expected > asset._max_heat_price_kwh * 4.0


def test_a_hot_discharge_still_tracks_the_requested_energy(asset: HeatPumpWaterTankAsset) -> None:
    start = _state(asset, 140, 130, 120, 4, 7)
    action = HeatPumpWaterTankAction(heat_to_store_kwh=-2.0, index=0)
    assert asset.deliverable_discharge_kwh(start, 2.0, 0) == pytest.approx(2.0)
    assert asset.unmet_load_penalty(start, start, action, 0) == 0.0

    heads = asset.successors(start, action, transition_variant=0)
    weighted = sum(weight * head.energy for head, weight in heads)
    assert weighted == pytest.approx(start.energy - 2.0, abs=0.05)
