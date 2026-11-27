"""recommend: one flo.params in, one flo.recommendation out, complete over
prices and true to the recommendation's axioms. Uses the cached transition
matrix for the one-tank, nine-sensor, 120-gallon shape (the same build
test_generate_bid needs)."""

import pytest

from bruce.sema.enums import SeasonalStorageMode
from bruce.sema.types import (
    FloOptimalPath,
    FloParams,
    FloRecommendation,
    FloRerunRequest,
    FloTuningBruce,
)
from bruce.service import params_from_request, recommend

HORIZON = 12
LMP = [28.14, 28.53, 46.38, 44.84, 44.8, 32.94, 39.68, 32.07, 30.07, 27.95, 29.87, 35.63]
DIST = [50.13, 50.13, 487.63, 487.63, 487.63, 487.63, 487.63, 54.98, 54.98, 54.98, 54.98, 487.63]


def oak_request() -> FloParams:
    return FloParams(
        g_node_alias="hw1.isone.me.versant.keene.oak",
        created_ms=1762008900000,
        start_unix_s=1762009200,
        step_duration_minutes=[60] * HORIZON,
        price_usd_mwh_list=[lmp + dist for lmp, dist in zip(LMP, DIST, strict=True)],
        oat_f_list=[20.0] * HORIZON,
        load_kwh_list=[6.0] * HORIZON,
        rswt_f_list=[120.0] * HORIZON,
        weather_forecast_id="5f1c2b3a-8d4e-4f60-9a7b-1c2d3e4f5a6b",
        price_forecast_id="0b6f4c1e-5a3d-4e9b-8f27-6c1d2e3f4a5b",
        seasonal_storage_mode=SeasonalStorageMode.BufferOnly,
        tanks_modeled=1,
        gallons_per_tank=120,
        temp_sensors_per_tank=9,
        max_tank_temp_f=160.0,
        initial_top_temp_f=130.0,
        initial_middle_temp_f=120.0,
        initial_bottom_temp_f=110.0,
        initial_thermocline1=3,
        initial_thermocline2=6,
        hp_currently_on=True,
        hp_max_kw_el=9.66,
        hp_turn_on_minutes=12,
        cop_intercept=2.731,
        cop_lwt_minus_oat_coeff=-0.007403,
        price_max_usd_mwh=2000.0,
        price_min_usd_mwh=-100.0,
        tuning=FloTuningBruce(
            num_layers=9,
            storage_losses_percent=0.5,
            hp_min_kw_el=0.0,
            hp_max_kw_th=25.0,
            max_load_kw_th=20.0,
            hp_constant_lift_f=20.0,
            load_bucket_upper_kwh_list=[],
            rwt_intercept_by_bucket_list=[49.463],
            rwt_slope_by_bucket_list=[0.427],
            rwt_min_f=50.0,
            rswt_penalty_enabled=True,
            rswt_penalty_weight=0.02,
            rswt_penalty_decay=0.9,
            rswt_penalty_exponent_rate=0.15,
            rswt_penalty_decay_max_hour=12,
            short_cycle_penalty_enabled=True,
            short_cycle_penalty_weight=6.0,
            stability_penalty_enabled=True,
            stability_penalty_weight=0.5,
            stability_penalty_decay=0.75,
            stability_penalty_threshold_kwh=10.0,
            stability_penalty_threshold_price_mwh=15.0,
            stability_penalty_horizon_hours=20,
            terminal_cost_enabled=True,
            terminal_cost_end_window_hours=24,
            experimental={"UnmetLoadPenaltyEnabled": True, "UnmetLoadPenaltyWeight": 10.0},
        ),
    )


@pytest.fixture(scope="module")
def recommendation() -> FloRecommendation:
    return recommend(oak_request())


def test_round_trips_through_the_word(recommendation: FloRecommendation) -> None:
    again = FloRecommendation.from_bytes(recommendation.to_bytes())
    assert again == recommendation
    assert recommendation.flo_name == "bruce"


def test_pairs_descend_from_the_ceiling(recommendation: FloRecommendation) -> None:
    pairs = recommendation.pq_pairs
    assert pairs[0].price_x1000 == 2000 * 1000
    assert pairs[0].quantity_x1000 == 0
    assert len(pairs) >= 2
    prices = [p.price_x1000 for p in pairs]
    assert prices == sorted(prices, reverse=True) and len(set(prices)) == len(prices)
    quantities = [p.quantity_x1000 for p in pairs]
    assert quantities == sorted(quantities)


def test_one_path_per_pair_and_first_step_is_the_bid(
    recommendation: FloRecommendation,
) -> None:
    assert len(recommendation.optimal_path_list) == len(recommendation.pq_pairs)
    for pair, path in zip(recommendation.pq_pairs, recommendation.optimal_path_list, strict=True):
        assert path.hp_avg_w_list[0] == pair.quantity_x1000
        assert path.start_unix_s == 1762009200
        assert len(path.hp_avg_w_list) == len(path.store_wh_list) == len(path.step_duration_minutes)
        assert len(path.hp_avg_w_list) == HORIZON


def test_the_store_and_the_penalties_follow_each_path(
    recommendation: FloRecommendation,
) -> None:
    stores = recommendation.store_state_list
    penalties = recommendation.path_penalties_list
    assert stores is not None and penalties is not None
    assert len(stores) == len(penalties) == len(recommendation.optimal_path_list)
    for path, store, charged in zip(recommendation.optimal_path_list, stores, penalties, strict=True):
        n = len(path.step_duration_minutes)
        assert store.store_layers == 9
        assert len(store.top_f_list) == len(store.thermocline2_list) == n
        assert all(1 <= t1 <= t2 <= 9 for t1, t2 in zip(store.thermocline1_list, store.thermocline2_list, strict=True))
        assert all(b <= m <= t for t, m, b in zip(store.top_f_list, store.middle_f_list, store.bottom_f_list, strict=True))
        assert len(charged.rswt_penalty_usd_list) == len(charged.short_cycle_penalty_usd_list) == n
        assert all(x >= 0 for x in charged.rswt_penalty_usd_list + charged.stability_penalty_usd_list + charged.short_cycle_penalty_usd_list)
        assert charged.terminal_cost_usd >= 0
        assert store.empty_store_wh < min(path.store_wh_list)
        assert max(path.store_wh_list) <= store.full_store_wh
        assert charged.terminal_cost_usd <= charged.terminal_cost_empty_store_usd
        assert charged.path_cost_usd >= charged.terminal_cost_usd + sum(
            charged.rswt_penalty_usd_list
            + charged.stability_penalty_usd_list
            + charged.short_cycle_penalty_usd_list
        )


def test_bytes_boundary(recommendation: FloRecommendation) -> None:
    from bruce.service import recommend_bytes

    reply = FloRecommendation.from_bytes(recommend_bytes(oak_request().to_bytes()))
    assert reply.pq_pairs == recommendation.pq_pairs
    assert reply.optimal_path_list == recommendation.optimal_path_list
    assert reply.store_state_list == recommendation.store_state_list


def test_the_previous_path_lines_up_from_its_second_step() -> None:
    """The previous path starts at the slot that cleared, the request at the
    slot after: the plan bruce compares against starts at the path's step 1,
    and the store it expected now is the store at the end of step 0."""
    request = oak_request()
    previous = FloOptimalPath(
        start_unix_s=request.start_unix_s - 3600,
        step_duration_minutes=[60, 60, 60],
        hp_avg_w_list=[3000, 4500, 0],
        store_wh_list=[180000, 185000, 182000],
    )
    request = request.model_copy(update={"previous_optimal_path": previous, "previous_estimate_elec_usd_mwh_now": 89.0})
    params = params_from_request(request)
    assert params.previous_plan_hp_kwh_el_list == [4.5, 0.0]
    assert params.previous_estimate_storage_kwh_now == 180.0
    assert params.previous_estimate_elec_price_mwh_now == 89.0


def test_thermoclines_map_onto_the_tunings_layers() -> None:
    from bruce.service import thermocline_layer

    # Nine sensors on nine layers: the observed position is the layer.
    assert [thermocline_layer(k, 9, 9) for k in (1, 3, 6, 9)] == [1, 3, 6, 9]
    # Twelve sensors across three tanks modeled on nine layers: same
    # fraction of the store, rounded to a layer, never below the first.
    assert [thermocline_layer(k, 12, 9) for k in (1, 4, 8, 12)] == [1, 3, 6, 9]
    # Coarser sensors than layers.
    assert [thermocline_layer(k, 3, 9) for k in (1, 2, 3)] == [3, 6, 9]


def test_params_take_the_layers_from_the_tuning() -> None:
    request = oak_request()
    params = params_from_request(request)
    assert isinstance(request.tuning, FloTuningBruce)
    assert params.num_layers == request.tuning.num_layers
    assert (params.initial_thermocline1, params.initial_thermocline2) == (3, 6)


def test_a_rerun_request_puts_the_forced_steps_on_the_optimizer_params() -> None:
    params = params_from_request(
        oak_request(), hp_forced_on_steps=[1, 4], hp_forced_off_steps=[2]
    )
    assert params.hp_forced_on_steps == [1, 4]
    assert params.hp_forced_off_steps == [2]


def test_a_002_request_is_read_as_005_with_the_unmet_load_penalty_off() -> None:
    from pathlib import Path

    from bruce.service import flo_params_from_bytes

    raw = Path("src/bruce/sema/samples/flo.params.002.json").read_bytes()
    params = flo_params_from_bytes(raw)
    assert params.version == "005"
    assert params.g_node_alias == "hw1.isone.me.versant.keene.oak"
    assert isinstance(params.tuning, FloTuningBruce)
    assert params.tuning.version == "003"
    assert params.tuning.experimental == {
        "UnmetLoadPenaltyEnabled": False,
        "UnmetLoadPenaltyWeight": 10.0,
    }


def test_a_003_request_is_read_as_005_with_its_knobs_in_experimental() -> None:
    from pathlib import Path

    from bruce.service import flo_params_from_bytes

    raw = Path("src/bruce/sema/samples/flo.params.003.json").read_bytes()
    params = flo_params_from_bytes(raw)
    assert params.version == "005"
    assert isinstance(params.tuning, FloTuningBruce)
    assert params.tuning.experimental == {
        "UnmetLoadPenaltyEnabled": True,
        "UnmetLoadPenaltyWeight": 10.0,
    }


def test_mode_tracking_defaults_on_when_its_experimental_key_is_absent() -> None:
    params = params_from_request(oak_request())
    assert params.heat_pump_mode_tracking_enabled is True


def test_mode_tracking_is_read_from_experimental() -> None:
    request = oak_request()
    assert isinstance(request.tuning, FloTuningBruce)
    request.tuning.experimental["HeatPumpModeTrackingEnabled"] = False
    params = params_from_request(request)
    assert params.heat_pump_mode_tracking_enabled is False


def test_a_knob_missing_from_experimental_is_refused_by_name() -> None:
    import pytest

    from bruce.service import params_from_request

    request = oak_request()
    assert isinstance(request.tuning, FloTuningBruce)
    request.tuning.experimental.pop("UnmetLoadPenaltyWeight")
    with pytest.raises(ValueError, match="UnmetLoadPenaltyWeight"):
        params_from_request(request)


def test_the_unmet_load_series_is_echoed_in_experimental() -> None:
    from bruce.service import recommend

    recommendation = recommend(oak_request())
    assert recommendation.path_penalties_list is not None
    for word, path in zip(recommendation.path_penalties_list, recommendation.optimal_path_list, strict=True):
        series = word.experimental["UnmetLoadPenaltyUsdList"]
        assert len(series) == len(path.hp_avg_w_list)
        assert all(v >= 0 for v in series)


def test_a_001_request_names_the_version_bruce_reads() -> None:
    from pathlib import Path

    from bruce.service import flo_params_from_bytes

    raw = Path("src/bruce/sema/samples/flo.params.001.json").read_bytes()
    with pytest.raises(ValueError, match="bruce reads flo.params 005 and this is 001"):
        flo_params_from_bytes(raw)


def test_a_rerun_of_an_older_params_names_the_version_bruce_reads() -> None:
    import json
    from pathlib import Path

    from bruce.service import flo_to_solve

    params = json.loads(
        Path("src/bruce/sema/samples/flo.params.001.json").read_text()
    )
    request = FloRerunRequest.from_dict(
        {
            "Params": params,
            "HpForcedOnSteps": [],
            "HpForcedOffSteps": [],
            "TypeName": "flo.rerun.request",
            "Version": "003",
        }
    )
    with pytest.raises(ValueError, match="bruce reads flo.params 005 and this is 001"):
        flo_to_solve(request)


def _cheapest(recommendation: FloRecommendation):
    penalties = recommendation.path_penalties_list
    assert penalties is not None
    index = min(range(len(penalties)), key=lambda i: penalties[i].path_cost_usd)
    return recommendation.optimal_path_list[index], penalties[index]


def test_forced_off_steps_are_zero_and_cost_more(recommendation: FloRecommendation) -> None:
    path, charged = _cheapest(recommendation)
    running = [i for i, watts in enumerate(path.hp_avg_w_list) if watts > 0][:2]
    assert len(running) == 2
    again = recommend(
        FloRerunRequest(
            params=oak_request(),
            hp_forced_on_steps=[],
            hp_forced_off_steps=running,
        )
    )
    for solved in again.optimal_path_list:
        assert solved.hp_avg_w_list[running[0]] == 0
        assert solved.hp_avg_w_list[running[1]] == 0
    _, again_charged = _cheapest(again)
    assert again_charged.path_cost_usd > charged.path_cost_usd


def test_forced_on_steps_run_at_the_heat_pumps_maximum() -> None:
    """A cold store has room for a full step of heat, so forcing the first
    step on (the heat pump starts off) is the turn-on-derated maximum: the
    nameplate with HpTurnOnMinutes taken off the front. The action grid
    lands on the nearest step under that."""
    request = oak_request().model_copy(
        update={
            "initial_top_temp_f": 80.0,
            "initial_middle_temp_f": 75.0,
            "initial_bottom_temp_f": 70.0,
            "hp_currently_on": False,
        }
    )
    again = recommend(
        FloRerunRequest(params=request, hp_forced_on_steps=[0], hp_forced_off_steps=[])
    )
    derated = round(
        request.hp_max_kw_el * (1 - request.hp_turn_on_minutes / 60) * 1000
    )
    for solved in again.optimal_path_list:
        watts = solved.hp_avg_w_list[0]
        assert abs(watts - derated) <= 40, watts
