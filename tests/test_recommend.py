"""recommend: one flo.params in, one flo.recommendation out, complete over
prices and true to the recommendation's axioms. Uses the cached transition
matrix for the one-tank, nine-sensor, 120-gallon shape (the same build
test_generate_bid needs)."""

import pytest

from bruce.sema.enums import SeasonalStorageMode
from bruce.sema.types import FloParams, FloRecommendation, FloTuningBruce
from bruce.service import recommend

HORIZON = 12
LMP = [28.14, 28.53, 46.38, 44.84, 44.8, 32.94, 39.68, 32.07, 30.07, 27.95, 29.87, 35.63]
DIST = [50.13, 50.13, 487.63, 487.63, 487.63, 487.63, 487.63, 54.98, 54.98, 54.98, 54.98, 487.63]


def oak_request() -> FloParams:
    return FloParams(
        g_node_alias="hw1.isone.me.versant.keene.oak",
        created_ms=1762008900000,
        start_unix_s=1762009200,
        step_duration_minutes=[60] * HORIZON,
        lmp_usd_mwh_list=LMP,
        dist_usd_mwh_list=DIST,
        oat_f_list=[20.0] * HORIZON,
        load_kwh_list=[6.0] * HORIZON,
        rswt_f_list=[120.0] * HORIZON,
        weather_forecast_id="5f1c2b3a-8d4e-4f60-9a7b-1c2d3e4f5a6b",
        seasonal_storage_mode=SeasonalStorageMode.BufferOnly,
        tanks_modeled=1,
        gallons_per_tank=120,
        temp_sensors_per_tank=9,
        max_tank_temp_f=160.0,
        initial_top_temp_f=120.0,
        initial_middle_temp_f=110.0,
        initial_bottom_temp_f=100.0,
        initial_thermocline1=3,
        initial_thermocline2=6,
        hp_currently_on=True,
        hp_max_kw_el=9.66,
        hp_turn_on_minutes=12,
        cop_intercept=1.5,
        cop_oat_coeff=0.015,
        cop_lwt_coeff=0.0,
        cop_min=1.5,
        cop_min_oat_f=0.0,
        price_max_usd_mwh=2000.0,
        price_min_usd_mwh=-100.0,
        tuning=FloTuningBruce(
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


def test_bytes_boundary(recommendation: FloRecommendation) -> None:
    from bruce.service import recommend_bytes

    reply = FloRecommendation.from_bytes(recommend_bytes(oak_request().to_bytes()))
    assert reply.pq_pairs == recommendation.pq_pairs
    assert reply.optimal_path_list == recommendation.optimal_path_list
