"""generate_bid: the bid curve descends from the market ceiling.

A half-charged store on a cold day, so buying is worth it below some price and
not above it: the curve has at least two segments. Needs the cached transition
matrix for the default geometry (CAN_COMPUTE_TRANSITION_MATRICES=true builds it
once, in about fifteen minutes).
"""

import pytest

from bruce.graph_optimizer import Graph
from bruce.graph_optimizer.assets.heat_pump_water_tank import (
    HeatPumpWaterTankAsset,
    HeatPumpWaterTankParams,
)

HORIZON = 12
LMP = [28.14, 28.53, 46.38, 44.84, 44.8, 32.94, 39.68, 32.07, 30.07, 27.95, 29.87, 35.63]
DIST = [50.13, 50.13, 487.63, 487.63, 487.63, 487.63, 487.63, 54.98, 54.98, 54.98, 54.98, 487.63]


@pytest.fixture(scope="module")
def graph() -> Graph:
    params = HeatPumpWaterTankParams(
        horizon=HORIZON,
        start_unix_s=1762009200,
        site_id="d1.isone.ver.keene.test",
        elec_price_mwh=[lmp + dist for lmp, dist in zip(LMP, DIST, strict=True)],
        rswt_f=[120.0] * HORIZON,
        load_kwh=[6.0] * HORIZON,
        oat_f=[20.0] * HORIZON,
        initial_top_temp=120,
        initial_middle_temp=110,
        initial_bottom_temp=100,
        initial_thermocline1=3,
        initial_thermocline2=6,
    )
    g = Graph(HeatPumpWaterTankAsset(params))
    g.find_shortest_path()
    g.trim_graph_for_waiting()
    return g


@pytest.fixture(scope="module")
def pq_pairs(graph: Graph):
    return graph.generate_bid(forecast_price_mwh=graph.params.elec_price_mwh[0])


def test_first_pair_is_at_the_price_ceiling(graph: Graph, pq_pairs) -> None:
    assert pq_pairs[0].price_mwh == graph.params.bid_max_price_mwh


def test_curve_has_more_than_one_segment(pq_pairs) -> None:
    assert len(pq_pairs) > 1


def test_prices_strictly_descend_and_quantities_do_not_decrease(pq_pairs) -> None:
    prices = [p.price_mwh for p in pq_pairs]
    quantities = [p.quantity_kwh for p in pq_pairs]
    assert prices == sorted(prices, reverse=True)
    assert len(set(prices)) == len(prices)
    assert quantities == sorted(quantities)


def test_adjacent_quantities_differ(pq_pairs) -> None:
    for a, b in zip(pq_pairs, pq_pairs[1:]):
        assert abs(b.quantity_kwh - a.quantity_kwh) > 0.01
