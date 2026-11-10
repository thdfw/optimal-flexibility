"""The FLO boundary: one request word in, one recommendation word out.

``recommend`` is the one callable a caller imports from bruce. It takes a
``flo.params`` whose ``Tuning`` is ``flo.tuning.bruce``, builds the
heat-pump-and-tank asset and its graph, solves, and answers with a
``flo.recommendation`` that is complete over prices: one PQ pair per
quantity change from the market ceiling down, and beside each pair the
plan its edge leads to, the tank store along that plan with its empty and
full energies, and the penalties and total cost bruce charged along it. The real-time price is never seen here; the
caller picks the cleared pair off the recommendation it holds.

The graph is built and discarded per call. Nothing spans calls except the
cached transition matrix for the plant shape, which must already exist
(``transitions_matrix``): a request whose shape has no matrix raises, so
matrices are generated as a separate step, never in the request path.
"""

from __future__ import annotations

import json
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import NamedTuple

from bruce.graph_optimizer import Graph
from bruce.graph_optimizer.assets.heat_pump_water_tank import (
    HeatPumpWaterTankAsset,
    HeatPumpWaterTankParams,
)
from bruce.graph_optimizer.assets.heat_pump_water_tank.state import compute_energy
from bruce.graph_optimizer.graph import Edge
from bruce.sema.enums import MarketPriceUnit, MarketQuantityUnit
from bruce.sema.types import (
    FloOptimalPath,
    FloParams,
    FloPathPenaltiesBruce,
    FloRecommendation,
    FloTankStorePath,
    FloTuningBruce,
    PriceQuantityUnitless,
)

FLO_NAME = "bruce"
FLO_REPO = "github.com/thdfw/optimal-flexibility"
UNKNOWN_COMMIT = "0" * 40
EMPTY_STORE_TEMP_F = 70


def bruce_git_commit() -> str:
    """The commit this bruce was installed from: the VCS pin recorded by
    the installer (a ``git+`` dependency), else the checkout's HEAD (an
    editable install), else forty zeros."""
    try:
        for dist in metadata.distributions():
            if dist.metadata["Name"] != "bruce-energy":
                continue
            direct = dist.read_text("direct_url.json")
            if direct:
                commit = json.loads(direct).get("vcs_info", {}).get("commit_id")
                if commit:
                    return str(commit)
    except (metadata.PackageNotFoundError, KeyError, ValueError):
        pass
    path = Path(__file__).resolve().parent
    for _ in range(8):
        if (path / ".git").is_dir():
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            break
        path = path.parent
    return UNKNOWN_COMMIT


def thermocline_layer(sensor_position: int, sensors: int, num_layers: int) -> int:
    """The model layer, counted from the top, at which a thermocline the
    house observed at ``sensor_position`` of ``sensors`` sits when the store
    is divided into ``num_layers``: the same fraction of the store, rounded
    to a layer and never above the top one."""
    return max(1, round(sensor_position * num_layers / sensors))


def params_from_request(request: FloParams) -> HeatPumpWaterTankParams:
    """Spell the request as bruce's own params. The store is modeled at the
    tuning's resolution, NumLayers, and the thermoclines the house observed
    in sensor positions are mapped onto those layers."""
    tuning = request.tuning
    if not isinstance(tuning, FloTuningBruce):
        raise TypeError(f"bruce answers flo.tuning.bruce, not {tuning.type_name}")
    sensors = request.tanks_modeled * request.temp_sensors_per_tank
    hours = [m / 60 for m in request.step_duration_minutes]
    # The previous path starts at the slot that cleared; this request starts
    # at the slot after it. Its step 0 is behind us, its store at the end of
    # step 0 is the store now, and its plan from step 1 on lines up with
    # this request's steps for the stability penalty.
    previous = request.previous_optimal_path
    previous_plan_kwh: list[float] | None = None
    previous_storage_kwh: float | None = None
    if previous is not None:
        previous_hours = [m / 60 for m in previous.step_duration_minutes]
        previous_plan_kwh = [
            w * h / 1000
            for w, h in zip(previous.hp_avg_w_list[1:], previous_hours[1:], strict=True)
        ]
        previous_storage_kwh = previous.store_wh_list[0] / 1000
    return HeatPumpWaterTankParams(
        horizon=len(hours),
        start_unix_s=request.start_unix_s,
        site_id=request.g_node_alias,
        timestep_duration_hours=hours,
        bruce_git_commit=bruce_git_commit(),
        elec_price_mwh=[
            lmp + dist
            for lmp, dist in zip(request.lmp_usd_mwh_list, request.dist_usd_mwh_list, strict=True)
        ],
        bid_min_price_mwh=request.price_min_usd_mwh,
        bid_max_price_mwh=request.price_max_usd_mwh,
        num_layers=tuning.num_layers,
        storage_volume_gallons=request.tanks_modeled * request.gallons_per_tank,
        storage_losses_percent=tuning.storage_losses_percent,
        max_tank_temp_f=request.max_tank_temp_f,
        hp_min_kw_elec=tuning.hp_min_kw_el,
        hp_max_kw_elec=request.hp_max_kw_el,
        hp_turn_on_minutes=request.hp_turn_on_minutes,
        hp_constant_lift_f=tuning.hp_constant_lift_f,
        cop_intercept=request.cop_intercept,
        cop_oat_coeff=request.cop_oat_coeff,
        cop_lwt_coeff=request.cop_lwt_coeff,
        cop_min=request.cop_min,
        cop_min_oat_f=request.cop_min_oat_f,
        load_bucket_upper_kwh=list(tuning.load_bucket_upper_kwh_list),
        rwt_intercept_by_bucket=list(tuning.rwt_intercept_by_bucket_list),
        rwt_slope_by_bucket=list(tuning.rwt_slope_by_bucket_list),
        rwt_min=tuning.rwt_min_f,
        hp_max_kw_th=tuning.hp_max_kw_th,
        max_load_kw_th=tuning.max_load_kw_th,
        rswt_penalty_enabled=tuning.rswt_penalty_enabled,
        rswt_penalty_weight=tuning.rswt_penalty_weight,
        rswt_penalty_decay=tuning.rswt_penalty_decay,
        rswt_penalty_exponent_rate=tuning.rswt_penalty_exponent_rate,
        rswt_penalty_decay_max_hour=tuning.rswt_penalty_decay_max_hour,
        short_cycle_penalty_enabled=tuning.short_cycle_penalty_enabled,
        short_cycle_penalty_weight=tuning.short_cycle_penalty_weight,
        stability_penalty_enabled=tuning.stability_penalty_enabled,
        stability_penalty_weight=tuning.stability_penalty_weight,
        stability_penalty_decay=tuning.stability_penalty_decay,
        stability_penalty_threshold_kwh=tuning.stability_penalty_threshold_kwh,
        stability_penalty_threshold_price_mwh=tuning.stability_penalty_threshold_price_mwh,
        stability_penalty_horizon_hours=tuning.stability_penalty_horizon_hours,
        previous_plan_hp_kwh_el_list=previous_plan_kwh,
        previous_estimate_storage_kwh_now=previous_storage_kwh,
        previous_estimate_elec_price_mwh_now=request.previous_estimate_elec_usd_mwh_now,
        terminal_cost_enabled=tuning.terminal_cost_enabled,
        terminal_cost_end_window_hours=tuning.terminal_cost_end_window_hours,
        hp_currently_on=request.hp_currently_on,
        initial_top_temp=request.initial_top_temp_f,
        initial_middle_temp=request.initial_middle_temp_f,
        initial_bottom_temp=request.initial_bottom_temp_f,
        initial_thermocline1=thermocline_layer(
            request.initial_thermocline1, sensors, tuning.num_layers
        ),
        initial_thermocline2=thermocline_layer(
            request.initial_thermocline2, sensors, tuning.num_layers
        ),
        rswt_f=list(request.rswt_f_list),
        load_kwh=list(request.load_kwh_list),
        oat_f=list(request.oat_f_list),
    )


def store_wh_at(temp_f: float, params: HeatPumpWaterTankParams) -> int:
    """The energy of the whole store at one temperature, in watt-hours, on
    the reference every state's energy is measured from."""
    n = params.num_layers
    return round(compute_energy(temp_f, temp_f, temp_f, n, n, params) * 1000)


class PathWords(NamedTuple):
    """The three words one first-step edge leads to: the plan, the tank
    store along it and bruce's penalties along it, step for step."""

    path: FloOptimalPath
    store: FloTankStorePath
    penalties: FloPathPenaltiesBruce


def path_words(graph: Graph, request: FloParams, first: Edge) -> PathWords:
    """The plan the first-step edge leads to: that edge, then the plan
    onward (``Graph.rollout``), each step at the head the rollout takes.
    Per step, the heat pump's average watts and the store at the end of the
    step (energy, the three band temperatures, the two thermoclines as
    layers of the tuning's NumLayers), and the three penalties the step
    is charged at that head; the terminal cost is the last head's, and the
    path cost is every step's cost at its head (electricity at the
    request's prices plus the three penalties) plus that terminal cost.
    Every list has the number of steps the path reaches."""
    asset = graph.asset
    if not isinstance(asset, HeatPumpWaterTankAsset):
        raise TypeError("path_words needs a HeatPumpWaterTankAsset graph")
    hours = [m / 60 for m in request.step_duration_minutes]
    hp_avg_w: list[int] = []
    store_wh: list[int] = []
    top_f: list[int] = []
    middle_f: list[int] = []
    bottom_f: list[int] = []
    thermocline1: list[int] = []
    thermocline2: list[int] = []
    rswt: list[float] = []
    stability: list[float] = []
    short_cycle: list[float] = []
    steps_cost = 0.0
    steps = graph.rollout(first, max_steps=len(hours))
    for h, (edge, head) in zip(hours, steps):
        t = edge.tail.time_step
        start = edge.tail.state
        end = head.state
        hp_avg_w.append(round(edge.elec_used_kwh / h * 1000))
        steps_cost += asset.cost(start, end, edge.action, t, edge.tail.mode)
        store_wh.append(round(float(end.energy) * 1000))
        top_f.append(round(end.top_temp))
        middle_f.append(round(end.middle_temp))
        bottom_f.append(round(end.bottom_temp))
        thermocline1.append(end.thermocline1)
        thermocline2.append(end.thermocline2)
        rswt.append(asset.rswt_penalty(start, end, edge.action, t))
        stability.append(asset.stability_penalty(t, edge.elec_used_kwh))
        short_cycle.append(asset.short_cycle_penalty(start, edge.action, t, edge.tail.mode))
    n = len(hp_avg_w)
    terminal_cost = asset.terminal_node_cost(steps[n - 1][1].state)
    return PathWords(
        path=FloOptimalPath(
            start_unix_s=request.start_unix_s,
            step_duration_minutes=list(request.step_duration_minutes[:n]),
            hp_avg_w_list=hp_avg_w,
            store_wh_list=store_wh,
        ),
        store=FloTankStorePath(
            store_layers=asset.params.num_layers,
            top_f_list=top_f,
            middle_f_list=middle_f,
            bottom_f_list=bottom_f,
            thermocline1_list=thermocline1,
            thermocline2_list=thermocline2,
            empty_store_wh=store_wh_at(EMPTY_STORE_TEMP_F, asset.params),
            full_store_wh=store_wh_at(asset.params.max_tank_temp_f, asset.params),
        ),
        penalties=FloPathPenaltiesBruce(
            rswt_penalty_usd_list=rswt,
            stability_penalty_usd_list=stability,
            short_cycle_penalty_usd_list=short_cycle,
            terminal_cost_usd=terminal_cost,
            terminal_cost_empty_store_usd=asset.terminal_cost_full(),
            path_cost_usd=steps_cost + terminal_cost,
        ),
    )


def recommend(request: FloParams) -> FloRecommendation:
    """Answer one request. The bid curve descends from the market ceiling
    (``Graph.generate_bid``); the first pair sits at the request's own
    PriceMaxUsdMwh (flo.recommendation axiom 1) and each pair's path is
    the first-step edge that pair's price selects (axiom 3)."""
    params = params_from_request(request)
    graph = Graph(HeatPumpWaterTankAsset(params))
    graph.find_shortest_path()
    pairs = graph.generate_bid(forecast_price_mwh=params.elec_price_mwh[0])
    pq_pairs: list[PriceQuantityUnitless] = []
    words: list[PathWords] = []
    for i, pair in enumerate(pairs):
        price_mwh = request.price_max_usd_mwh if i == 0 else pair.price_mwh
        first = graph.best_edge_at_price(pair.price_mwh)
        for_pair = path_words(graph, request, first)
        pq_pairs.append(
            PriceQuantityUnitless(
                price_x1000=round(price_mwh * 1000),
                quantity_x1000=for_pair.path.hp_avg_w_list[0],
            )
        )
        words.append(for_pair)
    return FloRecommendation(
        bidder_alias=request.g_node_alias,
        pq_pairs=pq_pairs,
        price_unit=MarketPriceUnit.USDPerMWh,
        quantity_unit=MarketQuantityUnit.AvgkW,
        optimal_path_list=[w.path for w in words],
        store_state_list=[w.store for w in words],
        path_penalties_list=[w.penalties for w in words],
        flo_name=FLO_NAME,
        flo_repo=FLO_REPO,
        flo_git_commit=params.bruce_git_commit,
        created_ms=int(time.time() * 1000),
    )


def recommend_bytes(request_bytes: bytes) -> bytes:
    """The boundary as bytes: ``flo.params`` JSON in, ``flo.recommendation``
    JSON out. A caller holds its own twin of each word and nothing of
    bruce's; this is the one call an LTN or a service makes."""
    return recommend(FloParams.from_bytes(request_bytes)).to_bytes()
