from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from bruce.graph_optimizer.graph import Edge, Graph, Node

# Forecast wall times and plot titles use Eastern Time (handles DST).
_AMERICA_NEW_YORK = ZoneInfo("America/New_York")

from .action import HeatPumpWaterTankAction
from .asset import HeatPumpWaterTankAsset
from .params import HeatPumpWaterTankParams
from .state import HeatPumpWaterTankState

HpGraph = Graph[HeatPumpWaterTankState, HeatPumpWaterTankAction, HeatPumpWaterTankParams]


def _graph_asset(graph: HpGraph) -> HeatPumpWaterTankAsset:
    asset = graph.asset
    if not isinstance(asset, HeatPumpWaterTankAsset):
        raise TypeError("plot_graph_results requires a HeatPumpWaterTankAsset")
    return asset


def _initial_node(graph: HpGraph) -> Node[HeatPumpWaterTankState]:
    return graph.find_initial_node()


def _edge_for_transition(
    graph: HpGraph,
    node: Node[HeatPumpWaterTankState],
) -> Edge[HeatPumpWaterTankState, HeatPumpWaterTankAction] | None:
    if node.next_node is None:
        return None
    for edge in graph.edges[node]:
        if edge.head is node.next_node:
            return edge
    return None


def _step_start_datetimes(params: HeatPumpWaterTankParams) -> list[datetime]:
    """Wall-clock start of each step in America/New_York; last is end of final step."""
    t0 = datetime.fromtimestamp(params.start_unix_s, tz=_AMERICA_NEW_YORK)
    out: list[datetime] = [t0]
    for dt_h in params.timestep_duration_hours:
        out.append(out[-1] + timedelta(hours=dt_h))
    return out


def _hp_heat_out_kwh(
    asset: HeatPumpWaterTankAsset,
    state: HeatPumpWaterTankState,
    action: HeatPumpWaterTankAction,
    time_step: int,
) -> float:
    cop = asset.params.COP(asset.params.oat_f[time_step])
    return asset.elec_used_kwh(state, action, time_step) * cop


def _thermal_and_normalized_prices(params: HeatPumpWaterTankParams) -> tuple[list[float], list[float]]:
    """Per-step price of heat ($/MWh_th) and the electricity price divided by the horizon's mean COP."""
    cops = [params.COP(oat) for oat in params.oat_f]
    mean_cop = sum(cops) / len(cops)
    thermal = [float(p / cop) for p, cop in zip(params.elec_price_mwh, cops, strict=True)]
    normalized = [float(p / mean_cop) for p in params.elec_price_mwh]
    return thermal, normalized


def graph_plot_data(graph: HpGraph) -> dict[str, Any]:
    """Series for the three-panel heat-pump plot. Times are America/New_York wall clocks."""
    import matplotlib
    from matplotlib.colors import Normalize, to_rgba

    asset = _graph_asset(graph)
    params = asset.params
    initial_node = _initial_node(graph)
    step_starts = _step_start_datetimes(params)
    sp_time: list[datetime] = step_starts

    sp_top_temp: list[float] = []
    sp_middle_temp: list[float] = []
    sp_bottom_temp: list[float] = []
    sp_thermocline1: list[int] = []
    sp_thermocline2: list[int] = []
    sp_hp_heat_out_kw: list[float] = []
    sp_stored_energy: list[float] = []
    sp_rswt_penalty: list[float] = []
    sp_stability_penalty: list[float] = []
    sp_short_cycle_penalty: list[float] = []

    node: Node[HeatPumpWaterTankState] | None = initial_node
    last_edge: Edge[HeatPumpWaterTankState, HeatPumpWaterTankAction] | None = None
    the_end = False

    while not the_end:
        assert node is not None
        if node.next_node is None:
            the_end = True
            if last_edge is not None:
                t = last_edge.tail.time_step
                dt = params.timestep_duration_hours[t]
                sp_hp_heat_out_kw.append(_hp_heat_out_kwh(asset, last_edge.tail.state, last_edge.action, t) / dt)
            sp_rswt_penalty.append(0.0)
            sp_stability_penalty.append(0.0)
            sp_short_cycle_penalty.append(0.0)
        else:
            edge = _edge_for_transition(graph, node)
            if edge is None:
                raise RuntimeError(f"No edge found from {node} to {node.next_node}")
            last_edge = edge
            t = node.time_step
            dt = params.timestep_duration_hours[t]
            cop = params.COP(params.oat_f[t])
            heat_out = _hp_heat_out_kwh(asset, node.state, edge.action, t)
            sp_hp_heat_out_kw.append(heat_out / dt)
            sp_rswt_penalty.append(asset.rswt_penalty(node.state, edge.head.state, edge.action, t))
            sp_stability_penalty.append(asset.stability_penalty(t, heat_out / cop))
            sp_short_cycle_penalty.append(asset.short_cycle_penalty(node.state, edge.action, t, node.mode))

        st = node.state
        sp_top_temp.append(st.top_temp)
        sp_middle_temp.append(st.middle_temp)
        sp_bottom_temp.append(st.bottom_temp)
        sp_thermocline1.append(st.thermocline1)
        sp_thermocline2.append(st.thermocline2)
        sp_stored_energy.append(st.energy)
        node = node.next_node

    sp_soc = [
        (e - asset.min_state_energy) / (asset.max_state_energy - asset.min_state_energy) * 100
        for e in sp_stored_energy
    ]
    terminal_node = initial_node
    while terminal_node.next_node is not None:
        terminal_node = terminal_node.next_node
    terminal_cost = asset.terminal_node_cost(terminal_node.state)
    terminal_cost_full = asset.terminal_cost_full()
    terminal_cost_pct = (
        100.0 * terminal_cost / terminal_cost_full if terminal_cost_full > 0 else 0.0
    )

    load_kw = [float(v) for v in (
        params.load_kwh[t] / params.timestep_duration_hours[t] for t in range(params.horizon)
    )]
    prices = [float(v) for v in params.elec_price_mwh]
    thermal_prices, normalized_prices = _thermal_and_normalized_prices(params)
    # Repeat the last step so a post-step line covers the final interval.
    if load_kw:
        load_kw.append(load_kw[-1])
    if prices:
        prices.append(prices[-1])
        thermal_prices.append(thermal_prices[-1])
        normalized_prices.append(normalized_prices[-1])

    norm = Normalize(vmin=60, vmax=180)
    cmap = matplotlib.colormaps["Reds"]

    def temp_rgba(temp: float) -> str:
        r, g, b, _ = to_rgba(cmap(float(norm(temp))))
        return f"rgba({round(r * 255)},{round(g * 255)},{round(b * 255)},0.7)"

    def text_color(temp: float, i: int) -> str:
        rswt = params.rswt_f[i] if i < len(params.rswt_f) else None
        if rswt is not None and rswt <= temp:
            return "green"
        if temp < 100:
            return "gray"
        return "white"

    def layer(
        heights: list[float],
        bases: list[float],
        temps: list[float],
        show_text: list[bool],
    ) -> dict[str, Any]:
        return {
            "y": [float(h) for h in heights],
            "base": [float(b) for b in bases],
            "color": [temp_rgba(float(t)) for t in temps],
            "text": [str(int(t)) if show else "" for t, show in zip(temps, show_text, strict=True)],
            "textColor": [text_color(float(t), i) for i, t in enumerate(temps)],
            "temp": [float(t) for t in temps],
        }

    n = params.num_layers
    top_h = [float(x) for x in sp_thermocline1]
    mid_h = [float(b - a) for a, b in zip(sp_thermocline1, sp_thermocline2, strict=True)]
    bot_h = [float(n - x) for x in sp_thermocline2]
    top_base = [float(n - x) for x in sp_thermocline1]
    mid_base = [float(n - x) for x in sp_thermocline2]
    bot_base = [0.0] * len(sp_time)
    width_ms = [
        params.timestep_duration_hours[min(t, params.horizon - 1)] * 0.9 * 3_600_000
        for t in range(len(sp_time))
    ]

    rswt_vals = [float(v) for v in sp_rswt_penalty[:-1]]
    stab_vals = [float(v) for v in sp_stability_penalty[:-1]]
    short_vals = [float(v) for v in sp_short_cycle_penalty[:-1]]
    sp_time_mid = [
        step_starts[t] + timedelta(hours=params.timestep_duration_hours[t] / 2.0)
        for t in range(params.horizon)
    ]
    terminal_vals = [0.0] * len(sp_time_mid)
    if terminal_vals and params.terminal_cost_enabled:
        terminal_vals[-1] = float(terminal_cost)

    penalty_series: list[dict[str, Any]] = []
    stack = [0.0] * len(sp_time_mid)

    def push_penalty(name: str, color: str, values: list[float]) -> None:
        nonlocal stack
        penalty_series.append({
            "name": name,
            "color": color,
            "y": values,
            "base": list(stack),
        })
        stack = [b + v for b, v in zip(stack, values, strict=True)]

    if params.rswt_penalty_enabled:
        push_penalty("RSWT penalty", "rgba(255,127,14,0.7)", rswt_vals)
    if params.stability_penalty_enabled:
        push_penalty("Stability penalty", "rgba(148,103,189,0.7)", stab_vals)
    if params.short_cycle_penalty_enabled:
        push_penalty("Short-cycle penalty", "rgba(23,190,207,0.7)", short_vals)
    if params.terminal_cost_enabled:
        push_penalty(
            f"Terminal cost ({terminal_cost_pct:.0f}% of max)",
            "rgba(44,160,44,0.7)",
            terminal_vals,
        )

    per_series_max: list[float] = []
    if params.rswt_penalty_enabled:
        per_series_max.append(max(rswt_vals, default=0.0))
    if params.stability_penalty_enabled:
        per_series_max.append(max(stab_vals, default=0.0))
    if params.short_cycle_penalty_enabled:
        per_series_max.append(max(short_vals, default=0.0))
    if params.terminal_cost_enabled:
        per_series_max.append(float(terminal_cost))
    max_stack = 0.0
    for i in range(len(sp_time_mid)):
        total = 0.0
        if params.rswt_penalty_enabled:
            total += rswt_vals[i]
        if params.stability_penalty_enabled:
            total += stab_vals[i]
        if params.short_cycle_penalty_enabled:
            total += short_vals[i]
        if params.terminal_cost_enabled:
            total += terminal_vals[i]
        max_stack = max(max_stack, total)
    max_penalty = max(max(per_series_max, default=0.0), max_stack)
    hp_max = max(sp_hp_heat_out_kw, default=0.0)

    def iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "siteId": params.site_id,
        "subtitle": f"{step_starts[0].strftime('%b %d, %Y %H:%M %Z')} — ${round(float(initial_node.pathcost), 2)}",
        "numLayers": n,
        "power": {
            "time": [iso(t) for t in sp_time],
            "hpKw": [float(v) for v in sp_hp_heat_out_kw],
            "loadKw": load_kw,
            "priceUsdMwh": prices,
            "priceNormalizedUsdMwh": normalized_prices,
            "thermalPriceUsdMwh": thermal_prices,
            "yRange": [-0.5, 1.5 * hp_max] if hp_max > 0 else None,
        },
        "storage": {
            "time": [iso(t) for t in sp_time],
            "widthMs": width_ms,
            "layers": [
                layer(bot_h, bot_base, sp_bottom_temp, [x != n for x in sp_thermocline2]),
                layer(mid_h, mid_base, sp_middle_temp, [h > 1 for h in mid_h]),
                layer(top_h, top_base, sp_top_temp, [True] * len(sp_time)),
            ],
            "soc": [float(v) for v in sp_soc],
        },
        "penalty": {
            "time": [iso(t) for t in sp_time_mid],
            "widthMs": [h * 0.9 * 3_600_000 for h in params.timestep_duration_hours],
            "series": penalty_series,
            "yRange": [0, max_penalty * 1.3] if max_penalty > 0 else None,
        },
    }


def plot_graph_results(
    graph: HpGraph,
    *,
    show: bool = True,
    save_as: str | None = None,
) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import Normalize

    asset = _graph_asset(graph)
    params = asset.params
    initial_node = _initial_node(graph)
    step_starts = _step_start_datetimes(params)
    # One timestamp per node on the path (initial through terminal) == horizon + 1.
    sp_time: list[datetime] = step_starts

    sp_top_temp: list[float] = []
    sp_middle_temp: list[float] = []
    sp_bottom_temp: list[float] = []
    sp_thermocline1: list[int] = []
    sp_thermocline2: list[int] = []
    sp_hp_heat_out_kw: list[float] = []
    sp_stored_energy: list[float] = []
    sp_rswt_penalty: list[float] = []
    sp_stability_penalty: list[float] = []
    sp_short_cycle_penalty: list[float] = []

    node: Node[HeatPumpWaterTankState] | None = initial_node
    last_edge: Edge[HeatPumpWaterTankState, HeatPumpWaterTankAction] | None = None
    the_end = False

    while not the_end:
        assert node is not None
        if node.next_node is None:
            the_end = True
            if last_edge is not None:
                t = last_edge.tail.time_step
                dt = params.timestep_duration_hours[t]
                sp_hp_heat_out_kw.append(_hp_heat_out_kwh(asset, last_edge.tail.state, last_edge.action, t) / dt)
            sp_rswt_penalty.append(0.0)
            sp_stability_penalty.append(0.0)
            sp_short_cycle_penalty.append(0.0)
        else:
            edge = _edge_for_transition(graph, node)
            if edge is None:
                raise RuntimeError(f"No edge found from {node} to {node.next_node}")
            last_edge = edge
            t = node.time_step
            dt = params.timestep_duration_hours[t]
            cop = params.COP(params.oat_f[t])
            heat_out = _hp_heat_out_kwh(asset, node.state, edge.action, t)
            sp_hp_heat_out_kw.append(heat_out / dt)
            sp_rswt_penalty.append(asset.rswt_penalty(node.state, edge.head.state, edge.action, t))
            sp_stability_penalty.append(asset.stability_penalty(t, heat_out / cop))
            sp_short_cycle_penalty.append(asset.short_cycle_penalty(node.state, edge.action, t, node.mode))

        st = node.state
        sp_top_temp.append(st.top_temp)
        sp_middle_temp.append(st.middle_temp)
        sp_bottom_temp.append(st.bottom_temp)
        sp_thermocline1.append(st.thermocline1)
        sp_thermocline2.append(st.thermocline2)
        sp_stored_energy.append(st.energy)
        node = node.next_node

    sp_soc = [
        (e - asset.min_state_energy) / (asset.max_state_energy - asset.min_state_energy) * 100
        for e in sp_stored_energy
    ]
    terminal_node = initial_node
    while terminal_node.next_node is not None:
        terminal_node = terminal_node.next_node
    terminal_cost = asset.terminal_node_cost(terminal_node.state)
    terminal_cost_full = asset.terminal_cost_full()
    terminal_cost_pct = (
        100.0 * terminal_cost / terminal_cost_full if terminal_cost_full > 0 else 0.0
    )

    load_kw = [
        params.load_kwh[t] / params.timestep_duration_hours[t]
        for t in range(params.horizon)
    ]
    thermal_prices, normalized_prices = _thermal_and_normalized_prices(params)

    start_dt = step_starts[0]
    title_start = start_dt.strftime("%b %d, %Y %H:%M %Z")
    total_cost_str = f"${round(initial_node.pathcost, 2)}"
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 3, 2]})

    ax[0].step(sp_time, sp_hp_heat_out_kw, where="post", color="tab:blue", alpha=0.6, label="HP")
    ax[0].step(sp_time[:-1], load_kw, where="post", color="tab:red", alpha=0.6, label="Load")
    ax[0].legend(loc="upper left")
    ax[0].set_ylabel("Heating power [kW]")
    if max(sp_hp_heat_out_kw) > 0:
        ax[0].set_ylim([-0.5, 1.5 * max(sp_hp_heat_out_kw)])
    ax0_price = ax[0].twinx()
    ax0_price.step(sp_time[:-1], thermal_prices, where="post", color="gray", alpha=0.6, label="Thermal price")
    ax0_price.step(
        sp_time[:-1],
        normalized_prices,
        where="post",
        color="gray",
        alpha=0.6,
        linestyle="--",
        label="Electricity price (normalized)",
    )
    ax0_price.legend(loc="upper right")
    ax0_price.set_ylabel("Thermal price [$/MWh_th]")

    norm = Normalize(vmin=60, vmax=180)
    cmap = matplotlib.colormaps["Reds"]
    tank_top_colors = [cmap(norm(x)) for x in sp_top_temp]
    tank_middle_colors = [cmap(norm(x)) for x in sp_middle_temp]
    tank_bottom_colors = [cmap(norm(x)) for x in sp_bottom_temp]

    th1_rev = [params.num_layers - x for x in sp_thermocline1]
    th2_rev = [params.num_layers - x for x in sp_thermocline2]

    bar_width_days = [
        timedelta(hours=params.timestep_duration_hours[min(t, params.horizon - 1)]) * 0.9
        for t in range(len(sp_time))
    ]

    bars_top = ax[1].bar(
        sp_time,
        sp_thermocline1,
        bottom=th1_rev,
        color=tank_top_colors,
        alpha=0.7,
        width=bar_width_days,
        align="edge",
    )
    bars_middle = ax[1].bar(
        sp_time,
        [y - x for x, y in zip(sp_thermocline1, sp_thermocline2)],
        bottom=th2_rev,
        color=tank_middle_colors,
        alpha=0.7,
        width=bar_width_days,
        align="edge",
    )
    bars_bottom = ax[1].bar(
        sp_time, th2_rev, bottom=0, color=tank_bottom_colors, alpha=0.7, width=bar_width_days, align="edge"
    )
    ax[1].set_ylabel("Storage state")
    ax[1].set_ylim([0, params.num_layers])
    ax[1].set_yticks([])

    for i, bar in enumerate(bars_top):
        height = bar.get_height()
        bar_color = "white"
        if i < len(params.rswt_f) and params.rswt_f[i] <= sp_top_temp[i]:
            bar_color = "green"
        elif sp_top_temp[i] < 100:
            bar_color = "gray"
        ax[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + height / 2,
            f"{int(sp_top_temp[i])}",
            ha="center",
            va="center",
            color=bar_color,
            fontsize=6,
        )
    for i, bar in enumerate(bars_middle):
        height = bar.get_height()
        bar_color = "white"
        if i < len(params.rswt_f) and params.rswt_f[i] <= sp_middle_temp[i]:
            bar_color = "green"
        elif sp_middle_temp[i] < 100:
            bar_color = "gray"
        if height > 1:
            ax[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height / 2,
                f"{int(sp_middle_temp[i])}",
                ha="center",
                va="center",
                color=bar_color,
                fontsize=6,
            )
    for i, bar in enumerate(bars_bottom):
        height = bar.get_height()
        bar_color = "white"
        if i < len(params.rswt_f) and params.rswt_f[i] <= sp_bottom_temp[i]:
            bar_color = "green"
        elif sp_bottom_temp[i] < 100:
            bar_color = "gray"
        if sp_thermocline2[i] == params.num_layers:
            continue
        ax[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + height / 2,
            f"{int(sp_bottom_temp[i])}",
            ha="center",
            va="center",
            color=bar_color,
            fontsize=6,
        )

    ax1_soc = ax[1].twinx()
    soc_time = step_starts[: len(sp_soc)]
    ax1_soc.plot(soc_time, sp_soc, color="black", alpha=0.4, label="SoC")
    ax1_soc.set_ylabel("State of charge [%]")
    ax1_soc.set_ylim([-1, 101])

    sp_time_mid = [
        step_starts[t]
        + timedelta(hours=params.timestep_duration_hours[t] / 2.0)
        for t in range(params.horizon)
    ]
    bar_width_penalty_days = [timedelta(hours=params.timestep_duration_hours[t]) * 0.9 for t in range(params.horizon)]
    rswt_vals = sp_rswt_penalty[:-1]
    stab_vals = sp_stability_penalty[:-1]
    short_vals = sp_short_cycle_penalty[:-1]
    terminal_vals = [0.0] * len(sp_time_mid)
    if terminal_vals and params.terminal_cost_enabled:
        terminal_vals[-1] = terminal_cost

    stack_bottom = [0.0] * len(sp_time_mid)
    if params.rswt_penalty_enabled:
        ax[2].bar(
            sp_time_mid,
            rswt_vals,
            bottom=stack_bottom,
            width=bar_width_penalty_days,
            color="tab:orange",
            alpha=0.7,
            label="RSWT penalty",
        )
        stack_bottom = [b + v for b, v in zip(stack_bottom, rswt_vals)]
    if params.stability_penalty_enabled:
        ax[2].bar(
            sp_time_mid,
            stab_vals,
            bottom=stack_bottom,
            width=bar_width_penalty_days,
            color="tab:purple",
            alpha=0.7,
            label="Stability penalty",
        )
        stack_bottom = [b + v for b, v in zip(stack_bottom, stab_vals)]
    if params.short_cycle_penalty_enabled:
        ax[2].bar(
            sp_time_mid,
            short_vals,
            bottom=stack_bottom,
            width=bar_width_penalty_days,
            color="tab:cyan",
            alpha=0.7,
            label="Short-cycle penalty",
        )
        stack_bottom = [b + v for b, v in zip(stack_bottom, short_vals)]
    if params.terminal_cost_enabled:
        ax[2].bar(
            sp_time_mid,
            terminal_vals,
            bottom=stack_bottom,
            width=bar_width_penalty_days,
            color="tab:green",
            alpha=0.7,
            label=f"Terminal cost ({terminal_cost_pct:.0f}% of max)",
        )

    ax[2].set_xlabel("Time (America/New_York)")
    ax[2].set_ylabel("Penalty [$]")
    ax[2].legend(loc="upper right", fontsize=8)

    per_series_max: list[float] = []
    if params.rswt_penalty_enabled:
        per_series_max.append(max(rswt_vals, default=0.0))
    if params.stability_penalty_enabled:
        per_series_max.append(max(stab_vals, default=0.0))
    if params.short_cycle_penalty_enabled:
        per_series_max.append(max(short_vals, default=0.0))
    if params.terminal_cost_enabled:
        per_series_max.append(terminal_cost)

    max_stack = 0.0
    for i in range(len(sp_time_mid)):
        total = 0.0
        if params.rswt_penalty_enabled:
            total += rswt_vals[i]
        if params.stability_penalty_enabled:
            total += stab_vals[i]
        if params.short_cycle_penalty_enabled:
            total += short_vals[i]
        if params.terminal_cost_enabled:
            total += terminal_vals[i]
        max_stack = max(max_stack, total)

    max_penalty = max(max(per_series_max, default=0.0), max_stack)
    if max_penalty > 0:
        ax[2].set_ylim([0, max_penalty * 1.3])

    for a in ax:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=_AMERICA_NEW_YORK))
        a.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.text(
        0.5,
        0.98,
        params.site_id,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        transform=fig.transFigure,
    )
    fig.text(
        0.5,
        0.945,
        f"{title_start} — {total_cost_str}",
        ha="center",
        va="top",
        fontsize=10,
        transform=fig.transFigure,
    )
    if save_as is not None:
        plt.savefig(save_as, dpi=130)
    if show:
        plt.show()
    else:
        plt.close()
