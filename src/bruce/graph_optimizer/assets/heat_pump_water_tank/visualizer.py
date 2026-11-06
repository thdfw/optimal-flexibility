from __future__ import annotations

from datetime import datetime, timedelta

from bruce.graph_optimizer.graph import Edge, Graph, Node

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
    """Wall-clock start of each step (index 0 .. horizon); last is end of final step."""
    t0 = datetime.fromtimestamp(params.start_unix_s)
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
    prices = params.elec_price_mwh

    start_dt = step_starts[0]
    title_start = start_dt.strftime("%b %d, %Y %H:%M")
    total_cost_str = f"${round(initial_node.pathcost, 2)}"
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 3, 2]})

    ax[0].step(sp_time, sp_hp_heat_out_kw, where="post", color="tab:blue", alpha=0.6, label="HP")
    ax[0].step(sp_time[:-1], load_kw, where="post", color="tab:red", alpha=0.6, label="Load")
    ax[0].legend(loc="upper left")
    ax[0].set_ylabel("Heating power [kW]")
    if max(sp_hp_heat_out_kw) > 0:
        ax[0].set_ylim([-0.5, 1.5 * max(sp_hp_heat_out_kw)])
    ax0_price = ax[0].twinx()
    ax0_price.step(sp_time[:-1], prices, where="post", color="gray", alpha=0.6, label="Electricity price")
    ax0_price.legend(loc="upper right")
    ax0_price.set_ylabel("Electricity price [$/MWh]")

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

    ax[2].set_xlabel("Time (local)")
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
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        a.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=25, ha="right")
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
