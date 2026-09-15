from __future__ import annotations

from optimizer.graph import Edge, Graph, Node

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
    asset = _graph_asset(graph)
    initial = HeatPumpWaterTankState.build(
        top_temp=asset.params.initial_top_temp,
        middle_temp=asset.params.initial_middle_temp,
        bottom_temp=asset.params.initial_bottom_temp,
        thermocline1=asset.params.initial_thermocline1,
        thermocline2=asset.params.initial_thermocline2,
        params=asset.params,
    )
    closest = asset.closest_state(initial)
    return graph.nodes_by[0][closest]


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


def _hp_heat_out_kwh(
    asset: HeatPumpWaterTankAsset,
    state: HeatPumpWaterTankState,
    action: HeatPumpWaterTankAction,
    time_step: int,
) -> float:
    cop = asset.params.COP(asset.params.oat_f[time_step])
    return asset.elec_used_kwh(state, action, time_step) * cop


def _rswt_penalty_for_edge(
    asset: HeatPumpWaterTankAsset,
    state: HeatPumpWaterTankState,
    next_state: HeatPumpWaterTankState,
    action: HeatPumpWaterTankAction,
    time_step: int,
) -> float:
    load = asset.params.load_kwh[time_step]
    rswt = asset.params.rswt_f[time_step]
    if action.heat_to_store_kwh >= 0 or load <= 0:
        return 0.0
    if state.top_temp >= rswt and next_state.top_temp >= rswt:
        return 0.0

    if state.top_temp == next_state.top_temp:
        swt = state.top_temp
    else:
        if state.thermocline2 > state.thermocline1:
            temp_below_top = state.middle_temp
            num_layers_available = state.thermocline2 - state.thermocline1
        else:
            temp_below_top = state.bottom_temp
            num_layers_available = asset.params.num_layers - state.thermocline2
        if next_state.top_temp == temp_below_top:
            num_layers_used = max(0, num_layers_available - next_state.thermocline1)
            swt = (state.top_temp * state.thermocline1 + temp_below_top * num_layers_used) / (
                state.thermocline1 + num_layers_used
            )
        elif next_state.top_temp < temp_below_top:
            swt = (state.top_temp + temp_below_top + next_state.top_temp) / 3
        else:
            swt = next_state.top_temp

    return asset.rswt_penalty(time_step, swt, rswt)


def plot_graph_results(
    graph: HpGraph,
    *,
    show: bool = True,
    save_as: str | None = None,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    asset = _graph_asset(graph)
    params = asset.params
    initial_node = _initial_node(graph)

    sp_top_temp: list[float] = []
    sp_middle_temp: list[float] = []
    sp_bottom_temp: list[float] = []
    sp_thermocline1: list[int] = []
    sp_thermocline2: list[int] = []
    sp_hp_heat_out_kw: list[float] = []
    sp_stored_energy: list[float] = []
    sp_rswt_penalty: list[float] = []
    sp_stability_penalty: list[float] = []

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
            sp_rswt_penalty.append(
                _rswt_penalty_for_edge(asset, node.state, edge.head.state, edge.action, t)
            )
            sp_stability_penalty.append(asset.stability_penalty(t, heat_out / cop))

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
    sp_time = list(range(params.horizon + 1))

    load_kw = [
        params.load_kwh[t] / params.timestep_duration_hours[t]
        for t in range(params.horizon)
    ]
    prices = params.elec_usd_mwh

    fig, ax = plt.subplots(3, 1, sharex=False, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 3, 2]})
    fig.suptitle(f"Horizon: {params.horizon} steps — Cost: {round(initial_node.pathcost, 2)} $", fontsize=10)

    penalties = [
        ("RSWT penalty", params.rswt_penalty_enabled),
        ("Stability penalty", params.stability_penalty_enabled),
    ]
    for idx, (label, enabled) in enumerate(penalties):
        color = "green" if enabled else "red"
        fig.text(
            0.99,
            0.99 - idx * 0.045,
            f"\u25cf {label}",
            ha="right",
            va="top",
            fontsize=7.5,
            color=color,
            transform=fig.transFigure,
        )

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

    bars_top = ax[1].bar(
        sp_time, sp_thermocline1, bottom=th1_rev, color=tank_top_colors, alpha=0.7, width=0.9, align="edge"
    )
    bars_middle = ax[1].bar(
        sp_time,
        [y - x for x, y in zip(sp_thermocline1, sp_thermocline2)],
        bottom=th2_rev,
        color=tank_middle_colors,
        alpha=0.7,
        width=0.9,
        align="edge",
    )
    bars_bottom = ax[1].bar(
        sp_time, th2_rev, bottom=0, color=tank_bottom_colors, alpha=0.7, width=0.9, align="edge"
    )
    ax[1].set_ylabel("Storage state")
    ax[1].set_ylim([0, params.num_layers])
    ax[1].set_yticks([])

    if 10 < len(sp_time) < 50:
        hour_labels = [f"{t}h" for t in range(0, len(sp_time) + 1, 2)]
        for a in ax:
            a.set_xticks(list(range(0, len(sp_time) + 1, 2)))
            a.set_xticklabels(hour_labels, fontsize=8)

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
    ax1_soc.plot(sp_time, sp_soc, color="black", alpha=0.4, label="SoC")
    ax1_soc.set_ylabel("State of charge [%]")
    ax1_soc.set_ylim([-1, 101])

    sp_time_mid = [t + 0.5 for t in sp_time[:-1]]
    rswt_vals = sp_rswt_penalty[:-1]
    stab_vals = sp_stability_penalty[:-1]
    ax[2].bar(sp_time_mid, rswt_vals, width=0.9, color="tab:orange", alpha=0.7, label="RSWT penalty")
    ax[2].bar(sp_time_mid, stab_vals, bottom=rswt_vals, width=0.9, color="tab:purple", alpha=0.7, label="Stability penalty")
    ax[2].set_xlabel("Time step")
    ax[2].set_ylabel("Penalty [$]")
    ax[2].legend(loc="upper right", fontsize=8)
    max_penalty = max(max(rswt_vals, default=0), max(stab_vals, default=0))
    if max_penalty > 0:
        ax[2].set_ylim([0, max_penalty * 1.3])

    xlim = ax[1].get_xlim()
    ax[0].set_xlim(xlim)
    ax[2].set_xlim(xlim)

    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as, dpi=130)
    if show:
        plt.show()
    else:
        plt.close()
