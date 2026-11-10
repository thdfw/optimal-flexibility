from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from bruce.graph_optimizer.graph import PriceQuantityPair


def _x_tick_quantities_kwh(quantities: list[float], hp_max_quantity_kwh: float) -> list[float]:
    ticks = list(quantities)
    if not any(abs(q - hp_max_quantity_kwh) < 0.001 for q in ticks):
        ticks.append(hp_max_quantity_kwh)
    return sorted(ticks)


@dataclass
class _BidGeometry:
    prices_mwh: list[float]
    quantities: list[float]
    step_quantities: list[float]
    step_prices: list[float]
    q_xlim: tuple[float, float]
    forecast_intersection: tuple[float, float]
    real_intersection: tuple[float, float] | None
    yticks: list[float]
    y_range: tuple[float, float]


def _bid_geometry(
    pq_pairs: list[PriceQuantityPair],
    forecast_price_mwh: float,
    bid_min_price_mwh: float,
    real_price_mwh: float | None,
    *,
    max_quantity_kwh: float,
) -> _BidGeometry:
    if not pq_pairs:
        raise ValueError("pq_pairs is empty")

    prices_x1000 = [int(round(p.price_mwh * 1000)) for p in pq_pairs]
    quantities = [p.quantity_kwh for p in pq_pairs]
    prices_mwh = [p / 1000 for p in prices_x1000]

    def quantity_at_price_mwh(price_mwh: float) -> float:
        """Pair i applies on (price_{i+1}, price_i] (pairs are sorted by descending price)."""
        if price_mwh > prices_mwh[0]:
            return quantities[0]
        for i, pq in enumerate(pq_pairs):
            if i + 1 < len(pq_pairs):
                floor_mwh = pq_pairs[i + 1].price_mwh
                if floor_mwh < price_mwh <= pq.price_mwh:
                    return pq.quantity_kwh
            elif bid_min_price_mwh <= price_mwh <= pq.price_mwh:
                return pq.quantity_kwh
        return quantities[-1]

    plot_price_min_mwh = bid_min_price_mwh
    forecast_intersection = (
        quantity_at_price_mwh(forecast_price_mwh),
        forecast_price_mwh,
    )
    real_intersection = (
        (quantity_at_price_mwh(real_price_mwh), real_price_mwh)
        if real_price_mwh is not None
        else None
    )

    ps_step: list[float] = []
    qs_step: list[float] = []
    for i in range(len(pq_pairs)):
        q = quantities[i]
        p_top = prices_mwh[i]
        p_bottom = prices_mwh[i + 1] if i + 1 < len(prices_mwh) else plot_price_min_mwh
        if i == 0:
            ps_step.append(p_top)
            qs_step.append(q)
        ps_step.append(p_bottom)
        qs_step.append(q)
        if i + 1 < len(pq_pairs):
            ps_step.append(p_bottom)
            qs_step.append(quantities[i + 1])

    q_xlim = (min(quantities) - 1, max_quantity_kwh)

    yticks = list(prices_mwh)
    if all(abs(p - bid_min_price_mwh) > 0.01 for p in yticks):
        yticks.append(bid_min_price_mwh)

    # Ceiling and floor are market bounds, often far from the bid. Crop each end
    # at the nearest supply line or other PQ price, with room to still see the
    # quantity that holds beyond that point.
    ceiling_mwh = prices_mwh[0]
    floor_mwh = bid_min_price_mwh
    supplies = [forecast_price_mwh]
    if real_price_mwh is not None:
        supplies.append(real_price_mwh)
    below_ceiling = [p for p in [*prices_mwh[1:], *supplies] if p < ceiling_mwh - 0.01]
    above_floor = [p for p in [prices_mwh[-1], *supplies] if p > floor_mwh + 0.01]
    top_feature = max(below_ceiling) if below_ceiling else None
    bottom_feature = min(above_floor) if above_floor else None
    span = max(
        (top_feature if top_feature is not None else ceiling_mwh)
        - (bottom_feature if bottom_feature is not None else floor_mwh),
        1.0,
    )
    margin = max(0.45 * span, 60.0)
    pad = 0.04 * span
    y_top = (
        top_feature + margin
        if top_feature is not None and ceiling_mwh > top_feature + margin
        else ceiling_mwh + pad
    )
    y_bottom = (
        bottom_feature - margin
        if bottom_feature is not None and floor_mwh < bottom_feature - margin
        else floor_mwh - pad
    )
    yticks = sorted((p for p in yticks if y_bottom <= p <= y_top), reverse=True)

    return _BidGeometry(
        prices_mwh=prices_mwh,
        quantities=quantities,
        step_quantities=qs_step,
        step_prices=ps_step,
        q_xlim=q_xlim,
        forecast_intersection=forecast_intersection,
        real_intersection=real_intersection,
        yticks=yticks,
        y_range=(y_bottom, y_top),
    )


def bid_plot_data(
    pq_pairs: list[PriceQuantityPair],
    forecast_price_mwh: float,
    bid_min_price_mwh: float,
    *,
    max_quantity_kwh: float,
    hp_max_quantity_kwh: float,
    real_price_mwh: float | None = None,
) -> dict[str, Any]:
    """Series for the bid plot (same content as ``plot_bid``), ready for plotly."""
    geo = _bid_geometry(
        pq_pairs,
        forecast_price_mwh,
        bid_min_price_mwh,
        real_price_mwh,
        max_quantity_kwh=max_quantity_kwh,
    )

    def supply(price_mwh: float, intersection: tuple[float, float]) -> dict[str, Any]:
        return {
            "priceMwh": float(price_mwh),
            "intersection": [float(intersection[0]), float(intersection[1])],
        }

    return {
        "pairs": {
            "quantityKwh": [float(q) for q in geo.quantities],
            "priceMwh": [float(p) for p in geo.prices_mwh],
        },
        "step": {
            "quantityKwh": [float(q) for q in geo.step_quantities],
            "priceMwh": [float(p) for p in geo.step_prices],
        },
        "forecast": supply(forecast_price_mwh, geo.forecast_intersection),
        "real": (
            supply(real_price_mwh, geo.real_intersection)
            if real_price_mwh is not None and geo.real_intersection is not None
            else None
        ),
        "xRange": [float(geo.q_xlim[0]), float(geo.q_xlim[1])],
        "xTickQuantitiesKwh": _x_tick_quantities_kwh(geo.quantities, hp_max_quantity_kwh),
        "yRange": [float(geo.y_range[0]), float(geo.y_range[1])],
        "yTicks": [float(p) for p in geo.yticks],
    }


def plot_bid(
    pq_pairs: list[PriceQuantityPair],
    forecast_price_mwh: float,
    bid_min_price_mwh: float,
    *,
    max_quantity_kwh: float,
    hp_max_quantity_kwh: float,
    real_price_mwh: float | None = None,
    show: bool = True,
    save_as: str | None = None,
) -> pd.DataFrame:
    geo = _bid_geometry(
        pq_pairs,
        forecast_price_mwh,
        bid_min_price_mwh,
        real_price_mwh,
        max_quantity_kwh=max_quantity_kwh,
    )
    quantities = geo.quantities
    prices_mwh = geo.prices_mwh
    q_xlim = geo.q_xlim
    intersection = geo.forecast_intersection

    pq_df = pd.DataFrame({"price_mwh": prices_mwh, "quantity_kwh": quantities})

    plt.figure(figsize=(8, 5))
    plt.plot(geo.step_quantities, geo.step_prices, label="Demand (bid)")
    plt.scatter(quantities, prices_mwh)
    plt.plot(
        q_xlim,
        [forecast_price_mwh, forecast_price_mwh],
        color="tab:orange",
        label=f"Supply, expected ({round(forecast_price_mwh, 1)} $/MWh)",
    )
    plt.scatter(
        [intersection[0]],
        [intersection[1]],
        color="tab:orange",
        zorder=5,
        label=f"Market clearing, expected ({round(intersection[0], 3)} kWh)",
    )

    if real_price_mwh is not None and geo.real_intersection is not None:
        real_intersection = geo.real_intersection
        plt.plot(
            q_xlim,
            [real_price_mwh, real_price_mwh],
            color="tab:red",
            label=f"Supply, actual ({round(real_price_mwh, 1)} $/MWh)",
        )
        plt.scatter(
            [real_intersection[0]],
            [real_intersection[1]],
            color="tab:red",
            zorder=5,
            label=f"Market clearing, actual ({round(real_intersection[0], 3)} kWh)",
        )

    plt.xticks(_x_tick_quantities_kwh(quantities, hp_max_quantity_kwh))
    plt.xlim(*q_xlim)
    plt.yticks(geo.yticks)
    plt.ylim(*geo.y_range)
    plt.ylabel("Price [/MWh]")
    plt.xlabel("Quantity [kWh]")
    plt.grid(alpha=0.3)
    legend_order = [
        "Demand (bid)",
        f"Supply, expected ({round(forecast_price_mwh, 1)} $/MWh)",
    ]
    if real_price_mwh is not None and geo.real_intersection is not None:
        real_intersection = geo.real_intersection
        legend_order.append(f"Supply, actual ({round(real_price_mwh, 1)} $/MWh)")
    legend_order.append(f"Market clearing, expected ({round(intersection[0], 3)} kWh)")
    if real_price_mwh is not None and geo.real_intersection is not None:
        legend_order.append(
            f"Market clearing, actual ({round(real_intersection[0], 3)} kWh)",
        )
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    plt.legend([by_label[label] for label in legend_order], legend_order)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=130)
    if show:
        plt.show()
    plt.close()

    return pq_df
