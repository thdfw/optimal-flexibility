from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from bruce.graph_optimizer.graph import PriceQuantityPair


def plot_bid(
    pq_pairs: list[PriceQuantityPair],
    forecast_price_mwh: float,
    bid_min_price_mwh: float,
    *,
    real_price_mwh: float | None = None,
    show: bool = True,
    save_as: str | None = None,
) -> pd.DataFrame:

    if not pq_pairs:
        raise ValueError("pq_pairs is empty")

    prices_x1000 = [int(round(p.price_mwh * 1000)) for p in pq_pairs]
    quantities = [p.quantity_kwh for p in pq_pairs]
    prices_mwh = [p / 1000 for p in prices_x1000]

    pq_df = pd.DataFrame({"price_mwh": prices_mwh, "quantity_kwh": quantities})

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
    intersection = (
        quantity_at_price_mwh(forecast_price_mwh),
        forecast_price_mwh,
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

    q_xlim = (min(quantities) - 1, max(quantities) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(qs_step, ps_step, label="demand (bid)")
    plt.scatter(quantities, prices_mwh)
    plt.plot(
        q_xlim,
        [forecast_price_mwh, forecast_price_mwh],
        color="tab:orange",
        label="expected supply (day ahead price)",
    )
    plt.scatter([intersection[0]], [intersection[1]], color="tab:orange", zorder=5)
    plt.text(
        intersection[0] + 0.25,
        intersection[1] + 15,
        f"({round(intersection[0], 3)}, {round(intersection[1], 1)})",
        fontsize=10,
        color="tab:orange",
    )

    if real_price_mwh is not None:
        real_intersection = (
            quantity_at_price_mwh(real_price_mwh),
            real_price_mwh,
        )
        plt.plot(
            q_xlim,
            [real_price_mwh, real_price_mwh],
            color="tab:red",
            label="actual supply (real time price)",
        )
        plt.scatter([real_intersection[0]], [real_intersection[1]], color="tab:red", zorder=5)
        label_y = real_intersection[1] - 15
        if abs(real_intersection[1] - forecast_price_mwh) < 20:
            label_y -= 15
        plt.text(
            real_intersection[0] + 0.25,
            label_y,
            f"({round(real_intersection[0], 3)}, {round(real_intersection[1], 1)})",
            fontsize=10,
            color="tab:red",
        )

    plt.xticks(quantities)
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
    yticks = [p for p in yticks if y_bottom <= p <= y_top]
    plt.yticks(sorted(yticks, reverse=True))
    plt.ylim(y_bottom, y_top)
    plt.ylabel("Price [/MWh]")
    plt.xlabel("Quantity [kWh]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=130)
    if show:
        plt.show()
    plt.close()

    return pq_df
