from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from optimizer.graph import PriceQuantityPair


def plot_bid(
    pq_pairs: list[PriceQuantityPair],
    forecast_price_usd_mwh: float,
    *,
    show: bool = True,
    save_as: str | None = None,
) -> pd.DataFrame:

    if not pq_pairs:
        raise ValueError("pq_pairs is empty")

    prices_x1000 = [int(round(p.price_usd_mwh * 1000)) for p in pq_pairs]
    quantities = [p.quantity_kwh for p in pq_pairs]
    prices_usd_mwh = [p / 1000 for p in prices_x1000]

    pq_df = pd.DataFrame({"price_usd_mwh": prices_usd_mwh, "quantity_kwh": quantities})

    expected_x1000 = int(round(forecast_price_usd_mwh * 1000))
    ps: list[float] = []
    qs: list[float] = []
    index_p = 0
    intersection = (quantities[0], forecast_price_usd_mwh)

    for p in sorted(set(range(min(prices_x1000), max(prices_x1000) + 1)) | {expected_x1000}):
        ps.append(p / 1000)
        if index_p + 1 < len(prices_x1000) and p >= prices_x1000[index_p + 1]:
            index_p += 1
        if p == expected_x1000:
            intersection = (quantities[index_p], forecast_price_usd_mwh)
        qs.append(quantities[index_p])

    plt.figure(figsize=(8, 5))
    plt.plot(qs, ps, label="demand (bid)")
    plt.scatter(quantities, prices_usd_mwh)
    plt.plot(
        [min(quantities) - 1, max(quantities) + 1],
        [forecast_price_usd_mwh, forecast_price_usd_mwh],
        label="supply (expected market price)",
    )
    plt.scatter([intersection[0]], [intersection[1]])
    plt.text(
        intersection[0] + 0.25,
        intersection[1] + 15,
        f"({round(intersection[0], 3)}, {round(intersection[1], 1)})",
        fontsize=10,
        color="tab:orange",
    )
    plt.xticks(quantities)
    if min(abs(x - forecast_price_usd_mwh) for x in prices_usd_mwh) < 5:
        plt.yticks(prices_usd_mwh)
    else:
        plt.yticks(prices_usd_mwh + [forecast_price_usd_mwh])
    plt.ylabel("Price [USD/MWh]")
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
