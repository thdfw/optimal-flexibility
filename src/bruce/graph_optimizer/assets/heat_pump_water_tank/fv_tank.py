"""Finite-volume tank stepper for tank-1 resistive elements (layer index 0 = top)."""

from __future__ import annotations

import math

import numpy as np

from ..utils import fahrenheit_to_kelvin
from .params import HeatPumpWaterTankParams

GAL_TO_L = 3.78541
HEIGHT_M_PER_TANK = 1.75
TANKS_IN_SERIES = 3
RHO = 998.0
CP = 4186.0
K_COND = 0.6
H_BUOY = 850.0
MIN_SWT_RWT_DELTA_K = 0.05
# Coarser step for transition-matrix build (see discussion vs 2 s browser sim).
DT_S = 30.0
P_ELEM_SPREAD = 1
BUOY_THRESHOLD_K = 0.05


def _tank_height_m() -> float:
    return TANKS_IN_SERIES * HEIGHT_M_PER_TANK


def _volume_l(params: HeatPumpWaterTankParams) -> float:
    return params.storage_volume_gallons * GAL_TO_L


def _heater_indices(n: int) -> tuple[int, int]:
    i_upper = max(0, n // 9 - 1)
    i_lower = max(0, 2 * n // 9 - 1)
    return i_upper, i_lower


def _heater_power_w(i: int, i_upper: int, i_lower: int, p_elem_w: float) -> float:
    spread = P_ELEM_SPREAD
    q = 0.0
    for e in (i_upper, i_lower):
        if abs(i - e) <= spread:
            q += p_elem_w / (2 * spread + 1)
    return q


def _flow_lps_from_load(
    load_power_w: float, swt_k: float, params: HeatPumpWaterTankParams, bucket: int
) -> float:
    if load_power_w <= 0:
        return 0.0
    swt_f = (swt_k - 273.15) * 9 / 5 + 32
    rwt_f = params.rwt(swt_f, bucket)
    rwt_k = fahrenheit_to_kelvin(rwt_f)
    delta_k = swt_k - rwt_k
    if delta_k <= MIN_SWT_RWT_DELTA_K:
        return 0.0
    return load_power_w / (CP * delta_k)


def _advect_upward(
    t: np.ndarray,
    load_power_w: float,
    vol_layer_l: float,
    dt: float,
    params: HeatPumpWaterTankParams,
    bucket: int,
) -> None:
    if load_power_w <= 0:
        return
    elapsed = 0.0
    guard = 0
    n = t.shape[0]
    while elapsed < dt - 1e-12 and guard < 20_000:
        guard += 1
        swt_k = float(t[0])
        flow_lps = _flow_lps_from_load(load_power_w, swt_k, params, bucket)
        if flow_lps <= 0:
            break
        c = flow_lps / vol_layer_l
        sub_dt = min(dt - elapsed, max(1e-4, 0.99 / c))
        swt_f = (swt_k - 273.15) * 9 / 5 + 32
        inlet_k = fahrenheit_to_kelvin(params.rwt(swt_f, bucket))
        next_t = t.copy()
        next_t[n - 1] = t[n - 1] + sub_dt * c * (inlet_k - t[n - 1])
        for i in range(n - 1):
            next_t[i] = t[i] + sub_dt * c * (t[i + 1] - t[i])
        t[:] = next_t
        elapsed += sub_dt


def simulate_elements_step(
    layers_f: list[float],
    params: HeatPumpWaterTankParams,
    *,
    duration_s: float,
    load_kwh: float,
    rwt_bucket: int,
) -> list[float]:
    """
    Advance layer temperatures (°F, index 0 = top) with resistive heat + circulation.

    ``load_kwh`` is thermal load over the step (kWh); for 1 h steps this is kWh/h ≈ kW.
    """
    n = params.num_layers
    if len(layers_f) != n:
        raise ValueError(f"expected {n} layer temps, got {len(layers_f)}")

    vol_l = _volume_l(params)
    h = _tank_height_m()
    dz = h / n
    vol_layer_l = vol_l / n
    m = (RHO * (vol_l / 1000)) / n
    area = (vol_l / 1000) / h

    t = np.array([fahrenheit_to_kelvin(tf) for tf in layers_f], dtype=np.float64)
    i_upper, i_lower = _heater_indices(n)
    p_elem_w = (params.tank1_elements_kw * 1000) / 2
    load_power_w = max(0.0, load_kwh) * 1000.0

    n_steps = max(1, int(math.ceil(duration_s / DT_S)))
    for step_i in range(n_steps):
        remaining = duration_s - step_i * DT_S
        dt = min(DT_S, remaining)

        d_t = np.zeros(n)
        for i in range(n):
            d_t[i] += _heater_power_w(i, i_upper, i_lower, p_elem_w) / (m * CP)

        for i in range(n - 1):
            flux_cond = (K_COND * area * (t[i + 1] - t[i])) / dz
            d_t[i] += flux_cond / (m * CP)
            d_t[i + 1] -= flux_cond / (m * CP)

            d = t[i + 1] - t[i]
            if d > BUOY_THRESHOLD_K:
                flux_buoy = H_BUOY * area * d
                d_t[i] += flux_buoy / (m * CP)
                d_t[i + 1] -= flux_buoy / (m * CP)

        t += d_t * dt
        _advect_upward(t, load_power_w, vol_layer_l, dt, params, rwt_bucket)

    return [(tk - 273.15) * 9 / 5 + 32 for tk in t.tolist()]
