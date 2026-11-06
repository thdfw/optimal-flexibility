"""Turn one journaled ``flo.params`` payload into the heat-pump graph plot.

The payload is the PascalCase object the journal stores. Tuning must be
``flo.tuning.bruce``. The plot is written to disk with the GUI backend off.
"""

from __future__ import annotations

import gc
from contextlib import suppress
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from bruce.graph_optimizer import Graph
from bruce.graph_optimizer.assets.heat_pump_water_tank import (
    HeatPumpWaterTankAsset,
    HeatPumpWaterTankParams,
    plot_graph_results,
)


class FloParamsError(ValueError):
    """The payload is not a bruce ``flo.params``."""


class _FloOptimalPath(BaseModel):
    model_config = ConfigDict(extra="ignore")

    StartUnixS: int
    StepDurationMinutes: list[float]
    HpAvgWList: list[float]
    StoreWhList: list[float]

    @model_validator(mode="after")
    def _same_length(self) -> Self:
        n = len(self.StepDurationMinutes)
        if n == 0 or len(self.HpAvgWList) != n or len(self.StoreWhList) != n:
            raise ValueError(
                "PreviousOptimalPath lists must be non-empty and the same length: "
                f"StepDurationMinutes={n}, HpAvgWList={len(self.HpAvgWList)}, "
                f"StoreWhList={len(self.StoreWhList)}"
            )
        return self


class _FloTuningBruce(BaseModel):
    model_config = ConfigDict(extra="ignore")

    TypeName: str
    NumLayers: int
    StorageLossesPercent: float
    HpMinKwEl: float
    HpConstantLiftF: float
    LoadBucketUpperKwhList: list[float]
    RwtInterceptByBucketList: list[float]
    RwtSlopeByBucketList: list[float]
    RwtMinF: float
    HpMaxKwTh: float
    MaxLoadKwTh: float
    RswtPenaltyEnabled: bool
    RswtPenaltyWeight: float
    RswtPenaltyDecay: float
    RswtPenaltyExponentRate: float
    RswtPenaltyDecayMaxHour: int
    ShortCyclePenaltyEnabled: bool
    ShortCyclePenaltyWeight: float
    StabilityPenaltyEnabled: bool
    StabilityPenaltyWeight: float
    StabilityPenaltyDecay: float
    StabilityPenaltyThresholdKwh: float
    StabilityPenaltyThresholdPriceMwh: float
    StabilityPenaltyHorizonHours: int
    TerminalCostEnabled: bool
    TerminalCostEndWindowHours: int

    @model_validator(mode="after")
    def _is_bruce(self) -> Self:
        if self.TypeName != "flo.tuning.bruce":
            raise ValueError(
                f"flo.params Tuning must be flo.tuning.bruce, got {self.TypeName!r}"
            )
        return self


class _FloParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    TypeName: str
    GNodeAlias: str
    StartUnixS: int
    StepDurationMinutes: list[float]
    LmpUsdMwhList: list[float]
    DistUsdMwhList: list[float]
    PriceMinUsdMwh: float
    PriceMaxUsdMwh: float
    GallonsPerTank: float
    TanksModeled: int
    MaxTankTempF: float
    HpMaxKwEl: float
    HpTurnOnMinutes: int
    CopIntercept: float
    CopOatCoeff: float
    CopLwtCoeff: float
    CopMin: float
    CopMinOatF: float
    HpCurrentlyOn: bool
    InitialTopTempF: float
    InitialMiddleTempF: float
    InitialBottomTempF: float
    InitialThermocline1: int
    InitialThermocline2: int
    TempSensorsPerTank: int
    RswtFList: list[float]
    LoadKwhList: list[float]
    OatFList: list[float]
    Tuning: _FloTuningBruce
    PreviousOptimalPath: _FloOptimalPath | None = None
    PreviousEstimateElecUsdMwhNow: float | None = None

    @model_validator(mode="after")
    def _is_flo_params(self) -> Self:
        if self.TypeName != "flo.params":
            raise ValueError(f"expected TypeName 'flo.params', got {self.TypeName!r}")
        n = len(self.StepDurationMinutes)
        lengths = {
            "LmpUsdMwhList": len(self.LmpUsdMwhList),
            "DistUsdMwhList": len(self.DistUsdMwhList),
            "RswtFList": len(self.RswtFList),
            "LoadKwhList": len(self.LoadKwhList),
            "OatFList": len(self.OatFList),
        }
        if n == 0 or any(length != n for length in lengths.values()):
            raise ValueError(
                "flo.params per-step lists must be non-empty and the same length: "
                f"StepDurationMinutes={n}, "
                + ", ".join(f"{name}={length}" for name, length in lengths.items())
            )
        return self


def _step_start_unix_s(start_unix_s: int, step_minutes: list[float], index: int) -> float:
    return start_unix_s + sum(step_minutes[:index]) * 60.0


def _index_at_unix_s(start_unix_s: int, step_minutes: list[float], target_unix_s: float) -> int:
    if not step_minutes or target_unix_s <= start_unix_s:
        return 0
    for i in range(len(step_minutes)):
        if target_unix_s < _step_start_unix_s(start_unix_s, step_minutes, i + 1):
            return i
    return len(step_minutes) - 1


def _align_previous_path(
    current_start_unix_s: int,
    horizon: int,
    previous: _FloOptimalPath,
) -> tuple[list[float], float]:
    """The previous run's heat-pump plan and stored energy, lined up on this run's start."""
    prev_minutes = [float(m) for m in previous.StepDurationMinutes]
    hp_kwh = [
        float(watts) * (minutes / 60.0) / 1000.0
        for watts, minutes in zip(previous.HpAvgWList, prev_minutes, strict=True)
    ]
    align_index = _index_at_unix_s(int(previous.StartUnixS), prev_minutes, float(current_start_unix_s))
    storage_kwh_now = float(previous.StoreWhList[align_index]) / 1000.0
    plan = hp_kwh[align_index : align_index + horizon]
    if len(plan) < horizon:
        plan.extend([0.0] * (horizon - len(plan)))
    return plan, storage_kwh_now


def heat_pump_params_from_flo(payload: dict[str, Any]) -> HeatPumpWaterTankParams:
    """The optimizer params for one ``flo.params`` object."""
    if not isinstance(payload, dict):
        raise FloParamsError(f"flo.params payload must be an object, got {type(payload).__name__}")
    if payload.get("TypeName") != "flo.params":
        raise FloParamsError(f"expected TypeName 'flo.params', got {payload.get('TypeName')!r}")
    tuning = payload.get("Tuning")
    tuning_name = tuning.get("TypeName") if isinstance(tuning, dict) else None
    if tuning_name != "flo.tuning.bruce":
        raise FloParamsError(f"flo.params Tuning must be flo.tuning.bruce, got {tuning_name!r}")
    try:
        flo = _FloParams.model_validate(payload)
    except ValidationError as e:
        raise FloParamsError(str(e)) from e

    tuning = flo.Tuning
    step_minutes = [float(m) for m in flo.StepDurationMinutes]
    horizon = len(step_minutes)
    previous_plan: list[float] | None = None
    previous_storage_kwh: float | None = None
    if flo.PreviousOptimalPath is not None:
        previous_plan, previous_storage_kwh = _align_previous_path(
            flo.StartUnixS,
            horizon,
            flo.PreviousOptimalPath,
        )

    return HeatPumpWaterTankParams(
        horizon=horizon,
        start_unix_s=flo.StartUnixS,
        site_id=flo.GNodeAlias,
        timestep_duration_hours=[m / 60.0 for m in step_minutes],
        elec_price_mwh=[a + b for a, b in zip(flo.LmpUsdMwhList, flo.DistUsdMwhList, strict=True)],
        bid_min_price_mwh=flo.PriceMinUsdMwh,
        bid_max_price_mwh=flo.PriceMaxUsdMwh,
        storage_volume_gallons=flo.GallonsPerTank * flo.TanksModeled,
        storage_losses_percent=tuning.StorageLossesPercent,
        max_tank_temp_f=flo.MaxTankTempF,
        hp_min_kw_elec=tuning.HpMinKwEl,
        hp_max_kw_elec=flo.HpMaxKwEl,
        hp_turn_on_minutes=flo.HpTurnOnMinutes,
        hp_constant_lift_f=tuning.HpConstantLiftF,
        cop_intercept=flo.CopIntercept,
        cop_oat_coeff=flo.CopOatCoeff,
        cop_lwt_coeff=flo.CopLwtCoeff,
        cop_min=flo.CopMin,
        cop_min_oat_f=flo.CopMinOatF,
        load_bucket_upper_kwh=list(tuning.LoadBucketUpperKwhList),
        rwt_intercept_by_bucket=list(tuning.RwtInterceptByBucketList),
        rwt_slope_by_bucket=list(tuning.RwtSlopeByBucketList),
        rwt_min=tuning.RwtMinF,
        hp_max_kw_th=tuning.HpMaxKwTh,
        max_load_kw_th=tuning.MaxLoadKwTh,
        rswt_penalty_enabled=tuning.RswtPenaltyEnabled,
        rswt_penalty_weight=tuning.RswtPenaltyWeight,
        rswt_penalty_decay=tuning.RswtPenaltyDecay,
        rswt_penalty_exponent_rate=tuning.RswtPenaltyExponentRate,
        rswt_penalty_decay_max_hour=tuning.RswtPenaltyDecayMaxHour,
        short_cycle_penalty_enabled=tuning.ShortCyclePenaltyEnabled,
        short_cycle_penalty_weight=tuning.ShortCyclePenaltyWeight,
        stability_penalty_enabled=tuning.StabilityPenaltyEnabled,
        stability_penalty_weight=tuning.StabilityPenaltyWeight,
        stability_penalty_decay=tuning.StabilityPenaltyDecay,
        stability_penalty_threshold_kwh=tuning.StabilityPenaltyThresholdKwh,
        stability_penalty_threshold_price_mwh=tuning.StabilityPenaltyThresholdPriceMwh,
        stability_penalty_horizon_hours=tuning.StabilityPenaltyHorizonHours,
        previous_plan_hp_kwh_el_list=previous_plan,
        previous_estimate_storage_kwh_now=previous_storage_kwh,
        previous_estimate_elec_price_mwh_now=flo.PreviousEstimateElecUsdMwhNow,
        terminal_cost_enabled=tuning.TerminalCostEnabled,
        terminal_cost_end_window_hours=tuning.TerminalCostEndWindowHours,
        hp_currently_on=flo.HpCurrentlyOn,
        initial_top_temp=flo.InitialTopTempF,
        initial_middle_temp=flo.InitialMiddleTempF,
        initial_bottom_temp=flo.InitialBottomTempF,
        initial_thermocline1=flo.InitialThermocline1,
        initial_thermocline2=flo.InitialThermocline2,
        rswt_f=list(flo.RswtFList),
        load_kwh=list(flo.LoadKwhList),
        oat_f=list(flo.OatFList),
        num_layers=tuning.NumLayers,
        temp_sensors_per_tank=flo.TempSensorsPerTank,
    )


def _use_headless_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)


def render_flo_params_png(payload: dict[str, Any], save_as: str | Path) -> None:
    """Solve the heat-pump graph for ``payload`` and write the plot to ``save_as``."""
    params = heat_pump_params_from_flo(payload)
    _use_headless_matplotlib()
    asset = HeatPumpWaterTankAsset(params)
    graph: Graph[Any, Any, Any] = Graph(asset)
    try:
        graph.find_shortest_path()
        plot_graph_results(graph, show=False, save_as=str(save_as))
    finally:
        with suppress(Exception):
            graph.cleanup()
        del graph
        del asset
        gc.collect()
