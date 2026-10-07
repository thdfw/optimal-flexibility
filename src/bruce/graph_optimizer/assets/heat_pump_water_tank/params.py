from ..base import Params


class HeatPumpWaterTankParams(Params):
    horizon: int = 48
    
    # Storage
    num_layers: int = 27
    storage_volume_gallons: float = 360
    storage_losses_percent: float = 0.5
    max_tank_temp_f: float = 160

    # Heat pump
    hp_min_kw_elec: float = 0
    hp_max_kw_elec: float = 9.66
    hp_min_kw_th_first_step: float = 5
    hp_min_kw_th_other_steps: float = 10
    hp_turn_on_minutes: int = 12
    hp_constant_lift_f: float = 20
    cop_intercept: float = 1.02
    cop_oat_coeff: float = 0.0257
    cop_lwt_coeff: float = 0
    cop_min: float = 1.4
    cop_min_oat_f: float = 15

    # Distribution system
    rwt_intercept: float = 44.5
    rwt_slope: float = 0.4684
    rwt_min: float = 50

    # Action range (storage change)
    hp_max_kw_th: float = 25
    max_load_kw_th: float = 20

    # RSWT penalty
    rswt_penalty_enabled: bool = True
    rswt_penalty_weight: float = 0.3
    rswt_penalty_decay: float = 0.9
    rswt_penalty_exponent_rate: float = 0.15
    rswt_penalty_decay_max_hour: int = 12

    # Plan stability penalty
    stability_penalty_enabled: bool = True
    stability_penalty_weight: float = 0.5
    stability_penalty_decay: float = 0.75
    stability_penalty_threshold_kwh: float = 10.0
    stability_penalty_threshold_price_mwh: float = 15.0
    stability_penalty_horizon_hours: int = 20
    previous_plan_hp_kwh_el_list: list[float] | None = None
    previous_estimate_storage_kwh_now: float | None = None
    previous_estimate_elec_price_mwh_now: float | None = None

    # Initial state
    hp_currently_on: bool = True
    initial_top_temp: float = 160
    initial_middle_temp: float = 160
    initial_bottom_temp: float = 100
    initial_thermocline1: int = 27
    initial_thermocline2: int = 27

    # Forecasts
    rswt_f: list[float]
    load_kwh: list[float]
    oat_f: list[float]

    def delta_T(self, swt: float) -> float:
        return self.hp_constant_lift_f

    def rwt(self, swt_f: float) -> float:
        predicted_rwt_f = self.rwt_intercept + self.rwt_slope * swt_f
        if self.rwt_min > swt_f:
            return swt_f
        return max(self.rwt_min, min(predicted_rwt_f, swt_f))

    def COP(self, oat: float) -> float:
        if oat < self.cop_min_oat_f:
            return self.cop_min
        else:
            return self.cop_intercept + self.cop_oat_coeff * oat

    def validate_bid_params_update(self, updated: Params) -> None:
        if not isinstance(updated, HeatPumpWaterTankParams):
            raise TypeError(f"expected HeatPumpWaterTankParams, got {type(updated).__name__}")
        allowed_fields = frozenset({
            "initial_top_temp",
            "initial_middle_temp",
            "initial_bottom_temp",
            "initial_thermocline1",
            "initial_thermocline2",
        })
        old_dict = self.model_dump()
        new_dict = updated.model_dump()
        disallowed_diffs: dict[str, tuple[object, object]] = {}
        for key, old_val in old_dict.items():
            new_val = new_dict.get(key)
            if old_val != new_val and key not in allowed_fields:
                disallowed_diffs[key] = (old_val, new_val)
        if disallowed_diffs:
            diff_msg = "\n".join(
                f"  {key}: {before} → {after}" for key, (before, after) in disallowed_diffs.items()
            )
            raise ValueError(f"Disallowed params update for bid generation:\n{diff_msg}")