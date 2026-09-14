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
    stability_penalty_horizon_hours: int = 20
    previous_plan_hp_kwh_el_list: list[float] | None = None
    previous_estimate_storage_kwh_now: float | None = None

    # Initial state
    hp_currently_on: bool = True
    initial_top_temp: float = 120
    initial_middle_temp: float = 110
    initial_bottom_temp: float = 100
    initial_thermocline1: int = 1
    initial_thermocline2: int = 2

    # Forecasts
    elec_usd_mwh: list[float]
    rswt_f: list[float]
    load_kwh: list[float]
    oat_f: list[float]

    def delta_T(self, swt: float) -> float:
        return self.hp_constant_lift_f

    def COP(self, oat: float) -> float:
        if oat < self.cop_min_oat_f:
            return self.cop_min
        else:
            return self.cop_intercept + self.cop_oat_coeff * oat
