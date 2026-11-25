from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np

from bruce.graph_optimizer.settings import get_logger

from ..base import Asset, TransitionMatrixParams
from ..utils import kelvin_to_fahrenheit

logger = get_logger("assets.heat_pump_water_tank")
from .action import HeatPumpWaterTankAction
from .params import NET_HEAT_TO_STORE_MIN_DISCHARGE_KWH, HeatPumpWaterTankParams
from .state import HeatPumpWaterTankState
from .transition_matrix_params import HeatPumpWaterTankTransitionMatrixParams

if TYPE_CHECKING:
    from .model import HeatPumpWaterTankModel

OTHER_SIDE_MAX_WINDOW_KWH = 2.0
COLDEST_TANK_TEMP_F = 70


class HeatPumpWaterTankAsset(Asset[HeatPumpWaterTankState, HeatPumpWaterTankAction, HeatPumpWaterTankParams]):

    def __init__(self, params: HeatPumpWaterTankParams):
        super().__init__(params)
        self._refresh_horizon_median_elec_price_kwh()
        self._refresh_max_heat_price_kwh()
        self._compute_difference_with_plans()
        self._available_actions_cache: list[
            dict[tuple[int, int], tuple[tuple[HeatPumpWaterTankAction, int], ...]]
        ] = []
        self._fill_to_full_action_by_state: dict[int, HeatPumpWaterTankAction | None] = {}
        self._build_available_actions_cache()

    def on_params_updated(self) -> None:
        logger.info(json.dumps(self.params.model_dump(mode="json"), separators=(",", ":")))
        self._refresh_horizon_median_elec_price_kwh()
        self._refresh_max_heat_price_kwh()
        self._compute_difference_with_plans()
        self._fill_to_full_action_by_state = {}
        self._build_available_actions_cache()

    @staticmethod
    def _median_kwh(prices_kwh: list[float]) -> float:
        if not prices_kwh:
            return 0.0
        sorted_prices = sorted(prices_kwh)
        n = len(sorted_prices)
        mid = n // 2
        if n % 2:
            return sorted_prices[mid]
        return (sorted_prices[mid - 1] + sorted_prices[mid]) / 2

    def _refresh_horizon_median_elec_price_kwh(self) -> None:
        """Median step electricity price over the horizon ($/kWh)."""
        prices_kwh = [p / 1000 for p in self.params.elec_price_mwh]
        self._horizon_median_elec_price_kwh = self._median_kwh(prices_kwh)

    def _refresh_max_heat_price_kwh(self) -> None:
        """Penalty price of heat ($/kWh thermal).

        COP is evaluated at ``max_tank_temp_f`` as the leaving-water temperature.
        Operating cost uses the midpoint bulk temperature plus half the lift instead.
        """
        self._max_heat_price_kwh = max(
            0.0,
            max(
                (
                    p / 1000 / self.params.COP(oat, self.params.max_tank_temp_f)
                    for p, oat in zip(self.params.elec_price_mwh, self.params.oat_f)
                ),
                default=0.0,
            ),
        )

    @property
    def name(self) -> str:
        return "heat_pump_water_tank"

    def transition_matrix_params(self) -> TransitionMatrixParams:
        return HeatPumpWaterTankTransitionMatrixParams.from_asset_params(self.params)

    def get_state_space(self) -> list[HeatPumpWaterTankState]:
        top_temps = sorted(range(90,170+10,10), reverse=True)
        middle_temps = [x-10 for x in top_temps[:-1]]
        bottom_temps = [150, 140, 130, 125, 120, 115, 110, 100, 90, 80]

        temperature_combinations = []
        for t in top_temps:
            for m in middle_temps:
                for b in bottom_temps:
                    if b<=m-10 and m<=t-10:
                        if t>=160 and b<125:
                            continue
                        elif t==150 and b<120:
                            continue
                        elif t==140 and b<115:
                            continue
                        elif t==130 and b<100:
                            continue
                        elif t==120 and b<90:
                            continue
                        temperature_combinations.append((t,m,b))
        additional_temperature_combinations = [
            (165, 155, 150), (165, 155, 145),
            (155, 145, 135), (155, 135, 120),
            (150, 145, 135),
            (145, 135, 120), (145, 130, 115),
            (140, 135, 125), (140, 130, 125),
            (135, 120, 115), (135, 125, 115),
            (90, 80, 70)
        ]
        for add_temp_combination in additional_temperature_combinations:
            if add_temp_combination not in temperature_combinations:
                temperature_combinations.append(add_temp_combination)
            else:
                logger.debug(f"Temperature combination {add_temp_combination} already exists")

        thermocline_combinations = []
        for t1 in range(1,self.params.num_layers+1):
            for t2 in range(1,self.params.num_layers+1):
                if t2>=t1:
                    thermocline_combinations.append((t1,t2))

        states: list[HeatPumpWaterTankState] = []

        for tmb in temperature_combinations:
            for th in thermocline_combinations:
                t, m, b = tmb
                th1, th2 = th
                if m==b and th1!=th2:
                    continue
                state = HeatPumpWaterTankState.build(
                    top_temp=t,
                    middle_temp=m,
                    bottom_temp=b,
                    thermocline1=th1,
                    thermocline2=th2,
                    params=self.params,
                    index=len(states),
                )
                states.append(state)

        max_temp = self.params.max_tank_temp_f
        self.max_state_energy = HeatPumpWaterTankState.build(
            max_temp, max_temp, max_temp, self.params.num_layers, self.params.num_layers, self.params
        ).energy
        self.min_state_energy = HeatPumpWaterTankState.build(
            COLDEST_TANK_TEMP_F,
            COLDEST_TANK_TEMP_F,
            COLDEST_TANK_TEMP_F,
            self.params.num_layers,
            self.params.num_layers,
            self.params,
        ).energy
        self.lowest_state_energy = min(state.energy for state in states)
        return states

    def get_action_space(self) -> list[HeatPumpWaterTankAction]:
        max_dt = max(self.params.timestep_duration_hours)
        max_load_kwh = self.params.max_load_kw_th * max_dt
        max_hp_kwh = self.params.hp_max_kw_th * max_dt
        self._heat_to_store_discretized = [
            x / 10
            for x in range(
                -int(max_load_kwh * 10),
                int((max_hp_kwh + 1) * 10) + 1,
            )
        ]
        self._heat_to_store_discretized_array = np.array(self._heat_to_store_discretized)
        self._action_by_heat_to_store_kwh = {
            heat: HeatPumpWaterTankAction(heat_to_store_kwh=heat, index=i)
            for i, heat in enumerate(self._heat_to_store_discretized)
        }
        return list(self._action_by_heat_to_store_kwh.values())

    def _ensure_states_by_energy(self) -> np.ndarray:
        energies = getattr(self, "_state_energies", None)
        if energies is not None and len(energies) == len(self.state_space):
            return energies
        energies = np.array([state.energy for state in self.state_space], dtype=np.float64)
        order = np.argsort(energies, kind="mergesort")
        self._state_energies = energies
        self._energy_order = order
        self._sorted_state_energies = energies[order]
        n_layers = self.params.num_layers
        layers = np.empty((len(self.state_space), n_layers), dtype=np.float64)
        for i, state in enumerate(self.state_space):
            layers[i] = self._temps_by_layer(state)
        self._layer_temp_matrix = layers
        return energies

    def _state_indices_within_energy(self, target: float, window: float) -> np.ndarray:
        """Indices of states whose energy is within ``window`` of ``target``, in state-space order."""
        self._ensure_states_by_energy()
        sorted_energy = self._sorted_state_energies
        lo = int(np.searchsorted(sorted_energy, target - window, side="left"))
        hi = int(np.searchsorted(sorted_energy, target + window, side="right"))
        lo = max(0, lo - 1)
        hi = min(len(sorted_energy), hi + 1)
        order = self._energy_order[lo:hi]
        keep = np.abs(self._state_energies[order] - target) <= window
        return np.sort(order[keep])

    def _losses_for_energies(self, energies: np.ndarray, dt: float) -> np.ndarray:
        return self.params.storage_losses_percent / 100 * (energies - self.min_state_energy) * dt

    def _expected_lwt_for_energies(self, energies: np.ndarray, heat_to_store: np.ndarray | float) -> np.ndarray:
        capacity_kwh_per_k = self.params.storage_volume_gallons * 3.785 * 4.187 / 3600
        end_energy = np.minimum(
            np.maximum(energies + heat_to_store, self.min_state_energy),
            self.max_state_energy,
        )
        start_f = (energies / capacity_kwh_per_k - 273.15) * (9 / 5) + 32
        end_f = (end_energy / capacity_kwh_per_k - 273.15) * (9 / 5) + 32
        return (start_f + end_f) / 2 + self.params.hp_constant_lift_f / 2

    def _cop_at(self, oat: float, lwt: np.ndarray) -> np.ndarray:
        return self.params.cop_intercept + self.params.cop_lwt_minus_oat_coeff * np.maximum(0, lwt - oat)

    def _thermal_for_energies(
        self,
        energies: np.ndarray,
        time_step: int,
        elec_in_kwh: float,
        losses: np.ndarray,
    ) -> np.ndarray:
        if elec_in_kwh <= 0:
            return np.zeros(energies.shape, dtype=np.float64)
        load_losses = self.params.load_kwh[time_step] + losses
        oat = self.params.oat_f[time_step]
        cop_at_start = self._cop_at(oat, self._expected_lwt_for_energies(energies, 0.0))
        capacity_kwh_per_k = self.params.storage_volume_gallons * 3.785 * 4.187 / 3600
        feedback = elec_in_kwh * self.params.cop_lwt_minus_oat_coeff * 0.5 * (1.8 / capacity_kwh_per_k)
        if feedback >= 1:
            heat_to_store = np.zeros(energies.shape, dtype=np.float64)
        else:
            heat_to_store = (elec_in_kwh * cop_at_start - load_losses) / (1 - feedback)
        heat_to_store = np.minimum(
            np.maximum(heat_to_store, self.min_state_energy - energies),
            self.max_state_energy - energies,
        )
        return elec_in_kwh * self._cop_at(oat, self._expected_lwt_for_energies(energies, heat_to_store))

    def _max_hp_for_energies(
        self,
        energies: np.ndarray,
        time_step: int,
        mode: int,
        losses: np.ndarray,
    ) -> np.ndarray:
        dt = self.params.timestep_duration_hours[time_step]
        turn_on_minutes = 0 if mode else self.params.hp_turn_on_minutes
        max_hp_elec_in = (1 - min(turn_on_minutes, dt * 60) / (dt * 60)) * self.params.hp_max_kw_elec * dt
        return self._thermal_for_energies(energies, time_step, max_hp_elec_in, losses)

    def _discrete_heat_to_store_index(self, desired: np.ndarray) -> np.ndarray:
        """Floor discharge below -0.25 kWh; otherwise take the nearest 0.1 kWh grid point."""
        grid = self._heat_to_store_discretized_array
        discharge = desired < -NET_HEAT_TO_STORE_MIN_DISCHARGE_KWH
        floored = np.maximum(np.searchsorted(grid, desired, side="right") - 1, 0)
        nearest = np.abs(grid - desired[:, None]).argmin(axis=1)
        return np.where(discharge, floored, nearest)

    def _cached_fill_to_full_action(self, state: HeatPumpWaterTankState) -> HeatPumpWaterTankAction | None:
        cache = self._fill_to_full_action_by_state
        if state.index in cache:
            return cache[state.index]
        desired = self.max_state_energy - state.energy
        fill_action: HeatPumpWaterTankAction | None = None
        if desired > 0:
            idx = int(np.abs(self._heat_to_store_discretized_array - desired).argmin())
            while idx >= 0:
                heat = self._heat_to_store_discretized[idx]
                if heat <= 0:
                    break
                candidate = self._action_by_heat_to_store_kwh[heat]
                successor = self.closest_state(
                    self.next_state(state, candidate), from_state=state, action=candidate
                )
                if successor.energy <= self.max_state_energy:
                    fill_action = candidate
                    break
                idx -= 1
        cache[state.index] = fill_action
        return fill_action

    def _build_available_actions_cache(self) -> None:
        """One vectorized pass per distinct hour and mode, instead of one Python call per state.

        Hours that share duration, load, outdoor temperature, and forced on/off reuse the same
        result. Fill-to-full still walks the tank model, once per state.
        """
        self._available_actions_cache = []
        self._fill_to_full_action_by_state = {}
        states = self.state_space
        energies = self._ensure_states_by_energy()
        actions = self.action_space
        state_indexes = [state.index for state in states]
        forced_on = set(self.params.hp_forced_on_steps)
        forced_off = set(self.params.hp_forced_off_steps)
        hours: dict[
            tuple[float, float, float, bool, bool],
            dict[tuple[int, int], tuple[tuple[HeatPumpWaterTankAction, int], ...]],
        ] = {}
        absent = -1

        for time_step in range(self.params.horizon):
            dt = self.params.timestep_duration_hours[time_step]
            load = self.params.load_kwh[time_step]
            hour_key = (dt, load, self.params.oat_f[time_step], time_step in forced_on, time_step in forced_off)
            cached_hour = hours.get(hour_key)
            if cached_hour is not None:
                self._available_actions_cache.append(cached_hour)
                continue

            losses = self._losses_for_energies(energies, dt)
            heat_to_full = self.max_state_energy - energies
            by_state_and_mode: dict[tuple[int, int], tuple[tuple[HeatPumpWaterTankAction, int], ...]] = {}
            is_forced_off = time_step in forced_off
            include_off = time_step not in forced_on
            off_idx = self._discrete_heat_to_store_index(0.0 - load - losses) if include_off else None

            for mode in (0, 1):
                max_hp = self._max_hp_for_energies(energies, time_step, mode, losses)
                if mode == 0 and self.params.hp_min_kw_elec > 0:
                    min_short = self._thermal_for_energies(
                        energies, time_step, self.params.hp_min_kw_elec * dt, losses
                    )
                else:
                    min_short = 0.0
                hp_for_full = heat_to_full + load + losses
                if is_forced_off:
                    include_on = np.zeros(len(states), dtype=bool)
                    fill_mask = include_on
                else:
                    include_on = hp_for_full >= max_hp
                    fill_mask = (
                        ~include_on
                        & (heat_to_full > 0)
                        & (hp_for_full > 0)
                        & (hp_for_full >= min_short)
                    )
                on_idx = (
                    self._discrete_heat_to_store_index(max_hp - load - losses)
                    if bool(include_on.any())
                    else None
                )

                fill_index_by_state: dict[int, int] = {}
                if bool(fill_mask.any()):
                    min_short_values = min_short if isinstance(min_short, np.ndarray) else None
                    for i in np.flatnonzero(fill_mask):
                        state = states[int(i)]
                        fill_action = self._cached_fill_to_full_action(state)
                        if fill_action is None:
                            continue
                        if min_short_values is not None and float(min_short_values[i]) > 0:
                            if self._hp_heat_out_kwh(state, fill_action, time_step) < float(min_short_values[i]):
                                continue
                        fill_index_by_state[int(i)] = fill_action.index

                off_codes = off_idx.tolist() if off_idx is not None else None
                on_codes = on_idx.tolist() if on_idx is not None else None
                include_on_list = include_on.tolist()
                intern: dict[tuple[int, int, int], tuple[tuple[HeatPumpWaterTankAction, int], ...]] = {}
                for i, state_index in enumerate(state_indexes):
                    oi = off_codes[i] if off_codes is not None else absent
                    ai = on_codes[i] if include_on_list[i] and on_codes is not None else absent
                    if ai == oi:
                        ai = absent
                    fi = fill_index_by_state.get(i, absent)
                    if fi == oi or fi == ai:
                        fi = absent
                    key = (oi, ai, fi)
                    result = intern.get(key)
                    if result is None:
                        chosen: list[tuple[HeatPumpWaterTankAction, int]] = []
                        if oi >= 0:
                            chosen.append((actions[oi], 0))
                        if ai >= 0:
                            chosen.append((actions[ai], 1))
                        if fi >= 0:
                            chosen.append((actions[fi], 0))
                        result = tuple(chosen)
                        intern[key] = result
                    by_state_and_mode[(state_index, mode)] = result

            hours[hour_key] = by_state_and_mode
            self._available_actions_cache.append(by_state_and_mode)

    def get_modes(self) -> list[int]:
        # 0: heat pump was off at the end of the previous step.
        # 1: heat pump ran at this step's full available power through the previous step.
        return [0, 1]

    def initial_mode(self) -> int:
        return 1 if self.params.hp_currently_on else 0

    def get_available_actions(
        self,
        state: HeatPumpWaterTankState,
        time_step: int,
        mode: int = 0,
    ) -> list[tuple[HeatPumpWaterTankAction, int]]:
        return list(self._available_actions_cache[time_step][(state.index, mode)])

    def allow_transition(
        self,
        state: HeatPumpWaterTankState,
        next_state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
        time_step: int,
    ) -> bool:
        if action.heat_to_store_kwh > 0 and next_state.energy > self.max_state_energy:
            return False
        return True

    def expected_lwt_f(self, state: HeatPumpWaterTankState, heat_to_store_kwh: float) -> float:
        """Leaving-water temperature (°F) averaged across the step.

        The outlet starts one lift above the starting bulk and finishes at the end bulk, where
        the heat pump stops. That average is the midpoint bulk temperature plus half the lift.
        """
        capacity_kwh_per_k = self.params.storage_volume_gallons * 3.785 * 4.187 / 3600
        end_energy = min(max(state.energy + heat_to_store_kwh, self.min_state_energy), self.max_state_energy)
        start_f = kelvin_to_fahrenheit(state.energy / capacity_kwh_per_k)
        end_f = kelvin_to_fahrenheit(end_energy / capacity_kwh_per_k)
        return (start_f + end_f) / 2 + self.params.hp_constant_lift_f / 2

    def elec_used_kwh(
        self,
        state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
        time_step: int,
    ) -> float:
        cop = self.params.COP(self.params.oat_f[time_step], self.expected_lwt_f(state, action.heat_to_store_kwh))
        return self._hp_heat_out_kwh(state, action, time_step) / cop

    def _hp_heat_out_kwh(
        self,
        state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
        time_step: int,
    ) -> float:
        load = self.params.load_kwh[time_step]
        losses = self._losses_kwh(state, time_step)
        return max(0.0, load + losses + action.heat_to_store_kwh)

    def _losses_kwh(self, state: HeatPumpWaterTankState, time_step: int) -> float:
        dt = self.params.timestep_duration_hours[time_step]
        return self.params.storage_losses_percent / 100 * (state.energy - self.min_state_energy) * dt

    def _thermal_kwh_for_elec_in(self, state: HeatPumpWaterTankState, time_step: int, elec_in_kwh: float) -> float:
        """Thermal kWh from ``elec_in_kwh`` at the COP of the tank's average temperature over the step.

        Stored heat warms the tank and lowers the COP, which lowers how much heat that electricity
        stores. The two are solved together; the tank's end energy stays inside its bounds.
        """
        if elec_in_kwh <= 0:
            return 0.0
        load_losses = self.params.load_kwh[time_step] + self._losses_kwh(state, time_step)
        cop_at_start = self.params.COP(self.params.oat_f[time_step], self.expected_lwt_f(state, 0.0))
        capacity_kwh_per_k = self.params.storage_volume_gallons * 3.785 * 4.187 / 3600
        # A stored kWh raises the average bulk by half of 1.8°F per kWh/K, and COP moves with that.
        feedback = elec_in_kwh * self.params.cop_lwt_minus_oat_coeff * 0.5 * (1.8 / capacity_kwh_per_k)
        if feedback >= 1:
            heat_to_store = 0.0
        else:
            heat_to_store = (elec_in_kwh * cop_at_start - load_losses) / (1 - feedback)
        heat_to_store = min(max(heat_to_store, self.min_state_energy - state.energy), self.max_state_energy - state.energy)
        lwt = self.expected_lwt_f(state, heat_to_store)
        return elec_in_kwh * self.params.COP(self.params.oat_f[time_step], lwt)

    def _max_hp_heat_out_kwh(self, state: HeatPumpWaterTankState, time_step: int, mode: int) -> float:
        dt = self.params.timestep_duration_hours[time_step]
        turn_on_minutes = 0 if mode else self.params.hp_turn_on_minutes
        max_hp_elec_in = (1 - min(turn_on_minutes, dt * 60) / (dt * 60)) * self.params.hp_max_kw_elec * dt
        return self._thermal_kwh_for_elec_in(state, time_step, max_hp_elec_in)

    def initial_state(self) -> HeatPumpWaterTankState:
        return HeatPumpWaterTankState.build(
            top_temp=self.params.initial_top_temp,
            middle_temp=self.params.initial_middle_temp,
            bottom_temp=self.params.initial_bottom_temp,
            thermocline1=self.params.initial_thermocline1,
            thermocline2=self.params.initial_thermocline2,
            params=self.params,
        )

    def get_model(self) -> HeatPumpWaterTankModel:
        from .model import HeatPumpWaterTankModel
        return HeatPumpWaterTankModel(self.params, self.state_space)

    def transition_matrix_variant_count(self) -> int:
        return len(self.params.load_bucket_upper_kwh) + 1

    def action_uses_transition_variants(self, action: HeatPumpWaterTankAction) -> bool:
        return action.heat_to_store_kwh < -NET_HEAT_TO_STORE_MIN_DISCHARGE_KWH

    def transition_variant(self, state: HeatPumpWaterTankState, action: HeatPumpWaterTankAction, time_step: int) -> int:
        if not self.action_uses_transition_variants(action):
            return 0
        load = max(0.0, self.params.load_kwh[time_step])
        for i, upper in enumerate(self.params.load_bucket_upper_kwh):
            if load < upper:
                return i
        return len(self.params.load_bucket_upper_kwh)

    def next_state(self, state: HeatPumpWaterTankState, action: HeatPumpWaterTankAction, transition_variant: int = 0,) -> HeatPumpWaterTankState:
        return self.model.next_state(state, action, transition_variant=transition_variant)

    def usable_energy_kwh(self, state: HeatPumpWaterTankState, rswt: float) -> float:
        total = 0
        for temp in self._temps_by_layer(state):
            if temp <= rswt:
                break
            total += (
                self.params.storage_volume_gallons * 3.785 / self.params.num_layers * 4.187 / 3600 * 5 / 9 
                * (temp - rswt)
            )
        return total

    def max_usable_energy_kwh(self, rswt: float) -> float:
        full = HeatPumpWaterTankState.build(
            self.params.max_tank_temp_f,
            self.params.max_tank_temp_f,
            self.params.max_tank_temp_f,
            self.params.num_layers,
            self.params.num_layers,
            self.params,
        )
        return self.usable_energy_kwh(full, rswt)

    def _terminal_end_window_stats(self) -> tuple[float, float]:
        window = self.params.terminal_cost_end_window_hours
        h = self.params.horizon
        start = max(0, h - window)
        prices_kwh = [p / 1000 for p in self.params.elec_price_mwh[start:h]]
        p_ref = self._median_kwh(prices_kwh)
        # The tank is held above the supply temperature, so replacement heat leaves at that temperature plus lift.
        cops = [
            self.params.COP(self.params.oat_f[t], self.params.rswt_f[t] + self.params.hp_constant_lift_f)
            for t in range(start, h)
        ]
        cop_bar = sum(cops) / len(cops)
        return p_ref, cop_bar

    def terminal_node_cost(self, state: HeatPumpWaterTankState | None = None) -> float:
        if not self.params.terminal_cost_enabled:
            return 0.0
        rswt = self.params.rswt_f[self.params.horizon - 1]
        p_ref, cop_bar = self._terminal_end_window_stats()
        e_max = self.max_usable_energy_kwh(rswt)
        if state is None:
            usable_gap = e_max
        else:
            usable_gap = max(0.0, e_max - self.usable_energy_kwh(state, rswt))
        return p_ref / cop_bar * usable_gap

    def terminal_cost_full(self) -> float:
        return self.terminal_node_cost(None)

    def _temps_by_layer(self, state: HeatPumpWaterTankState) -> list[float]:
        n = self.params.num_layers
        th1, th2 = state.thermocline1, state.thermocline2
        return (
            [state.top_temp] * th1
            + [state.middle_temp] * (th2 - th1)
            + [state.bottom_temp] * (n - th2)
        )

    def state_distance(self, state1: HeatPumpWaterTankState, state2: HeatPumpWaterTankState) -> float:
        temps_1 = self._temps_by_layer(state1)
        temps_2 = self._temps_by_layer(state2)
        return sum(abs(a-b) for a, b in zip(temps_1, temps_2))

    def closest_state(
        self,
        state: HeatPumpWaterTankState,
        from_state: HeatPumpWaterTankState | None = None,
        action: HeatPumpWaterTankAction | None = None,
    ) -> HeatPumpWaterTankState:
        energy_window_kwh = 0.5
        target = state.energy
        indices = self._state_indices_within_energy(target, energy_window_kwh)
        while indices.size == 0:
            if energy_window_kwh > 1 and 80 < state.top_temp < 170:
                logger.warning(f"No state within ±{energy_window_kwh} kWh of predicted state {state}")
            indices = self._state_indices_within_energy(target, energy_window_kwh)
            energy_window_kwh += 0.5

        if (
            from_state is not None
            and action is not None
            and action.heat_to_store_kwh < -NET_HEAT_TO_STORE_MIN_DISCHARGE_KWH
        ):
            colder = indices[self._state_energies[indices] < from_state.energy]
            while colder.size == 0 and indices.size < len(self.state_space):
                energy_window_kwh += 0.5
                indices = self._state_indices_within_energy(target, energy_window_kwh)
                colder = indices[self._state_energies[indices] < from_state.energy]
            if colder.size:
                indices = colder

        return self.state_space[self._closest_state_index(state, indices)]

    def _closest_state_index(self, state: HeatPumpWaterTankState, indices: np.ndarray) -> int:
        """Same winner as ``min`` over state distance, energy gap, then hotter top, then earlier index.
        A predicted state whose second thermocline is below the bottom layer is compared on the
        layers the tank has, as ``state_distance`` compares it."""
        if indices.size == 1:
            return int(indices[0])
        predicted = np.asarray(self._temps_by_layer(state)[: self.params.num_layers], dtype=np.float64)
        distance = np.abs(self._layer_temp_matrix[indices] - predicted).sum(axis=1)
        energy_gap = np.abs(self._state_energies[indices] - state.energy)
        top_temp = self._layer_temp_matrix[indices, 0]
        winner = int(np.lexsort((indices, -top_temp, energy_gap, distance))[0])
        return int(indices[winner])

    def successors(
        self,
        state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
        transition_variant: int = 0,
    ) -> list[tuple[HeatPumpWaterTankState, float]]:
        """The closest state and a state on the other side of the predicted energy, weighted so
        that their average energy is the predicted energy (see docs/two-node-transitions.md)."""
        predicted = self.next_state(state, action, transition_variant=transition_variant)
        closest = self.closest_state(predicted, from_state=state, action=action)
        target = state.energy + action.heat_to_store_kwh
        if abs(closest.energy - target) < 1e-6:
            return [(closest, 1.0)]
        other = self._other_side_state(predicted, target, closest.energy < target, state, action)
        if other is None:
            return [(closest, 1.0)]
        lower, upper = (closest, other) if closest.energy < target else (other, closest)
        upper_weight = (target - lower.energy) / (upper.energy - lower.energy)
        return [(lower, 1 - upper_weight), (upper, upper_weight)]

    def _other_side_state(
        self,
        predicted: HeatPumpWaterTankState,
        target: float,
        above: bool,
        from_state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
    ) -> HeatPumpWaterTankState | None:
        """The state above (or below) ``target`` energy that is closest to ``predicted``, searching
        in growing 0.5 kWh windows as ``closest_state`` does. Charging never goes above full, and
        discharging never ends with more energy than it started with."""
        if above:
            max_energy = target + OTHER_SIDE_MAX_WINDOW_KWH
            if action.heat_to_store_kwh > 0:
                max_energy = min(max_energy, self.max_state_energy)
            elif action.heat_to_store_kwh < 0:
                max_energy = min(max_energy, from_state.energy)
            candidates = [c for c in self.state_space if target < c.energy <= max_energy]
        else:
            min_energy = target - OTHER_SIDE_MAX_WINDOW_KWH
            candidates = [c for c in self.state_space if min_energy <= c.energy < target]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                math.ceil(abs(candidate.energy - target) / 0.5),
                self.state_distance(predicted, candidate),
                abs(candidate.energy - target),
                -candidate.top_temp,
            ),
        )

    def rollout_quantity(self, state: HeatPumpWaterTankState) -> float:
        return state.energy

    def cost(
        self,
        state: HeatPumpWaterTankState,
        next_state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
        time_step: int,
        mode: int = 0,
    ) -> float:

        elec_price_kwh = self.params.elec_price_mwh[time_step] / 1000

        hp_kwh_el = self.elec_used_kwh(state, action, time_step)
        cost = elec_price_kwh * hp_kwh_el

        # RSWT penalty
        cost += self.rswt_penalty(state, next_state, action, time_step)

        # Unmet load penalty
        cost += self.unmet_load_penalty(state, next_state, action, time_step)

        # Plan stability penalty
        cost += self.stability_penalty(time_step, hp_kwh_el)

        # Short-cycle penalty
        cost += self.short_cycle_penalty(state, action, time_step, mode)

        return cost

    def short_cycle_penalty(
        self,
        state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
        time_step: int,
        mode: int = 0,
    ) -> float:
        if not self.params.short_cycle_penalty_enabled:
            return 0.0
        if mode:
            return 0.0

        for cached_action, next_mode in self.get_available_actions(state, time_step, mode):
            if cached_action == action:
                if next_mode:
                    return 0.0
                break

        heat_from_hp = self._hp_heat_out_kwh(state, action, time_step)
        if heat_from_hp <= 0.15:
            return 0.0
        max_hp_heat_out = self._max_hp_heat_out_kwh(state, time_step, mode=0)
        if max_hp_heat_out <= 1e-9:
            return 0.0
        fraction = min(1.0, heat_from_hp / max_hp_heat_out)
        return (
            self._horizon_median_elec_price_kwh
            * self.params.short_cycle_penalty_weight
            * (1.0 - fraction)
        )

    def _compute_difference_with_plans(self):
        current_node_energy = HeatPumpWaterTankState.build(
            top_temp=self.params.initial_top_temp,
            middle_temp=self.params.initial_middle_temp,
            bottom_temp=self.params.initial_bottom_temp,
            thermocline1=self.params.initial_thermocline1,
            thermocline2=self.params.initial_thermocline2,
            params=self.params,
        ).energy
        if self.params.previous_estimate_storage_kwh_now is not None:
            self.storage_difference_with_plan_kwh = round(abs(current_node_energy - self.params.previous_estimate_storage_kwh_now), 2)
        if self.params.previous_estimate_elec_price_mwh_now is not None:
            self.first_hour_price_difference_mwh = round(
                abs(self.params.elec_price_mwh[0] - self.params.previous_estimate_elec_price_mwh_now),
                2,
            )

    def stability_penalty(self, time_step: int, hp_kwh_el: float) -> float:
        if not self.params.stability_penalty_enabled:
            return 0
        
        if (
            self.params.previous_plan_hp_kwh_el_list is None
            or self.params.previous_estimate_storage_kwh_now is None
            or self.params.previous_estimate_elec_price_mwh_now is None
        ):
            return 0

        previous_plan = self.params.previous_plan_hp_kwh_el_list
        if time_step >= len(previous_plan):
            return 0

        elapsed_hours = sum(self.params.timestep_duration_hours[:time_step])
        if elapsed_hours >= self.params.stability_penalty_horizon_hours:
            return 0

        if self.storage_difference_with_plan_kwh >= self.params.stability_penalty_threshold_kwh:
            return 0

        if self.first_hour_price_difference_mwh >= self.params.stability_penalty_threshold_price_mwh:
            return 0

        duration_of_this_time_step = self.params.timestep_duration_hours[time_step]
        duration_of_first_time_step = self.params.timestep_duration_hours[0]

        if time_step == 0:
            previous_plan_for_this_time_step = previous_plan[0]
        else:
            previous_plan_for_this_time_step = (
                (
                    (duration_of_this_time_step - duration_of_first_time_step) * previous_plan[time_step-1]
                    + duration_of_first_time_step * previous_plan[time_step]
                ) / duration_of_this_time_step
            )

        weight = self.params.stability_penalty_weight
        decay = self.params.stability_penalty_decay
        return (
            self._horizon_median_elec_price_kwh
            * weight
            * decay**elapsed_hours
            * abs(hp_kwh_el - previous_plan_for_this_time_step)
        )

    def rswt_penalty(
        self,
        state: HeatPumpWaterTankState,
        next_state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
        time_step: int,
    ) -> float:
        if not self.params.rswt_penalty_enabled:
            return 0
        rswt = self.params.rswt_f[time_step]
        if action.heat_to_store_kwh >= 0 or self.params.load_kwh[time_step] <= 0:
            return 0
        if state.top_temp >= rswt and next_state.top_temp >= rswt:
            return 0

        load_bucket = self.transition_variant(state, action, time_step)
        drawn_layers = self.layers_drawn(state, -action.heat_to_store_kwh, load_bucket, rswt=rswt)

        exponent_rate = self.params.rswt_penalty_exponent_rate
        weight = self.params.rswt_penalty_weight
        decay = self.params.rswt_penalty_decay
        max_hour = self.params.rswt_penalty_decay_max_hour
        elapsed_hours = sum(self.params.timestep_duration_hours[:time_step])
        layer_gallons = self.params.storage_volume_gallons / self.params.num_layers

        shortfall = sum(
            fraction * layer_gallons * max(0.0, math.exp(exponent_rate * (rswt - temp)) - 1)
            for temp, fraction in drawn_layers
        )
        return (
            self._horizon_median_elec_price_kwh
            * decay ** (max_hour - min(elapsed_hours, max_hour))
            * weight
            * shortfall
        )

    def unmet_load_penalty(
        self,
        state: HeatPumpWaterTankState,
        next_state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
        time_step: int,
    ) -> float:
        """Load the storage is credited with supplying but cannot, because the discharge would take it
        below the coldest state in the state space. Losses come out of this first, since a storage at
        the floor has no losses to give."""
        if not self.params.unmet_load_penalty_enabled or action.heat_to_store_kwh >= 0:
            return 0.0
        target = state.energy + action.heat_to_store_kwh
        if target >= self.lowest_state_energy:
            return 0.0
        unmet_load_kwh = max(0.0, next_state.energy - target - self._losses_kwh(state, time_step))
        return self.params.unmet_load_penalty_weight * self._max_heat_price_kwh * unmet_load_kwh

    def layers_drawn(
        self,
        state: HeatPumpWaterTankState,
        energy_kwh: float,
        load_bucket: int = 0,
        *,
        rswt: float,
    ) -> list[tuple[float, float]]:
        """Temperature of the water drawn from the top of the storage to supply energy_kwh, and how many
        layers the load would have drawn at rswt for the heat that water gives.

        Counting layers at rswt rather than layers drawn keeps lukewarm water, which gives little heat
        per layer and is drawn many times over, from weighing more than colder water.

        Each layer drawn is returned to the bottom at RWT, so once all original layers have been drawn,
        the returned water is drawn next, in the order it came back. Once the water drawn is too cold
        to give any heat, the rest of energy_kwh is left uncounted: that undeliverable heat is priced
        by the unmet-load penalty, not as water drawn below RSWT.
        """
        layer_kwh_per_f = self.params.storage_volume_gallons * 3.785 / self.params.num_layers * 4.187 / 3600 * 5 / 9
        rswt_layer_kwh = layer_kwh_per_f * (rswt - self.params.rwt(rswt, load_bucket))
        if rswt_layer_kwh <= 0:
            return []
        layers = self._temps_by_layer(state)
        drawn: list[tuple[float, float]] = []
        remaining_kwh = energy_kwh
        while remaining_kwh > 1e-9:
            returned: list[float] = []
            for temp in layers:
                rwt = self.params.rwt(temp, load_bucket)
                layer_kwh = layer_kwh_per_f * (temp - rwt)
                if layer_kwh <= 0:
                    return drawn
                supplied_kwh = min(remaining_kwh, layer_kwh)
                drawn.append((temp, supplied_kwh / rswt_layer_kwh))
                remaining_kwh -= supplied_kwh
                if remaining_kwh <= 1e-9:
                    return drawn
                returned.append(rwt)
            layers = returned
        return drawn
