from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from bruce.graph_optimizer.settings import get_logger

from ..base import Asset, TransitionMatrixParams

logger = get_logger("assets.heat_pump_water_tank")
from .action import HeatPumpWaterTankAction
from .params import HeatPumpWaterTankParams
from .state import HeatPumpWaterTankState
from .transition_matrix_params import HeatPumpWaterTankTransitionMatrixParams

if TYPE_CHECKING:
    from .model import HeatPumpWaterTankModel


class HeatPumpWaterTankAsset(Asset[HeatPumpWaterTankState, HeatPumpWaterTankAction, HeatPumpWaterTankParams]):

    def __init__(self, params: HeatPumpWaterTankParams):
        super().__init__(params)
        self._refresh_horizon_median_elec_price_kwh()
        self._compute_difference_with_plans()
        self._available_actions_cache: list[
            dict[tuple[int, int], tuple[tuple[HeatPumpWaterTankAction, int], ...]]
        ] = []
        self._fill_to_full_action_by_state: dict[int, HeatPumpWaterTankAction | None] = {}
        self._build_available_actions_cache()

    def on_params_updated(self) -> None:
        self._refresh_horizon_median_elec_price_kwh()
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
            70, 70, 70, self.params.num_layers, self.params.num_layers, self.params
        ).energy
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

    def _build_available_actions_cache(self) -> None:
        self._available_actions_cache = []
        self._fill_to_full_action_by_state = {}
        for time_step in range(self.params.horizon):
            by_state_and_mode: dict[tuple[int, int], tuple[tuple[HeatPumpWaterTankAction, int], ...]] = {}
            for state in self.state_space:
                for mode in (0, 1):
                    by_state_and_mode[(state.index, mode)] = self._available_actions_for(state, time_step, mode)
            self._available_actions_cache.append(by_state_and_mode)

    def _available_actions_for(
        self,
        state: HeatPumpWaterTankState,
        time_step: int,
        mode: int,
    ) -> tuple[tuple[HeatPumpWaterTankAction, int], ...]:
        energy = state.energy
        dt = self.params.timestep_duration_hours[time_step]
        load = self.params.load_kwh[time_step]
        losses = self.params.storage_losses_percent / 100 * (energy - self.min_state_energy) * dt
        max_hp_heat_out = self._max_hp_heat_out_kwh(time_step, mode)
        min_short_cycle_heat = (
            self.params.hp_min_kw_elec * dt * self.params.COP(self.params.oat_f[time_step])
            if mode == 0 and self.params.hp_min_kw_elec > 0
            else 0.0
        )

        heat_to_store_for_full = self.max_state_energy - energy
        hp_heat_out_for_full = heat_to_store_for_full + load + losses

        # (heat out, next mode: 1 iff this step runs at full available power through its end)
        hp_heat_out_levels: list[tuple[float, int]] = [(0.0, 0)]
        fill_to_full = False
        if hp_heat_out_for_full >= max_hp_heat_out:
            hp_heat_out_levels.append((max_hp_heat_out, 1))
        elif (
            heat_to_store_for_full > 0
            and hp_heat_out_for_full > 0
            and hp_heat_out_for_full >= min_short_cycle_heat
        ):
            fill_to_full = True

        actions: list[tuple[HeatPumpWaterTankAction, int]] = []
        seen: set[HeatPumpWaterTankAction] = set()
        for hp_heat_out, next_mode in hp_heat_out_levels:
            heat_to_store_desired = hp_heat_out - load - losses
            idx = int(np.abs(self._heat_to_store_discretized_array - heat_to_store_desired).argmin())
            action = self._action_by_heat_to_store_kwh[self._heat_to_store_discretized[idx]]
            if action not in seen:
                seen.add(action)
                actions.append((action, next_mode))

        if fill_to_full:
            if state.index not in self._fill_to_full_action_by_state:
                desired = self.max_state_energy - state.energy
                fill_action: HeatPumpWaterTankAction | None = None
                if desired > 0:
                    idx = int(np.abs(self._heat_to_store_discretized_array - desired).argmin())
                    while idx >= 0:
                        heat = self._heat_to_store_discretized[idx]
                        if heat <= 0:
                            break
                        candidate = self._action_by_heat_to_store_kwh[heat]
                        successor = self.closest_state(self.next_state(state, candidate))
                        if successor.energy <= self.max_state_energy:
                            fill_action = candidate
                            break
                        idx -= 1
                self._fill_to_full_action_by_state[state.index] = fill_action
            fill_action = self._fill_to_full_action_by_state[state.index]
            if fill_action is not None and fill_action not in seen:
                if (
                    min_short_cycle_heat > 0
                    and self._hp_heat_out_kwh(state, fill_action, time_step) < min_short_cycle_heat
                ):
                    fill_action = None
            if fill_action is not None and fill_action not in seen:
                seen.add(fill_action)
                actions.append((fill_action, 0))
        return tuple(actions)

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

    def elec_used_kwh(self, state: HeatPumpWaterTankState, action: HeatPumpWaterTankAction, time_step: int) -> float:
        cop = self.params.COP(self.params.oat_f[time_step])
        return self._hp_heat_out_kwh(state, action, time_step) / cop

    def _hp_heat_out_kwh(
        self,
        state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
        time_step: int,
    ) -> float:
        dt = self.params.timestep_duration_hours[time_step]
        load = self.params.load_kwh[time_step]
        losses = self.params.storage_losses_percent / 100 * (state.energy - self.min_state_energy) * dt
        return max(0.0, load + losses + action.heat_to_store_kwh)

    def _max_hp_heat_out_kwh(self, time_step: int, mode: int) -> float:
        dt = self.params.timestep_duration_hours[time_step]
        cop = self.params.COP(self.params.oat_f[time_step])
        turn_on_minutes = 0 if mode else self.params.hp_turn_on_minutes
        max_hp_elec_in = (1 - min(turn_on_minutes, dt * 60) / (dt * 60)) * self.params.hp_max_kw_elec * dt
        return max_hp_elec_in * cop

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
        return action.heat_to_store_kwh < -0.5

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
        cops = [self.params.COP(self.params.oat_f[t]) for t in range(start, h)]
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

    def closest_state(self, state: HeatPumpWaterTankState) -> HeatPumpWaterTankState:
        energy_window_kwh = 0.5
        shortlist = [
            candidate
            for candidate in self.state_space
            if abs(candidate.energy - state.energy) <= energy_window_kwh
        ]
        while not shortlist:
            if energy_window_kwh > 1 and 80 < state.top_temp < 170:
                logger.warning(f"No state within ±{energy_window_kwh} kWh of predicted state {state}")
            shortlist = [
                candidate
                for candidate in self.state_space
                if abs(candidate.energy - state.energy) <= energy_window_kwh
            ]
            energy_window_kwh += 0.5

        return min(
            shortlist,
            key=lambda candidate: (
                self.state_distance(state, candidate),
                abs(candidate.energy - state.energy),
                -candidate.top_temp,
            ),
        )

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
        max_hp_heat_out = self._max_hp_heat_out_kwh(time_step, mode=0)
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
        drawn_layers = self.layers_drawn(state, -action.heat_to_store_kwh, load_bucket)

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

    def layers_drawn(
        self,
        state: HeatPumpWaterTankState,
        energy_kwh: float,
        load_bucket: int = 0,
    ) -> list[tuple[float, float]]:
        """Temperature and (fractional) number of layers drawn from the top of the storage to supply energy_kwh.

        Each layer drawn is returned to the bottom at RWT, so once all original layers have been drawn,
        the returned water is drawn next, in the order it came back.
        """
        layer_kwh_per_f = self.params.storage_volume_gallons * 3.785 / self.params.num_layers * 4.187 / 3600 * 5 / 9
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
                fraction = min(1.0, remaining_kwh / layer_kwh)
                drawn.append((temp, fraction))
                remaining_kwh -= fraction * layer_kwh
                if remaining_kwh <= 1e-9:
                    return drawn
                returned.append(rwt)
            layers = returned
        return drawn
