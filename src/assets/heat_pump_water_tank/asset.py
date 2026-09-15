from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..base import Asset
from .action import HeatPumpWaterTankAction
from .params import HeatPumpWaterTankParams
from .state import HeatPumpWaterTankState

if TYPE_CHECKING:
    from .model import HeatPumpWaterTankModel


class HeatPumpWaterTankAsset(Asset[HeatPumpWaterTankState, HeatPumpWaterTankAction, HeatPumpWaterTankParams]):

    def __init__(self, params: HeatPumpWaterTankParams):
        super().__init__(params)
        self._compute_storage_difference_with_plan_kwh()

    def on_params_updated(self) -> None:
        self._compute_storage_difference_with_plan_kwh()

    @property
    def name(self) -> str:
        return "heat_pump_water_tank"

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
                print(f"Temperature combination {add_temp_combination} already exists")

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
            heat: HeatPumpWaterTankAction(heat_to_store_kwh=heat)
            for heat in self._heat_to_store_discretized
        }
        return list(self._action_by_heat_to_store_kwh.values())

    def get_available_actions(self, state: HeatPumpWaterTankState, time_step: int) -> list[HeatPumpWaterTankAction]:
        dt = self.params.timestep_duration_hours[time_step]
        load = self.params.load_kwh[time_step]
        losses = self.params.storage_losses_percent/100 * (state.energy-self.min_state_energy) * dt
        cop = self.params.COP(self.params.oat_f[time_step])

        if time_step==0:
            turn_on_minutes = self.params.hp_turn_on_minutes if not self.params.hp_currently_on else 0
        else:
            turn_on_minutes = self.params.hp_turn_on_minutes/2

        max_hp_elec_in = (1-min(turn_on_minutes, dt*60)/(dt*60)) * self.params.hp_max_kw_elec * dt
        max_hp_heat_out = max_hp_elec_in * cop

        hp_heat_out_levels = [0]

        # Can not put out more heat than what would fill the storage
        heat_to_store_for_full = self.max_state_energy - state.energy
        hp_heat_out_for_full = heat_to_store_for_full + load + losses

        min_charge_kwh = (self.params.hp_min_kw_th_first_step if time_step==0 else self.params.hp_min_kw_th_other_steps) * dt

        if hp_heat_out_for_full >= max_hp_heat_out:
            hp_heat_out_levels += [max_hp_heat_out]
        elif hp_heat_out_for_full > min_charge_kwh:
            hp_heat_out_levels += [hp_heat_out_for_full]

        # If the HP is already on, add the "meet the load" edge in the first step
        if time_step==0 and load>0 and self.params.hp_currently_on:
            hp_heat_out_levels += [load+losses]

        heat_to_store_options = [hp_heat_out-load-losses for hp_heat_out in hp_heat_out_levels]
        actions: list[HeatPumpWaterTankAction] = []
        seen: set[HeatPumpWaterTankAction] = set()
        for heat_to_store_desired in heat_to_store_options:
            idx = int(np.abs(self._heat_to_store_discretized_array - heat_to_store_desired).argmin())
            heat = float(self._heat_to_store_discretized_array[idx])
            action = self._action_by_heat_to_store_kwh[heat]
            if action not in seen:
                seen.add(action)
                actions.append(action)
        
        return actions

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
        dt = self.params.timestep_duration_hours[time_step]
        load = self.params.load_kwh[time_step]
        losses = self.params.storage_losses_percent / 100 * (state.energy - self.min_state_energy) * dt
        cop = self.params.COP(self.params.oat_f[time_step])
        heat_from_hp = max(0.0, load + losses + action.heat_to_store_kwh)
        return heat_from_hp / cop

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

    def next_state(self, state: HeatPumpWaterTankState, action: HeatPumpWaterTankAction) -> HeatPumpWaterTankState:
        return self.model.next_state(state, action)

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
                print(f"No state within ±{energy_window_kwh} kWh of predicted state {state}")
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
    ) -> float:

        elec_usd_kwh = self.params.elec_usd_mwh[time_step]/1000
        rswt = self.params.rswt_f[time_step]
        load = self.params.load_kwh[time_step]

        hp_kwh_el = self.elec_used_kwh(state, action, time_step)
        cost = elec_usd_kwh * hp_kwh_el

        # RSWT penalty
        if action.heat_to_store_kwh<0 and load>0 and (state.top_temp<rswt or next_state.top_temp<rswt):
            if state.top_temp == next_state.top_temp:
                swt = state.top_temp
            else:
                if state.thermocline2 > state.thermocline1:
                    temp_below_top_now = state.middle_temp
                    num_layers_available = state.thermocline2 - state.thermocline1
                else:
                    temp_below_top_now = state.bottom_temp
                    num_layers_available = self.params.num_layers - state.thermocline2

                if next_state.top_temp == temp_below_top_now:
                    num_layers_used = max(0, num_layers_available - next_state.thermocline1)
                    swt = (state.top_temp*state.thermocline1 + temp_below_top_now*num_layers_used)/(state.thermocline1 + num_layers_used)
                elif next_state.top_temp < temp_below_top_now:
                    swt = (state.top_temp + temp_below_top_now + next_state.top_temp)/3
                else:
                    swt = next_state.top_temp

            cost += self.rswt_penalty(time_step, swt, rswt)

        # Plan stability penalty
        cost += self.stability_penalty(time_step, hp_kwh_el)

        return cost

    def _compute_storage_difference_with_plan_kwh(self):
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

    def stability_penalty(self, time_step: int, hp_kwh_el: float) -> float:
        if not self.params.stability_penalty_enabled:
            return 0
        
        if self.params.previous_plan_hp_kwh_el_list is None or self.params.previous_estimate_storage_kwh_now is None:
            return 0

        previous_plan = self.params.previous_plan_hp_kwh_el_list
        if time_step >= len(previous_plan):
            return 0

        elapsed_hours = sum(self.params.timestep_duration_hours[:time_step])
        if elapsed_hours >= self.params.stability_penalty_horizon_hours:
            return 0

        if self.storage_difference_with_plan_kwh >= self.params.stability_penalty_threshold_kwh:
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
        return weight * decay**elapsed_hours * abs(hp_kwh_el - previous_plan_for_this_time_step)

    def rswt_penalty(self, time_step: int, swt: float, rswt: float) -> float:
        if not self.params.rswt_penalty_enabled:
            return 0
        if swt>=rswt:
            return 0
        exponent_rate = self.params.rswt_penalty_exponent_rate
        weight = self.params.rswt_penalty_weight
        decay = self.params.rswt_penalty_decay
        max_hour = self.params.rswt_penalty_decay_max_hour
        elapsed_hours = sum(self.params.timestep_duration_hours[:time_step])
        penalty = decay**(max_hour - min(elapsed_hours, max_hour)) * weight * np.exp(exponent_rate*(rswt-swt))
        return penalty
