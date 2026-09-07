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
        print(f"=> {len(temperature_combinations)} temperature combinations")

        thermocline_combinations = []
        for t1 in range(1,self.params.num_layers+1):
            for t2 in range(1,self.params.num_layers+1):
                if t2>=t1:
                    thermocline_combinations.append((t1,t2))
        print(f"=> {len(thermocline_combinations)} thermocline combinations")

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

        print(f"=> Created a total of {len(states)} states")
        self.max_state_energy = HeatPumpWaterTankState.build(
            180, 180, 180, self.params.num_layers, self.params.num_layers, self.params
        ).energy
        self.min_state_energy = HeatPumpWaterTankState.build(
            70, 70, 70, self.params.num_layers, self.params.num_layers, self.params
        ).energy
        return states

    def get_action_space(self) -> list[HeatPumpWaterTankAction]:
        actions = []
        for heat_to_store_kwh in range(-int(self.params.max_load_kwh_th), int(self.params.max_hp_kwh_th+1)+1):
            actions.append(HeatPumpWaterTankAction(heat_to_store_kwh=heat_to_store_kwh))
        return actions

    def get_available_actions(self, state: HeatPumpWaterTankState, time_step: int) -> list[HeatPumpWaterTankAction]:
        raise NotImplementedError

    def get_model(self) -> HeatPumpWaterTankModel:
        from .model import HeatPumpWaterTankModel
        return HeatPumpWaterTankModel(self.params, self.state_space)

    def next_state(self, state: HeatPumpWaterTankState, action: HeatPumpWaterTankAction) -> HeatPumpWaterTankState:
        return self.model.next_state(state, action)

    def _temp_at_layer(self, state: HeatPumpWaterTankState, layer: int) -> float:
        if layer < state.thermocline1:
            return state.top_temp
        if layer < state.thermocline2:
            return state.middle_temp
        return state.bottom_temp

    def state_distance(self, state1: HeatPumpWaterTankState, state2: HeatPumpWaterTankState) -> float:
        distance = 0
        for layer in range(self.params.num_layers):
            t_1 = self._temp_at_layer(state1, layer)
            t_2 = self._temp_at_layer(state2, layer)
            distance += abs(t_1 - t_2)
        return distance

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
        cop = self.params.COP(self.params.oat_f[time_step])

        # Electricity cost
        cost = elec_usd_kwh * action.heat_to_store_kwh/cop

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

        return cost

    def rswt_penalty(self, time_step: int, swt: float, rswt: float) -> float:
        if not self.params.rswt_penalty_enabled:
            return 0
        if swt>=rswt:
            return 0
        exponent_rate = self.params.rswt_penalty_exponent_rate
        weight = self.params.rswt_penalty_weight
        decay = self.params.rswt_penalty_decay
        max_hour = self.params.rswt_penalty_decay_max_hour
        penalty = decay**(max_hour - min(time_step,max_hour)) * weight * np.exp(exponent_rate*(rswt-swt))
        return penalty
