from logging import Logger, getLogger

from ..base import Model
from .heat_pump_water_tank import HeatPumpWaterTankAction, HeatPumpWaterTankParams, HeatPumpWaterTankState


class HeatPumpWaterTankModel(Model[HeatPumpWaterTankState, HeatPumpWaterTankAction, HeatPumpWaterTankParams]):
    def __init__(
            self,
            params: HeatPumpWaterTankParams,
            states: list[HeatPumpWaterTankState],
            logger: Logger = getLogger(__name__)
    ):
        super().__init__(params)
        self.states: list[HeatPumpWaterTankState] = states
        self.logger = logger
        self.discharge_return_temp = {
            155: 125,
            140: 116,
            130: 110,
            120: 100,
            110: 90,
            100: 80,
            90: 70,
            0: 60
        }

    def get_discharge_return_temp(self, t: float) -> float:
        sorted_items = sorted(self.discharge_return_temp.items())
        t_keys = [k for k, _ in sorted_items]
        if t <= t_keys[0]:
            return self.discharge_return_temp[t_keys[0]]
        if t >= t_keys[-1]:
            return self.discharge_return_temp[t_keys[-1]]
        for i in range(len(t_keys) - 1):
            t_low, t_high = t_keys[i], t_keys[i + 1]
            if t_low <= t <= t_high:
                b_low = self.discharge_return_temp[t_low]
                b_high = self.discharge_return_temp[t_high]
                frac = (t - t_low) / (t_high - t_low) if t_high != t_low else 0.0
                return b_low + (b_high - b_low) * frac
        return self.discharge_return_temp[t_keys[-1]]

    def next_state(self, state: HeatPumpWaterTankState, action: HeatPumpWaterTankAction) -> HeatPumpWaterTankState:
        store_heat_in = action.heat_to_store_kwh
        if store_heat_in > 0:
            self.logger.debug(f"Charge {state} by {store_heat_in} kWh")
            next_state = self.charge(state, store_heat_in)
        elif store_heat_in < -0.5:
            self.logger.debug(f"Discharge {state} by {-store_heat_in} kWh")
            next_state = self.discharge(state, store_heat_in)
        else:
            self.logger.debug(f"Stay at {state}")
            next_state = state
        return next_state

    def charge(self, n: HeatPumpWaterTankState, store_heat_in: float) -> HeatPumpWaterTankState:
        next_state_energy = n.energy + store_heat_in
        self.logger.debug(f"Charging {n} by {store_heat_in}")
        self.logger.debug(f"Current energy {round(n.energy,2)}, looking for {round(next_state_energy,2)}")

        t = n.top_temp
        m = n.middle_temp
        b = n.bottom_temp
        th1 = n.thermocline1
        th2 = n.thermocline2

        if th2!=self.params.num_layers:
            return self.charge_tmb(t,m,b,th1,th2,next_state_energy)
        elif th1!=self.params.num_layers:
            return self.charge_tm(t,m,th1,next_state_energy)
        else:
            return self.charge_t(t,next_state_energy)

    def charge_tmb(self, t, m, b, th1, th2, next_state_energy) -> HeatPumpWaterTankState:
        if th2==self.params.num_layers:
            return self.charge_tm(t, m, th1, next_state_energy)
        if th1==th2:
            return self.charge_tm(t, b, th2, next_state_energy)

        b_heated = b + self.params.delta_T(b)

        if b_heated == t:
            '''
            Heated water from the bottom is at the same temperature as the top layer.
            We can just move the thermoclines down until either:
                - We heated the storage by more than store_heat_in kWh
                - We ran out of bottom layers, giving us a (t,m) case
            '''
            candidate_states = []
            found_candidates = False
            
            while th2 < self.params.num_layers:
                th1 += 1
                th2 += 1
                state = HeatPumpWaterTankState(t,m,b,th1,th2,self.params)
                candidate_states.append(state)
                self.logger.debug(f"Adding {state} to candidates {state.energy}")
                if next_state_energy < state.energy:
                    found_candidates = True
                    break
            
            if found_candidates:
                # We will not heat up all bottom layers
                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))
            else:
                # We will charge past the bottom layers
                return self.charge_tm(t,m,th1,next_state_energy)                

        elif b_heated < t:
            '''
            Heated water from the bottom is still colder than the top layer.
            Two main scenarios exist:
                1. All of the bottom will be heated, then we will have a (t,m) case
                2. Not all of the bottom will be heated, we should determine how much of it will be

            In both cases, if m > b_heated, only the top and heated bottom will mix.
            Otherwise the middle will also mix.
            '''

            heated_bottom_energy = HeatPumpWaterTankState(t,m,b_heated,th1,th2,self.params).energy
            b_layers = self.params.num_layers - th2

            # We will charge past the bottom layers
            if heated_bottom_energy < next_state_energy:
                if m < b_heated:
                    mixed_layers = th1 + b_layers
                    mixed_temp = (t*th1 + b_heated*b_layers)/mixed_layers
                    return self.charge_tm(mixed_temp, m, mixed_layers, next_state_energy)
                else:
                    mixed_temp = (t*th1 + m*(th2-th1) + b_heated*b_layers) / self.params.num_layers
                    return self.charge_t(mixed_temp, next_state_energy)
                
            # We will not heat up all bottom layers
            else:
                current_energy = HeatPumpWaterTankState(t,m,b,th1,th2,self.params).energy
                energy_heating_one_layer = (heated_bottom_energy-current_energy) / b_layers
                layers_to_heat = (next_state_energy-current_energy) / energy_heating_one_layer
                layers_to_heat = min(layers_to_heat, b_layers-1)

                candidate_states = []
                for b_layers_to_heat in [int(layers_to_heat), int(layers_to_heat)+1]:
                    if m < b_heated:
                        mixed_layers = th1 + b_layers_to_heat
                        mixed_temp = (t*th1 + b_heated*b_layers_to_heat) / mixed_layers
                        state = HeatPumpWaterTankState(mixed_temp, m, b, mixed_layers, th2+b_layers_to_heat, self.params)
                    else:
                        mixed_layers = th2 + b_layers_to_heat
                        mixed_temp = (t*th1 + m*(th2-th1) + b_heated*b_layers_to_heat) / mixed_layers
                        state = HeatPumpWaterTankState(mixed_temp, mixed_temp, b, mixed_layers, mixed_layers, self.params)
                    candidate_states.append(state)

                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))

        elif b_heated > t:
            '''
            Heated water from the bottom is hotter than the top layer.
            Two main scenarios exist:
                1. All of the bottom will be heated, then we will have a new (t,m,b) case
                2. Not all of the bottom will be heated, we should determine how much of it will be
            '''
            
            heated_bottom_energy = HeatPumpWaterTankState(t,m,b_heated,th1,th2,self.params).energy
            b_layers = self.params.num_layers - th2

            # We will charge past the bottom layers
            if heated_bottom_energy < next_state_energy:
                return self.charge_tmb(b_heated, t, m, b_layers, th1+b_layers, next_state_energy)
                
            # We will not heat up all bottom layers
            else:
                current_energy = HeatPumpWaterTankState(t,m,b,th1,th2,self.params).energy
                energy_heating_one_layer = (heated_bottom_energy-current_energy) / b_layers
                layers_to_heat = (next_state_energy-current_energy) / energy_heating_one_layer
                layers_to_heat = min(layers_to_heat, b_layers-1)

                candidate_states = []
                for b_layers_to_heat in [int(layers_to_heat), int(layers_to_heat)+1]:
                    # Node is (b_heated, t, m, b) => need to combine two adjacent layers
                    # Combine b_heated and t
                    candidate_states.append(
                        HeatPumpWaterTankState(
                            top_temp = (b_heated*b_layers_to_heat + t*th1) / (b_layers_to_heat + th1), 
                            middle_temp = m, 
                            bottom_temp = b, 
                            thermocline1 = b_layers_to_heat + th1, 
                            thermocline2 = b_layers_to_heat + th2,
                            params = self.params, 
                        )
                    )
                    # Combine t and m
                    candidate_states.append(
                        HeatPumpWaterTankState(
                            top_temp = b_heated, 
                            middle_temp = (t*th1 + m*(th2-th1))/th2, 
                            bottom_temp = b, 
                            thermocline1 = b_layers_to_heat, 
                            thermocline2 = b_layers_to_heat + th2,
                            params = self.params, 
                        )
                    )
                    # Combine m and b
                    candidate_states.append(
                        HeatPumpWaterTankState(
                            top_temp = b_heated, 
                            middle_temp = t, 
                            bottom_temp = (m*(th2-th1)+b*(b_layers-b_layers_to_heat))/(th2-th1+b_layers-b_layers_to_heat), 
                            thermocline1 = b_layers_to_heat, 
                            thermocline2 = b_layers_to_heat + th1,
                            params = self.params, 
                        )
                    )

                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))

    def charge_tm(self, t, m, th1, next_state_energy) -> HeatPumpWaterTankState:
        if th1==self.params.num_layers:
            return self.charge_t(t, next_state_energy)

        m_heated = m + self.params.delta_T(m)

        if m_heated == t:
            '''
            Heated water from the middle is at the same temperature as the top layer.
            We can just move the first thermocline down until either:
                - We heated the storage by more than store_heat_in kWh
                - We ran out of middle layers, giving us a (t) case
            '''
            candidate_states = []
            
            while th1 < self.params.num_layers:
                th1 += 1
                state = HeatPumpWaterTankState(t,m,m,th1,self.params.num_layers,self.params)
                candidate_states.append(state)
                self.logger.debug(f"Adding {state} to candidates {state.energy}")
                # We will not heat up more middle layers
                if next_state_energy < state.energy:
                    return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))
            
            # We will charge past the middle layers
            return self.charge_t(t,next_state_energy) 

        elif m_heated < t:
            '''
            Heated water from the middle is still colder than the top layer.
            Two main scenarios exist:
                1. All of the middle will be heated, then we will have a (t) case
                2. Not all of the middle will be heated, we should determine how much of it will be
            '''

            heated_middle_energy = HeatPumpWaterTankState(t,m_heated,m_heated,th1,self.params.num_layers,self.params).energy
            m_layers = self.params.num_layers - th1

            # We will charge past the middle layers
            if heated_middle_energy < next_state_energy:
                mixed_layers = self.params.num_layers
                mixed_temp = (t*th1 + m_heated*m_layers)/mixed_layers
                return self.charge_t(mixed_temp, next_state_energy)
                
            # We will not heat up all middle layers
            else:
                current_energy = HeatPumpWaterTankState(t,m,m,th1,self.params.num_layers,self.params).energy
                energy_heating_one_layer = (heated_middle_energy-current_energy) / m_layers
                layers_to_heat = (next_state_energy-current_energy) / energy_heating_one_layer
                layers_to_heat = min(layers_to_heat, m_layers-1)

                candidate_states = []
                for m_layers_to_heat in [int(layers_to_heat), int(layers_to_heat)+1]:
                    mixed_layers = th1 + m_layers_to_heat
                    mixed_temp = (t*th1 + m_heated*m_layers_to_heat) / mixed_layers
                    state = HeatPumpWaterTankState(mixed_temp, m, m, mixed_layers, self.params.num_layers, self.params)
                    candidate_states.append(state)

                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))  

        elif m_heated > t:
            '''
            Heated water from the middle is hotter than the top layer.
            Two main scenarios exist:
                1. All of the middle will be heated, then we will have a new (t,m) case
                2. Not all of the middle will be heated, we should determine how much of it will be
            '''
            
            heated_middle_energy = HeatPumpWaterTankState(t,m_heated,m_heated,th1,self.params.num_layers,self.params).energy
            m_layers = self.params.num_layers - th1

            # We will charge past the middle layers
            if heated_middle_energy < next_state_energy:
                return self.charge_tm(m_heated, t, m_layers, next_state_energy)
                
            # We will not heat up all middle layers
            else:
                current_energy = HeatPumpWaterTankState(t,m,m,th1,self.params.num_layers,self.params).energy
                energy_heating_one_layer = (heated_middle_energy-current_energy) / m_layers
                layers_to_heat = (next_state_energy-current_energy) / energy_heating_one_layer
                layers_to_heat = min(layers_to_heat, m_layers-1)

                candidate_states = []
                for m_layers_to_heat in [int(layers_to_heat), int(layers_to_heat)+1]:
                    # Node is (m_heated, t, m) => new (t,m,b) case
                    if m_layers_to_heat == 0:
                        state = HeatPumpWaterTankState(t, m, m, th1, th1, self.params)
                    else:
                        state = self.charge_tmb(m_heated, t, m, m_layers_to_heat, m_layers_to_heat+th1, next_state_energy)         
                    candidate_states.append(state)

                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))

    def charge_t(self, t, next_state_energy) -> HeatPumpWaterTankState:
        t_heated = t + self.params.delta_T(t)
        
        '''
        Heated water can only be hotter than the top layer.
        Two main scenarios exist:
            1. All of the top will be heated, then we will have a new (t) case
            2. Not all of the top will be heated, we should determine how much of it will be
        '''

        heated_top_energy = HeatPumpWaterTankState(t_heated,t_heated,t_heated,self.params.num_layers,self.params.num_layers, self.params).energy
        t_layers = self.params.num_layers

        # We will charge past the top layers
        if heated_top_energy < next_state_energy:
            return self.charge_t(t_heated, next_state_energy)
            
        # We will not heat up all top layers
        else:
            current_energy = HeatPumpWaterTankState(t,t,t,self.params.num_layers,self.params.num_layers, self.params).energy
            energy_heating_one_layer = (heated_top_energy-current_energy) / t_layers
            layers_to_heat = (next_state_energy-current_energy) / energy_heating_one_layer
            layers_to_heat = min(layers_to_heat, t_layers-1)

            candidate_states = []
            for t_layers_to_heat in [int(layers_to_heat), int(layers_to_heat)+1]:
                # Node is (t_heated, t) => new (t,m) case
                if t_layers_to_heat == 0:
                    state = HeatPumpWaterTankState(t, t, t, self.params.num_layers, self.params.num_layers, self.params)
                else:
                    state = self.charge_tm(t_heated, t, t_layers_to_heat, next_state_energy)  
                candidate_states.append(state)

            return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))

    def discharge(self, n: HeatPumpWaterTankState, store_heat_in: float) -> HeatPumpWaterTankState:
        next_state_energy = n.energy + store_heat_in
        self.logger.debug(f"Discharging {n} by {store_heat_in}")
        self.logger.debug(f"Current energy {round(n.energy,2)}, looking for {round(next_state_energy,2)}")

        t = n.top_temp
        m = n.middle_temp
        b = n.bottom_temp
        th1 = n.thermocline1
        th2 = n.thermocline2

        if th2!=self.params.num_layers:
            return self.discharge_tmb(t, m, b, th1, th2, next_state_energy)
        elif th1!=self.params.num_layers:
            return self.discharge_tm(t, m, th1, next_state_energy)
        else:
            return self.discharge_t(t, next_state_energy)

    def discharge_tmb(self, t, m, b, th1, th2, next_state_energy) -> HeatPumpWaterTankState:
        if th2==self.params.num_layers:
            return self.discharge_tm(t, m, th1, next_state_energy)
        if th1==th2:
            return self.discharge_tm(t, b, th2, next_state_energy)

        t_cooled = self.get_discharge_return_temp(t)

        if t_cooled == b:
            '''
            Cooled water from the top is at the same temperature as the bottom layer.
            We can just move the thermoclines down until either:
                - We cooled the storage by more than store_heat_in kWh
                - We ran out of top layers, giving us a (t,m) case
            '''
            candidate_states = []
            found_candidates = False
            
            while th1 > 1:
                th1 += -1
                th2 += -1
                state = HeatPumpWaterTankState(t,m,b,th1,th2,self.params)
                candidate_states.append(state)
                self.logger.debug(f"Adding {state} to candidates {state.energy}")
                if next_state_energy > state.energy:
                    found_candidates = True
                    break
            
            if found_candidates:
                # We will not cool down all top layers
                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))
            else:
                # We will cool past the top layers
                return self.discharge_tm(m,b,th2,next_state_energy)                

        elif t_cooled < b:
            '''
            Cooled water from the top is colder than the bottom layer.
            Two main scenarios exist:
                1. All of the top will be cooled, then we will have a new (t,m,b) case
                2. Not all of the top will be cooled, we should determine how much of it will be
            '''

            cooled_top_energy = HeatPumpWaterTankState(t_cooled, m, b, th1, th2, self.params).energy
            t_layers = th1

            # We will discharge past the top layers
            if cooled_top_energy > next_state_energy:
                return self.discharge_tmb(m, b, t_cooled, th2-th1, self.params.num_layers-th1, next_state_energy)

            # We will not discharge all top layers
            else:
                current_energy = HeatPumpWaterTankState(t,m,b,th1,th2,self.params).energy
                energy_cooling_one_layer = (cooled_top_energy - current_energy) / t_layers
                layers_to_cool = (next_state_energy - current_energy) / energy_cooling_one_layer
                layers_to_cool = min(layers_to_cool, t_layers-1)

                candidate_states = []
                for t_layers_to_cool in [int(layers_to_cool), int(layers_to_cool)+1]:
                    # Node is (t, m, b, t_cooled) => need to combine two adjacent layers
                    # Combine t and m
                    candidate_states.append(
                        HeatPumpWaterTankState(
                            top_temp = (t*(th1-t_layers_to_cool) + m*(th2-th1)) / (th2 - t_layers_to_cool), 
                            middle_temp = b, 
                            bottom_temp = t_cooled, 
                            thermocline1 = th2 - t_layers_to_cool, 
                            thermocline2 = self.params.num_layers - t_layers_to_cool,
                            params = self.params, 
                        )
                    )
                    # Combine m and b
                    candidate_states.append(
                        HeatPumpWaterTankState(
                            top_temp = t, 
                            middle_temp = (m*(th2-th1) + b*(self.params.num_layers-th2)) / (self.params.num_layers-th1), 
                            bottom_temp = t_cooled, 
                            thermocline1 = th1 - t_layers_to_cool, 
                            thermocline2 = self.params.num_layers - t_layers_to_cool,
                            params = self.params, 
                        )
                    )
                    # Combine b and t_cooled
                    candidate_states.append(
                        HeatPumpWaterTankState(
                            top_temp = t, 
                            middle_temp = m, 
                            bottom_temp = (b*(self.params.num_layers-th2) + t_cooled*t_layers_to_cool) / (self.params.num_layers-th2+t_layers_to_cool), 
                            thermocline1 = th1 - t_layers_to_cool, 
                            thermocline2 = th2 - t_layers_to_cool,
                            params = self.params, 
                        )
                    )

                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))

        elif t_cooled > b:
            '''
            Cooled water from the top is still warmer than the bottom layer.
            Two main scenarios exist:
                1. All of the top will be cooled, then we will have a (t,m) case
                2. Not all of the top will be cooled, we should determine how much of it will be

            In both cases, if m > t_cooled, only the cooled top and bottom will mix.
            Otherwise the middle will also mix.
            '''

            cooled_top_energy = HeatPumpWaterTankState(t_cooled,m,b,th1,th2,self.params).energy
            t_layers = th1

            # We will discharge past the top layers
            if cooled_top_energy > next_state_energy:
                if m > t_cooled:
                    mixed_layers = self.params.num_layers - th2 + t_layers
                    mixed_temp = (b*(self.params.num_layers - th2) + t_cooled*t_layers)/mixed_layers
                    return self.discharge_tm(m, mixed_temp, self.params.num_layers-mixed_layers, next_state_energy)
                else:
                    mixed_temp = (t_cooled*th1 + m*(th2-th1) + b*(self.params.num_layers-th2)) / self.params.num_layers
                    return self.discharge_t(mixed_temp, next_state_energy)
                
            # We will not discharge all top layers
            else:
                current_energy = HeatPumpWaterTankState(t,m,b,th1,th2,self.params).energy
                energy_cooling_one_layer = (cooled_top_energy-current_energy) / t_layers
                layers_to_cool = (next_state_energy-current_energy) / energy_cooling_one_layer
                layers_to_cool = min(layers_to_cool, t_layers-1)

                candidate_states = []
                for t_layers_to_cool in [int(layers_to_cool), int(layers_to_cool)+1]:
                    if m > t_cooled:
                        mixed_layers = self.params.num_layers - th2 + t_layers_to_cool
                        mixed_temp = (b*(self.params.num_layers - th2) + t_cooled*t_layers_to_cool)/mixed_layers
                        state = HeatPumpWaterTankState(t, m, mixed_temp, th1-t_layers_to_cool, th2-t_layers_to_cool, self.params)
                    else:
                        mixed_layers = self.params.num_layers - (th1 - t_layers_to_cool)
                        mixed_temp = (t_cooled*(t_layers_to_cool) + m*(th2-th1) + b*(self.params.num_layers-th2)) / mixed_layers
                        state = HeatPumpWaterTankState(t, mixed_temp, mixed_temp, th1-t_layers_to_cool, th1-t_layers_to_cool, self.params)
                    candidate_states.append(state)

                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))

    def discharge_tm(self, t, m, th1, next_state_energy) -> HeatPumpWaterTankState:
        if th1==self.params.num_layers:
            return self.discharge_t(t, next_state_energy)

        t_cooled = self.get_discharge_return_temp(t)

        if t_cooled == m:
            '''
            Cooled water from the top is at the same temperature as the middle layer.
            We can just move the thermoclines down until either:
                - We cooled the storage by more than store_heat_in kWh
                - We ran out of top layers, giving us a (t) case
            '''
            candidate_states = []
            found_candidates = False
            
            while th1 > 1:
                th1 += -1
                state = HeatPumpWaterTankState(t,m,m,th1,self.params.num_layers,self.params)
                candidate_states.append(state)
                self.logger.debug(f"Adding {state} to candidates {state.energy}")
                if next_state_energy > state.energy:
                    found_candidates = True
                    break
            
            if found_candidates:
                # We will not cool down all top layers
                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))
            else:
                # We will cool past the top layers
                return self.discharge_t(m, next_state_energy)  

        elif t_cooled < m:
            '''
            Cooled water from the top is colder than the bottom layer.
            Two main scenarios exist:
                1. All of the top will be cooled, then we will have a new (t,m,b) case
                2. Not all of the top will be cooled, we should determine how much of it will be
            '''

            cooled_top_energy = HeatPumpWaterTankState(t_cooled, m, m, th1, th1, self.params).energy
            t_layers = th1

            # We will discharge past the top layers
            if cooled_top_energy > next_state_energy:
                return self.discharge_tm(m, t_cooled, self.params.num_layers-th1, next_state_energy)

            # We will not discharge all top layers
            else:
                current_energy = HeatPumpWaterTankState(t,m,m,th1,th1,self.params).energy
                energy_cooling_one_layer = (cooled_top_energy - current_energy) / t_layers
                layers_to_cool = (next_state_energy - current_energy) / energy_cooling_one_layer
                layers_to_cool = min(layers_to_cool, t_layers-1)

                candidate_states = []
                for t_layers_to_cool in [int(layers_to_cool), int(layers_to_cool)+1]:
                    # Node is (t, m, t_cooled) => new (t,m,b) case
                    if t_layers_to_cool == 0:
                        state = HeatPumpWaterTankState(t, m, m, th1, th1, self.params)
                    else:
                        state = self.discharge_tmb(t, m, t_cooled, th1-t_layers_to_cool, self.params.num_layers-t_layers_to_cool, next_state_energy)         
                    candidate_states.append(state)

                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))

        elif t_cooled > m:
            '''
            Cooled water from the top is still warmer than the bottom layer.
            Two main scenarios exist:
                1. All of the top will be cooled, then we will have a (t) case
                2. Not all of the top will be cooled, we should determine how much of it will be
            '''

            cooled_top_energy = HeatPumpWaterTankState(t_cooled,m,m,th1,th1,self.params).energy
            t_layers = th1

            # We will discharge past the top layers
            if cooled_top_energy > next_state_energy:
                mixed_layers = self.params.num_layers
                mixed_temp = (t_cooled*t_layers + m*(self.params.num_layers-t_layers))/mixed_layers
                return self.discharge_t(mixed_temp, next_state_energy)
                
            # We will not discharge all top layers
            else:
                current_energy = HeatPumpWaterTankState(t,m,m,th1,th1,self.params).energy
                energy_cooling_one_layer = (cooled_top_energy-current_energy) / t_layers
                layers_to_cool = (next_state_energy-current_energy) / energy_cooling_one_layer
                layers_to_cool = min(layers_to_cool, t_layers-1)

                candidate_states = []
                for t_layers_to_cool in [int(layers_to_cool), int(layers_to_cool)+1]:
                    mixed_layers = self.params.num_layers - th1 + t_layers_to_cool
                    mixed_temp = (m*(self.params.num_layers - th1) + t_cooled*t_layers_to_cool)/mixed_layers
                    state = HeatPumpWaterTankState(t, mixed_temp, mixed_temp, th1-t_layers_to_cool, th1-t_layers_to_cool, self.params)
                    candidate_states.append(state)

                return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))

    def discharge_t(self, t, next_state_energy) -> HeatPumpWaterTankState:
        t_cooled = self.get_discharge_return_temp(t)
        if t_cooled==t:
            return HeatPumpWaterTankState(t, t, t, self.params.num_layers, self.params.num_layers, self.params)
        
        '''
        Cooled water can only be colder than the top layer.
        Two main scenarios exist:
            1. All of the top will be cooled, then we will have a new (t) case
            2. Not all of the top will be cooled, we should determine how much of it will be
        '''

        cooled_top_energy = HeatPumpWaterTankState(t_cooled,t_cooled,t_cooled,self.params.num_layers,self.params.num_layers, self.params).energy
        t_layers = self.params.num_layers

        # We will discharge past the top layers
        if cooled_top_energy > next_state_energy:
            return self.discharge_t(t_cooled, next_state_energy)
            
        # We will not discharge all top layers
        else:
            current_energy = HeatPumpWaterTankState(t,t,t,self.params.num_layers,self.params.num_layers, self.params).energy
            energy_cooling_one_layer = (cooled_top_energy-current_energy) / t_layers
            layers_to_cool = (next_state_energy-current_energy) / energy_cooling_one_layer
            layers_to_cool = min(layers_to_cool, t_layers-1)

            candidate_states = []
            for t_layers_to_cool in [int(layers_to_cool), int(layers_to_cool)+1]:
                # Node is (t,t_cooled) => new (t,m) case
                if t_layers_to_cool == 0:
                    state = HeatPumpWaterTankState(t, t, t, self.params.num_layers, self.params.num_layers, self.params)
                else:
                    state = self.discharge_tm(t, t_cooled, self.params.num_layers-t_layers_to_cool, next_state_energy)  
                candidate_states.append(state)
            
            return min(list(candidate_states), key=lambda x: abs(x.energy-next_state_energy))


# if __name__ == "__main__":
#     import logging
#     from logging import Logger
#     import matplotlib.pyplot as plt
    
#     params = WinterOakSupergraphParams(
#         num_layers=27,
#         storage_volume_gallons=360,
#         constant_delta_t=20,
#         max_hp_kwh_th=25,
#         max_load_kwh_th=20
#     )
#     model = RuleBasedStorageModel(params, [], {}, logging.getLogger())
#     temps = [t for t in range(90, 181)]
#     return_temps = [model.get_discharge_return_temp(t) for t in temps]
#     plt.figure()
#     plt.plot(temps, return_temps)
#     pts_x = list(model.discharge_return_temp.keys())
#     pts_y = list(model.discharge_return_temp.values())
#     plt.scatter(pts_x, pts_y, color="C1", zorder=5)
#     for x, y in model.discharge_return_temp.items():
#         plt.annotate(f"({x}, {y})", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
#     plt.xlabel("Discharge supply temp (°F)")
#     plt.ylabel("Return temp (°F)")
#     plt.title("get_discharge_return_temp")
#     plt.grid(True)
#     plt.show()
