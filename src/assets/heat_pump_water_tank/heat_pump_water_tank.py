from ..base import Action, Asset, Params, State


class HeatPumpWaterTankParams(Params):
    pass


class HeatPumpWaterTankState(State):
    pass


class HeatPumpWaterTankAction(Action):
    pass


class HeatPumpWaterTankAsset(
    Asset[HeatPumpWaterTankState, HeatPumpWaterTankAction, HeatPumpWaterTankParams]
):

    @property
    def name(self) -> str:
        return "heat_pump_water_tank"

    def get_state_space(self) -> list[HeatPumpWaterTankState]:
        raise NotImplementedError

    def get_action_space(self) -> list[HeatPumpWaterTankAction]:
        raise NotImplementedError

    def get_available_actions(
        self, state: HeatPumpWaterTankState, params: HeatPumpWaterTankParams, time_step: int
    ) -> list[HeatPumpWaterTankAction]:
        raise NotImplementedError

    def next_state(
        self, state: HeatPumpWaterTankState, action: HeatPumpWaterTankAction
    ) -> HeatPumpWaterTankState:
        raise NotImplementedError

    def state_distance(
        self, state1: HeatPumpWaterTankState, state2: HeatPumpWaterTankState
    ) -> float:
        raise NotImplementedError

    def cost_function(
        self,
        state: HeatPumpWaterTankState,
        action: HeatPumpWaterTankAction,
        params: HeatPumpWaterTankParams,
        time_step: int,
    ) -> float:
        raise NotImplementedError
