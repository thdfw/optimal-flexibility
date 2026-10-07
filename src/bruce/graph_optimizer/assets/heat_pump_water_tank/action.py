from ..base import Action


class HeatPumpWaterTankAction(Action):
    heat_to_store_kwh: float

    def __repr__(self) -> str:
        return str(self.heat_to_store_kwh)
