from ..base import Action


class HeatPumpWaterTankAction(Action):
    heat_to_store_kwh: float

    def to_key(self) -> str:
        return str(self.heat_to_store_kwh)
