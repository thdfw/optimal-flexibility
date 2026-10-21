from typing import Literal

from ..base import Action


class HeatPumpWaterTankAction(Action):
    net_heat_to_store_kwh: float
    heat_source: Literal["hp", "elements1"] = "hp"

    def __repr__(self) -> str:
        return f"{self.heat_source}: net {self.net_heat_to_store_kwh}"
