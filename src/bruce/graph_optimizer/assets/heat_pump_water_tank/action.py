from typing import Self

from pydantic import model_validator

from ..base import Action


class HeatPumpWaterTankAction(Action):
    hp_heat_to_store_kwh: float = 0.0
    e1_heat_to_store_kwh: float = 0.0

    @model_validator(mode="after")
    def _hp_heat_to_store_xor_e1_heat_to_store(self) -> Self:
        hp_on = self.hp_heat_to_store_kwh != 0.0
        e1_on = self.e1_heat_to_store_kwh != 0.0
        if hp_on and e1_on:
            raise ValueError("hp_heat_to_store_kwh and e1_heat_to_store_kwh cannot both be non-zero")
        return self

    def __repr__(self) -> str:
        return f"HP: {self.hp_heat_to_store_kwh} - E1: {self.e1_heat_to_store_kwh}"
