from typing import Self

from ..base import State
from .params import HeatPumpWaterTankParams
from ..utils import fahrenheit_to_kelvin


def compute_energy(
    top_temp: float,
    middle_temp: float,
    bottom_temp: float,
    thermocline1: int,
    thermocline2: int,
    params: HeatPumpWaterTankParams,
) -> float:
    m_layer_kg = params.storage_volume_gallons * 3.785 / params.num_layers
    kWh_top = thermocline1 * m_layer_kg * 4.187/3600 * fahrenheit_to_kelvin(top_temp)
    kWh_middle = (thermocline2 - thermocline1) * m_layer_kg * 4.187/3600 * fahrenheit_to_kelvin(middle_temp)
    kWh_bottom = (params.num_layers - thermocline2) * m_layer_kg * 4.187/3600 * fahrenheit_to_kelvin(bottom_temp)
    return kWh_top + kWh_middle + kWh_bottom


class HeatPumpWaterTankState(State):
    top_temp: float
    middle_temp: float
    bottom_temp: float
    thermocline1: int
    thermocline2: int
    energy: float

    def to_key(self) -> str:
        return (
            f"{self.top_temp}"
            f"({self.thermocline1})"
            f"{self.middle_temp}"
            f"({self.thermocline2})"
            f"{self.bottom_temp}"
        )

    @classmethod
    def build(
        cls,
        top_temp: float,
        middle_temp: float,
        bottom_temp: float,
        thermocline1: int,
        thermocline2: int,
        params: HeatPumpWaterTankParams,
    ) -> Self:
        return cls(
            top_temp=top_temp,
            middle_temp=middle_temp,
            bottom_temp=bottom_temp,
            thermocline1=thermocline1,
            thermocline2=thermocline2,
            energy=compute_energy(top_temp, middle_temp, bottom_temp, thermocline1, thermocline2, params),
        )
