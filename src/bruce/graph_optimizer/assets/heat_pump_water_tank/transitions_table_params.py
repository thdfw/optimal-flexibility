from typing import Self

from ..base import Params, TransitionsTableParams
from .params import HeatPumpWaterTankParams


class HeatPumpWaterTankTransitionsTableParams(TransitionsTableParams):
    num_layers: int
    storage_volume_gallons: float
    hp_constant_lift_f: float
    rwt_intercept: float
    rwt_slope: float
    rwt_min: float
    max_load_kw_th: float
    hp_max_kw_th: float
    max_timestep_duration_hours: float

    @classmethod
    def from_asset_params(cls, params: Params) -> Self:
        if not isinstance(params, HeatPumpWaterTankParams):
            raise TypeError(f"expected HeatPumpWaterTankParams, got {type(params).__name__}")
        return cls(
            num_layers=params.num_layers,
            storage_volume_gallons=params.storage_volume_gallons,
            hp_constant_lift_f=params.hp_constant_lift_f,
            rwt_intercept=params.rwt_intercept,
            rwt_slope=params.rwt_slope,
            rwt_min=params.rwt_min,
            max_load_kw_th=params.max_load_kw_th,
            hp_max_kw_th=params.hp_max_kw_th,
            max_timestep_duration_hours=max(params.timestep_duration_hours),
        )
