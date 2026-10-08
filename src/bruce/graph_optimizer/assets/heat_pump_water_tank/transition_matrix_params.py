from typing import Self

from ..base import Params, TransitionMatrixParams
from .params import HeatPumpWaterTankParams


class HeatPumpWaterTankTransitionMatrixParams(TransitionMatrixParams):
    num_layers: int
    storage_volume_gallons: float
    hp_constant_lift_f: float
    load_bucket_upper_kwh: list[float]
    rwt_intercept_by_bucket: list[float]
    rwt_slope_by_bucket: list[float]
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
            load_bucket_upper_kwh=params.load_bucket_upper_kwh,
            rwt_intercept_by_bucket=params.rwt_intercept_by_bucket,
            rwt_slope_by_bucket=params.rwt_slope_by_bucket,
            rwt_min=params.rwt_min,
            max_load_kw_th=params.max_load_kw_th,
            hp_max_kw_th=params.hp_max_kw_th,
            max_timestep_duration_hours=max(params.timestep_duration_hours),
        )
