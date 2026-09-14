import pandas as pd

from assets.heat_pump_water_tank import HeatPumpWaterTankAsset, HeatPumpWaterTankParams
from optimizer.graph import Graph

df = pd.read_csv('data/input_data.csv')
horizon_timesteps = 48
timestep_duration_hours = [1.0] * horizon_timesteps

params = HeatPumpWaterTankParams(
    horizon=horizon_timesteps,
    timestep_duration_hours=timestep_duration_hours,
    elec_usd_mwh = df['elec_usd_mwh'].tolist()[:horizon_timesteps],
    rswt_f = df['rswt_f'].tolist()[:horizon_timesteps],
    load_kwh = df['load_kwh'].tolist()[:horizon_timesteps],
    oat_f = df['oat_f'].tolist()[:horizon_timesteps],
)
asset = HeatPumpWaterTankAsset(params)
graph = Graph(asset)
