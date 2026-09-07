import pandas as pd

from assets.heat_pump_water_tank import HeatPumpWaterTankAsset, HeatPumpWaterTankParams
from optimizer.graph import Graph

df = pd.read_csv('data/input_data.csv')
horizon_hours = 48

params = HeatPumpWaterTankParams(
    horizon=horizon_hours,
    elec_usd_mwh = df['elec_usd_mwh'].tolist()[:horizon_hours],
    rswt_f = df['rswt_f'].tolist()[:horizon_hours],
    load_kwh = df['load_kwh'].tolist()[:horizon_hours],
    oat_f = df['oat_f'].tolist()[:horizon_hours],
)
asset = HeatPumpWaterTankAsset(params)
graph = Graph(asset)
