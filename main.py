import time
import pandas as pd

from bruce.graph_optimizer.assets.heat_pump_water_tank import (
    HeatPumpWaterTankAsset,
    HeatPumpWaterTankParams,
    plot_graph_results,
)
from bruce.graph_optimizer import Graph, get_logger, plot_bid, setup_logging

setup_logging()
logger = get_logger("main")

# Temporary: example input data
LMP = [28.14, 28.53, 46.38, 44.84, 44.8, 32.94, 39.68, 32.07, 30.07, 27.95, 29.87, 35.63, 45.63, 50.21, 54.74, 55.89, 46.9, 40.9, 33.31, 25.57, 27.12, 26.18, 24.94, 24.54, 28.14, 28.53, 46.38, 44.84, 44.8, 32.94, 39.68, 32.07, 30.07, 27.95, 29.87, 35.63, 45.63, 50.21, 54.74, 55.89, 46.9, 40.9, 33.31, 25.57, 27.12, 26.18, 24.94, 24.54]
DIST = [50.13, 50.13, 487.63, 487.63, 487.63, 487.63, 487.63, 54.98, 54.98, 54.98, 54.98, 487.63, 487.63, 487.63, 487.63, 50.13, 50.13, 50.13, 50.13, 50.13, 50.13, 50.13, 50.13, 50.13, 50.13, 50.13, 487.63, 487.63, 487.63, 487.63, 487.63, 54.98, 54.98, 54.98, 54.98, 487.63, 487.63, 487.63, 487.63, 50.13, 50.13, 50.13, 50.13, 50.13, 50.13, 50.13, 50.13, 50.13]
input_data = {
    'elec_usd_mwh': [lmp+dist for lmp, dist in zip(LMP, DIST)],
    'rswt_f': [140]*len(LMP),
    'load_kwh': [5]*len(LMP),
    'oat_f': [30]*len(LMP)
}
df = pd.DataFrame(input_data)

# df = pd.read_csv('data/input_data.csv')

horizon_timesteps = 48
timestep_duration_hours = [1.0] * horizon_timesteps

params = HeatPumpWaterTankParams(
    horizon=horizon_timesteps,
    start_unix_s=int(time.time()),
    site_id="local.dev",
    timestep_duration_hours=timestep_duration_hours,
    elec_usd_mwh = df['elec_usd_mwh'].tolist()[:horizon_timesteps],
    rswt_f = df['rswt_f'].tolist()[:horizon_timesteps],
    load_kwh = df['load_kwh'].tolist()[:horizon_timesteps],
    oat_f = df['oat_f'].tolist()[:horizon_timesteps],
)
asset = HeatPumpWaterTankAsset(params)
graph = Graph(asset)
graph.find_shortest_path()
plot_graph_results(graph)

forecast_price = params.elec_usd_mwh[0]
pq_pairs = graph.generate_bid(forecast_price_usd_mwh=forecast_price)
plot_bid(pq_pairs, forecast_price)