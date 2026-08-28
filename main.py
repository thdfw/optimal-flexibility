from assets.heat_pump_water_tank import HeatPumpWaterTankAsset, HeatPumpWaterTankParams
from optimizer.graph import Graph

params = HeatPumpWaterTankParams(
    horizon=10
)

asset = HeatPumpWaterTankAsset(params)
graph = Graph(asset)
