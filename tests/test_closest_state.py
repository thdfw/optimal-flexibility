"""closest_state answers every state the model can predict. A large
discharge can predict a store with its second thermocline one layer below
the bottom of the tank, which is outside the state space; closest_state
compares the layers the tank has and answers a state of the space."""

from bruce.graph_optimizer.assets.heat_pump_water_tank.asset import (
    HeatPumpWaterTankAsset,
)
from bruce.service import params_from_request

from test_recommend import oak_request


def test_a_prediction_below_the_bottom_layer_has_a_closest_state() -> None:
    asset = HeatPumpWaterTankAsset(params_from_request(oak_request()))
    layers = asset.params.num_layers
    discharge = min(asset.action_space, key=lambda action: action.heat_to_store_kwh)
    assert discharge.heat_to_store_kwh < 0

    below_the_bottom = [
        (state, predicted)
        for state in asset.state_space
        if (predicted := asset.next_state(state, discharge)).thermocline2 > layers
    ]
    assert below_the_bottom

    space = set(asset.state_space)
    for state, predicted in below_the_bottom:
        closest = asset.closest_state(predicted, from_state=state, action=discharge)
        assert closest in space
