# FLO penalties

The graph is made of nodes organized in 48 layers, which represent the feasible tank states in every time step, as well as edges, which represent all feasible transitions between adjacent time steps. Each edge (transition) has a cost, and the goal is to find the path that has the lowest cost from the current node to any node at the end of the 48-hour horizon.

# RSWT penalty

If all physically feasible transitions were allowed and the cost of a transition was just the cost of electricity used, then the lowest cost path would always be the one that keeps the heat pump off whenever the electricity price is positive. Such a path eventually leads to the tanks being below RSWT, which means it will eventually fail to keep the house warm. 

This is why it is necessary to include a RSWT constraint, which can be done either by excluding edges which would use water below RSWT (”hard constraint”), or by adding a cost penalty to such edges (”soft constraint”).

The main problem with removing below-RSWT edges is that it is overly conservative. This is because the graph only approximates reality: the model of the tanks is imperfect (temperatures are discretized in 10ºF steps, it allows at most three uniform layers, and assumes sharp thermoclines), and the house's thermal mass means a short period below RSWT doesn't immediately drop the house below setpoint. 

A path the model would label as "below RSWT" may therefore be perfectly acceptable in practice. A penalty that grows with how far below RSWT and how much water is drawn below RSWT lets the FLO make that trade-off, and can avoid running the heat pump in expensive hours to satisfy an uncertain boundary.

For each edge during which the storage is discharging and the top layer is below RSWT at the start or end of the time step, we apply the following cost penalty:

$$
\text{rswt-penalty}_t = \omega \cdot \bar{p} \cdot \lambda^{\,H - \min(t,\, H)} \cdot \sum_i V_i \cdot \max\left(0,\; e^{\,k\,(\text{RSWT}_t - T_i)} - 1\right)
$$

where:

- $\omega$ is the penalty weight (in kWh of electricity per gallon)
- $\bar{p}$ is the average electricity price over the planning horizon (in $/kWh)
- $\lambda$ is the decay factor
- $H$ is the hour after which the penalty reaches its full weight
- $t$ is the number of hours between the start of the horizon and the start of the time step
- $i$ indexes the layers of water drawn from the storage during the time step
- $V_i$ is the volume drawn from layer $i$ during the time step (in gallons)
- $k$ is the exponent rate
- $T_i$ is the temperature of layer $i$

Scaling the penalty by $\bar{p}$ allows us to use the same weights across different price levels (which can change depending on the location and time of year).

The parameters to tune are $\omega, \lambda, H, k$.

# Stability penalty

In a given time step, the amount of heat that will enter/leave the storage is the difference between the heat pump’s thermal output and the distribution system’s energy use (as measured in the basement). 

It has proven to be very difficult to achieve accurate hour-by-hour estimates of both the heat pump’s thermal output (due to defrosts and COP variability) and the distribution system’s energy use (non-continuous heat calls), resulting in relatively poor hour-by-hour estimates of the heat to the storage. On average, however, these estimates are close to accurate (the effect of distributed defrosts averages out between time steps, and water eventually returns from the house’s distribution system to the basement). 

The consequence is that the FLO will make 48-hour plans for a given state and set of forecasts, execute the first time step, and then often notice that the new state is very different from the one it had predicted. If we knew our forecasts to be highly accurate hour-by-hour, the right behavior for the FLO in this situation would be to immediately adjust the future plans. But since we know that our forecasts are only accurate on average, it is generally far better to avoid dramatic plan changes and trust that things will balance out eventually. 

That being said, sticking to previous plans at all costs is not the optimal strategy either, as there are some real deviations from predicted energy use (e.g. a window was left open, a thermostat setpoint was increased, weather was different than forecast, etc.). Changing plans as a consequence of large price variations should also remain acceptable.

For a strategy that combines both robustness against noise and flexibility in face of real variation, we use a stability penalty that adds a cost to plan changes that result from small tank state variations (more likely to be responding to noise). The cost of changing plans also decreases exponentially down the planning horizon.

If the currently stored energy is within $\Delta E_{\max}$ of what the previous plan predicted for the current time, and the electricity price forecast for the current hour is within $\Delta p_{\max}$ of the previous forecast, we apply the following cost penalty to each edge that starts within the first $H$ hours of the horizon:

$$
\text{stability-penalty}_t = \omega \cdot \bar{p} \cdot \lambda^{\,t} \cdot \left| e_t - \hat{e}_t \right|
$$

where:

- $\omega$ is the penalty weight (dimensionless)
- $\bar{p}$ is the average electricity price over the planning horizon (in $/kWh)
- $\lambda$ is the decay factor
- $t$ is the number of hours between the start of the horizon and the start of the time step
- $H$ is the number of hours after which the penalty no longer applies
- $e_t$ is the heat pump electricity use on the edge (in kWh)
- $\hat{e}_t$ is the heat pump electricity use that the previous plan had for the same time step (in kWh)

As for the RSWT penalty, scaling by $\bar{p}$ keeps the penalty's strength the same relative to the cost savings that would justify a plan change, whatever the price level.

The parameters to tune are $\omega, \lambda, H, \Delta E_{\max}, \Delta p_{\max}$.

# Terminal cost

Without a terminal cost, energy left in the storage at the end of the horizon has no value, so the FLO has an incentive to empty the storage by the last time step. In reality, heat stored at the end of the horizon is heat that won't have to be produced later.

We therefore give each node in the last layer a cost equal to the cost of producing the heat it would take to bring the storage back to full:

$$
\phi(s) = \frac{\bar{p}_{\text{end}}}{\overline{\text{COP}}_{\text{end}}} \cdot \left( E^{\max} - E(s) \right)
$$

where:

- $s$ is the tank state of the node
- $E(s)$ is the usable energy in state $s$ (in kWh), approximated by walking layers from the top down and summing energy relative to RSWT in each layer
- $E^{\max}$ is the same sum for a full storage at max temperature
- $\bar{p}_{\text{end}}$ is the average electricity price over the last 24 hours of the horizon (in $/kWh), used as a proxy for the prices just after the horizon
- $\overline{\text{COP}}_{\text{end}}$ is the average heat pump COP over the same 24 hours

This does not force the storage to be full at the end of the horizon. If electricity in the last hours is more expensive than $\bar{p}_{\text{end}}$, the FLO will still discharge the storage, since the stored heat is then worth less than it would cost to produce. Since only the first time step of each plan is executed, the terminal cost mainly keeps the end of the plans realistic and rarely changes the current decision.

There are no weights to tune; the only choice is the window used to compute $\bar{p}_{\text{end}}$ and $\overline{\text{COP}}_{\text{end}}$.

# OPTIONAL (not for now): Adding thermal mass

Because the house has a thermal mass, allowing the FLO to use some water below RSWT will not immediately result in breaking the SLA (the house dropping below setpoint). This is why using some water below RSWT can be acceptable in specific conditions: when done for a short period of time (e.g. not 5 hours in a row) and the thermal mass has to be re-gained before being used again (e.g. not 1 hour below RSWT, then 1 hour exactly at RSWT, and then again one hour below RSWT).

The conditions in which using the house’s thermal mass is acceptable (not too many time steps in a row, and the thermal mass needs to be replenished before future use) are both path-dependent constraints, which - in our graph approach - can only be handled by expanding the state space. This can be seen as adding a third dimension to the graph, in which nodes not only represent a time step and tank state, but also a house thermal mass state.

Adding M discrete house thermal mass states will make the graph M times larger, which will increase the time required to build and solve it, as well as the RAM usage. To make things simple and add the least possible states, choosing M=2 (the house’s thermal mass is either full or empty) seems like an appropriate direction.