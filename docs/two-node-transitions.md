# Two-node transitions

The nodes of the graph represent a discrete set of feasible states. From a given state and action, the model predicts the next state, and that prediction is then matched to the closest node (state) in the next layer of the graph. This matching step introduces an error, as the energy of the predicted and closest next states can differ. For the current model of the storage tanks, that error is up to about 1 kWh in either direction.

When given the opportunity, the FLO will exploit this and pick the paths along which rounding happens to add energy. This can make the FLO make sub-optimal decisions, especially in periods with a small price spread (e.g. weekend prices).

The two-node method presented below removes the bias introduced by the rounding step.

# Method

## Single head node (biased)

The graph is solved backward from the last time step. Each node stores its pathcost \(\phi(n)\): the minimum total cost from that node through the end of the horizon (including electricity costs, penalties, and the terminal cost).

Each edge has a single head node \(n\) on the next layer: the node whose state is closest to the model prediction. The selected edge is the one with the smallest total cost (\(c\) is the cost of this edge only):

\[
C_{\text{edge}} = c + \phi(n)
\]

When building the shortest path, we walk forward from the initial node, each step following that node's cheapest edge to its single head (`next_node`). The chain is fixed: rounding was chosen in the backward pass and cannot be undone.

## Two head nodes (unbiased)

Instead of a single head, each edge leads to two head nodes on the next layer: \(n^-\) with less stored energy than the prediction, and \(n^+\) with more. The backward pass does not commit to either head. It scores the edge as a weighted average of the two futures:

\[
C_{\text{edge}} = w^- \left( c^- + \phi(n^-) \right) + w^+ \left( c^+ + \phi(n^+) \right)
\]

where \(c^\pm\) is the cost of this hour if the head is \(n^\pm\) (the RSWT penalty depends on the head). The weights are chosen so the weighted average energy matches the predicted energy \(E^*\):

\[
w^+ = \frac{E^* - E(n^-)}{E(n^+) - E(n^-)}, \qquad w^- = 1 - w^+
\]

When there is no suitable node on one side (storage close to full or empty, or no state with a similar temperature profile), we keep an edge with a single head with weight 1.

When building the shortest path, we still follow each node's best edge hour by hour, but at each step we pick \(n^-\) or \(n^+\) whichever keeps the running gap between grid energy and the model's predicted energy smallest. In plain terms: same actions as before, but each rounding nudge goes the way that cancels earlier drift, so storage tracks physics within about one grid step instead of drifting several kWh over 48 hours.
