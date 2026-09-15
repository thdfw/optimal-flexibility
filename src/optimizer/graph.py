import time
from dataclasses import dataclass
from typing import Generic

from assets.base import A, Asset, P, S
from optimizer.transitions_table import get_transitions_table


@dataclass(frozen=True)
class PriceQuantityPair:
    price_usd_mwh: float
    quantity_kwh: float


class Node(Generic[S]):
    def __init__(self, state: S, time_step: int):
        self.state = state
        self.time_step = time_step
        self.pathcost = 1e9
        self.next_node: Node[S] | None = None

    def __repr__(self):
        return f"[{self.time_step}]{self.state}"


class Edge(Generic[S, A]):
    def __init__(
        self,
        tail: Node[S],
        head: Node[S],
        cost: float,
        action: A,
        elec_used_kwh: float,
    ):
        self.tail: Node[S] = tail
        self.head: Node[S] = head
        self.cost = cost
        self.action = action
        self.elec_used_kwh = elec_used_kwh

    def __repr__(self):
        return f"Edge[{self.tail} --{round(self.cost, 3)}--> {self.head}]"


class Graph(Generic[S, A, P]):
    def __init__(self, asset: Asset[S, A, P]):
        self.asset = asset
        self.params = asset.params
        self.N = asset.params.horizon
        self._time_and_log(self.load_transitions, "transitions")
        self._time_and_log(self.create_nodes, "nodes")
        self._time_and_log(self.create_edges, "edges")
        self._time_and_log(self.find_shortest_path, "shortest_path")

    def _time_and_log(self, func, step: str) -> None:
        start = time.perf_counter()
        func()
        elapsed = round(time.perf_counter() - start, 1)
        if step == "transitions":
            print(f"Loaded transitions table in {elapsed} seconds")
        elif step == "nodes":
            print(f"Created nodes in {elapsed} seconds")
        elif step == "edges":
            print(f"Created edges in {elapsed} seconds")
        elif step == "shortest_path":
            print(f"Found shortest path in {elapsed} seconds")

    def load_transitions(self) -> None:
        self.transitions = get_transitions_table(self.asset)
        self.states_by_key = {state.to_key(): state for state in self.asset.state_space}

    def create_nodes(self):
        """For every time step, create a layer of nodes corresponding to all available states."""
        self.nodes: dict[int, list[Node[S]]] = {
            time_step: [Node(state, time_step) for state in self.asset.state_space]
            for time_step in range(self.N + 1)
        }

        for node in self.nodes[self.N]:
            node.pathcost = 0

        self.nodes_by: dict[int, dict[S, Node[S]]] = {
            time_step: {node.state: node for node in self.nodes[time_step]}
            for time_step in range(self.N + 1)
        }

    def create_edges(self):
        """Create edges for each available (node, action) pair with the corresponding cost."""
        self.edges: dict[Node[S], list[Edge[S, A]]] = {}
        self.bid_edges: dict[Node[S], list[Edge[S, A]]] = {}

        for time_step in range(self.N):
            for node in self.nodes[time_step]:
                self.edges[node] = []
                if time_step <= 1:
                    self.bid_edges[node] = []

                available_actions = self.asset.get_available_actions(node.state, time_step)

                for action in available_actions:
                    next_state_key = self.transitions[action.to_key()][node.state.to_key()]
                    next_state = self.states_by_key[next_state_key]
                    if not self.asset.allow_transition(node.state, next_state, action, time_step):
                        continue
                    next_node = self.nodes_by[time_step + 1][next_state]
                    cost = self.asset.cost(node.state, next_state, action, time_step)
                    elec_used_kwh = self.asset.elec_used_kwh(node.state, action, time_step)
                    edge = Edge(node, next_node, cost, action, elec_used_kwh)
                    self.edges[node].append(edge)
                    if time_step <= 1:
                        self.bid_edges[node].append(edge)

        del self.transitions
        del self.states_by_key

    def find_shortest_path(self):
        """Find the shortest path using backward induction."""
        for time_step in range(self.N - 1, -1, -1):
            for node in self.nodes[time_step]:
                if not self.edges[node]:
                    print(f"No edges found for node {node}")
                    continue
                best_edge = min(self.edges[node], key=lambda e: e.head.pathcost + e.cost)
                node.pathcost = best_edge.head.pathcost + best_edge.cost
                node.next_node = best_edge.head

    def find_initial_node(self) -> Node[S]:
        closest = self.asset.closest_state(self.asset.initial_state())
        return self.nodes_by[0][closest]

    def generate_bid(self, forecast_price_usd_mwh: float, initial_node: Node[S] | None = None) -> list[PriceQuantityPair]:
        if initial_node is None:
            initial_node = self.find_initial_node()

        bid_edges = self.bid_edges.get(initial_node, [])
        if not bid_edges:
            raise ValueError(f"No bid edges from initial node {initial_node}")

        price_range_usd_mwh = sorted(set(range(-100, 2000)) | {forecast_price_usd_mwh})
        forecast_usd_kwh = forecast_price_usd_mwh / 1000
        
        pq_pairs: list[PriceQuantityPair] = []

        for trial_price in price_range_usd_mwh:
            trial_usd_kwh = float(trial_price) / 1000

            best_edge = min(
                bid_edges, 
                key=lambda e: 
                e.head.pathcost 
                + e.elec_used_kwh * trial_usd_kwh # Cost of elec used at trial price
                + (e.cost - e.elec_used_kwh * forecast_usd_kwh), # Cost of penalties
            )
            best_quantity = max(0, best_edge.elec_used_kwh)

            if not pq_pairs or best_quantity - pq_pairs[-1].quantity_kwh > 0.01:
                pq_pairs.append(
                    PriceQuantityPair(
                        price_usd_mwh=float(trial_price),
                        quantity_kwh=best_quantity,
                    )
                )

        best_at_forecast = min(bid_edges, key=lambda e: e.head.pathcost + e.cost)
        initial_node.pathcost = best_at_forecast.head.pathcost + best_at_forecast.cost
        initial_node.next_node = best_at_forecast.head

        return pq_pairs
