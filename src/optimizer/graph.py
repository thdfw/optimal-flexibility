import contextlib
import gc
import time
from dataclasses import dataclass
from typing import Generic

from assets.base import A, Asset, P, S
from optimizer.settings import get_logger
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
        self.logger = get_logger("graph")
        self.asset = asset
        self.params = asset.params
        self.N = asset.params.horizon
        self._time_and_log(self.load_transitions, "Loaded transitions table")
        self._time_and_log(self.create_nodes, "Created nodes")
        self._time_and_log(self.create_edges, "Created edges")
        self._time_and_log(self.find_shortest_path, "Found shortest path")

    def _time_and_log(self, func, label: str) -> None:
        start = time.perf_counter()
        func()
        elapsed = round(time.perf_counter() - start, 1)
        self.logger.info(f"{label} in {elapsed} seconds")

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
        self.bid_nodes: list[Node[S]] = list(self.nodes[0])

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
                    self.logger.warning(f"No edges found for node {node}")
                    continue
                best_edge = min(self.edges[node], key=lambda e: e.head.pathcost + e.cost)
                node.pathcost = best_edge.head.pathcost + best_edge.cost
                node.next_node = best_edge.head

    def find_initial_node(self) -> Node[S]:
        closest = self.asset.closest_state(self.asset.initial_state())
        nodes_by_0 = getattr(self, "nodes_by", None)
        if nodes_by_0 is not None:
            return nodes_by_0[0][closest]
        for node in self.bid_nodes:
            if node.state == closest:
                return node
        raise RuntimeError(f"No step-0 node for initial state {closest}")

    def trim_graph_for_waiting(self) -> None:
        """Keep only nodes and edges for the first time step to generate a bid later."""
        start = time.perf_counter()
        if self.N >= 2:
            for node in self.nodes[2]:
                node.next_node = None
        with contextlib.suppress(AttributeError):
            del self.nodes_by
        with contextlib.suppress(AttributeError):
            del self.nodes
        with contextlib.suppress(AttributeError):
            del self.edges
        gc.collect()
        elapsed = round(time.perf_counter() - start, 1)
        self.logger.info(f"Trimmed graph in {elapsed} seconds")

    def cleanup(self) -> None:
        """Break circular references so the graph can be garbage-collected."""
        all_node_ids: set[int] = set()
        nodes_to_clear: list[Node[S]] = []
        edges_to_clear: list[Edge[S, A]] = []

        bid_edges: dict[Node[S], list[Edge[S, A]]] = getattr(self, "bid_edges", None) or {}
        for node, edge_list in bid_edges.items():
            if id(node) not in all_node_ids:
                all_node_ids.add(id(node))
                nodes_to_clear.append(node)
            for edge in edge_list:
                edges_to_clear.append(edge)
                for n in (edge.tail, edge.head):
                    if n is not None and id(n) not in all_node_ids:
                        all_node_ids.add(id(n))
                        nodes_to_clear.append(n)

        for node in getattr(self, "bid_nodes", None) or []:
            if id(node) not in all_node_ids:
                all_node_ids.add(id(node))
                nodes_to_clear.append(node)

        i = 0
        while i < len(nodes_to_clear):
            nxt = nodes_to_clear[i].next_node
            if nxt is not None and id(nxt) not in all_node_ids:
                all_node_ids.add(id(nxt))
                nodes_to_clear.append(nxt)
            i += 1

        for node in nodes_to_clear:
            node.next_node = None
        for edge in edges_to_clear:
            edge.tail = None  # type: ignore[assignment]
            edge.head = None  # type: ignore[assignment]

        for edge_list in bid_edges.values():
            edge_list.clear()

        for attr in ("bid_edges", "bid_nodes", "nodes", "nodes_by", "edges"):
            with contextlib.suppress(AttributeError):
                delattr(self, attr)

    def generate_bid(self, forecast_price_usd_mwh: float, updated_params: P | None = None) -> list[PriceQuantityPair]:
        if updated_params is not None:
            self.asset.update_params(updated_params)
            self.params = self.asset.params

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

        self.logger.info(f"Done ({len(pq_pairs)} PQ pairs found).")
        return pq_pairs
