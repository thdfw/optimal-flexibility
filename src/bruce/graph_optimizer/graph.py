import contextlib
import gc
import time
from dataclasses import dataclass
from typing import Generic

from bruce.graph_optimizer.assets.base import A, Asset, P, S
from bruce.graph_optimizer.settings import get_logger
from bruce.graph_optimizer.transitions_matrix import get_transition_matrix


@dataclass(frozen=True)
class PriceQuantityPair:
    price_mwh: float
    quantity_kwh: float


class Node(Generic[S]):
    def __init__(self, state: S, time_step: int):
        self.state = state
        self.time_step = time_step
        self.pathcost = 1e9
        self.next_node: Node[S] | None = None
        self.shortest_path_elec: list[float] = []

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
        self.initial_node: Node[S] | None = None
        self._time_and_log(self.load_transitions, "Loaded transition matrix")
        self._time_and_log(self.create_nodes, "Created nodes")
        self._time_and_log(self.create_edges, "Created edges")

    def _time_and_log(self, func, label: str) -> None:
        start = time.perf_counter()
        func()
        elapsed = round(time.perf_counter() - start, 1)
        self.logger.info(f"{label} in {elapsed} seconds")

    def load_transitions(self) -> None:
        self.transition_matrix = get_transition_matrix(self.asset)

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
                    variant = self.asset.transition_variant(node.state, action, time_step)
                    if self.transition_matrix.ndim == 2:
                        next_state_index = int(
                            self.transition_matrix[action.index, node.state.index]
                        )
                    else:
                        next_state_index = int(
                            self.transition_matrix[action.index, node.state.index, variant]
                        )
                    next_state = self.asset.state_space[next_state_index]
                    if not self.asset.allow_transition(node.state, next_state, action, time_step):
                        continue
                    next_node = self.nodes[time_step+1][next_state_index]
                    cost = self.asset.cost(node.state, next_state, action, time_step)
                    elec_used_kwh = self.asset.elec_used_kwh(node.state, action, time_step)
                    edge = Edge(node, next_node, cost, action, elec_used_kwh)
                    self.edges[node].append(edge)
                    if time_step <= 1:
                        self.bid_edges[node].append(edge)

        del self.transition_matrix

    def find_shortest_path(self) -> None:
        self._time_and_log(self._find_shortest_path, "Found shortest path")
        self._populate_shortest_path_elec()

    def _find_shortest_path(self) -> None:
        for time_step in range(self.N - 1, -1, -1):
            for node in self.nodes[time_step]:
                if not self.edges[node]:
                    self.logger.warning(f"No edges found for node {node}")
                    continue
                best_edge = min(self.edges[node], key=lambda e: e.head.pathcost + e.cost)
                node.pathcost = best_edge.head.pathcost + best_edge.cost
                node.next_node = best_edge.head

    def _populate_shortest_path_elec(self) -> None:
        if not hasattr(self, "nodes") or 1 not in self.nodes:
            return
        path_steps = getattr(self.params, "stability_penalty_horizon_hours", self.N)
        path_steps = min(int(path_steps), self.N)
        for node in self.nodes[1]:
            node.shortest_path_elec = []
            current = node
            for _ in range(path_steps):
                if current.next_node is None:
                    break
                edge_list = self.edges.get(current)
                if not edge_list:
                    break
                edge = next((e for e in edge_list if e.head is current.next_node), None)
                if edge is None:
                    break
                node.shortest_path_elec.append(round(edge.elec_used_kwh, 1))
                current = current.next_node

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

    def generate_bid(self, forecast_price_mwh: float, updated_params: P | None = None) -> list[PriceQuantityPair]:
        if updated_params is not None:
            self.params.validate_bid_params_update(updated_params)
            self.asset.update_params(updated_params)
            self.params = self.asset.params

        initial_node = self.find_initial_node()

        bid_edges = self.bid_edges.get(initial_node, [])
        if not bid_edges:
            raise ValueError(f"No bid edges from initial node {initial_node}")

        bid_min = int(self.params.bid_min_price_mwh)
        bid_max = int(self.params.bid_max_price_mwh)
        price_range_mwh = sorted(set(range(bid_min, bid_max)) | {forecast_price_mwh})
        forecast_price_kwh = forecast_price_mwh / 1000

        pq_pairs: list[PriceQuantityPair] = []

        for trial_price in price_range_mwh:
            trial_price_kwh = float(trial_price) / 1000

            best_edge = min(
                bid_edges,
                key=lambda e: (
                    e.head.pathcost
                    + e.elec_used_kwh * trial_price_kwh # Cost of elec used at trial price
                    + (e.cost - e.elec_used_kwh * forecast_price_kwh) # Cost of penalties
                ),
            )
            best_quantity = max(0, best_edge.elec_used_kwh)

            if not pq_pairs or best_quantity - pq_pairs[-1].quantity_kwh > 0.01:
                pq_pairs.append(
                    PriceQuantityPair(
                        price_mwh=float(trial_price),
                        quantity_kwh=best_quantity,
                    )
                )

        best_at_forecast = min(bid_edges, key=lambda e: e.head.pathcost + e.cost)
        initial_node.pathcost = best_at_forecast.head.pathcost + best_at_forecast.cost
        initial_node.next_node = best_at_forecast.head
        self.initial_node = initial_node

        self.logger.info(f"Done ({len(pq_pairs)} PQ pairs found).")
        return pq_pairs

    def get_next_node_at_price(self, price_mwh: float) -> None:
        """Pick the best hour-0 edge at clearing price and set ``initial_node.next_node``."""
        if self.initial_node is None:
            self.initial_node = self.find_initial_node()
        initial_node = self.initial_node
        bid_edges = self.bid_edges.get(initial_node, [])
        if not bid_edges:
            raise ValueError(f"No bid edges from initial node {initial_node}")

        forecast_price_mwh = self.params.elec_price_mwh[0]
        forecast_price_kwh = forecast_price_mwh / 1000
        trial_price_kwh = price_mwh / 1000

        best_edge = min(
            bid_edges,
            key=lambda e: (
                e.head.pathcost
                + e.elec_used_kwh * trial_price_kwh
                + (e.cost - e.elec_used_kwh * forecast_price_kwh)
            ),
        )
        initial_node.next_node = best_edge.head
