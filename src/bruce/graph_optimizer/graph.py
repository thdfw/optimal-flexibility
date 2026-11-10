import contextlib
import gc
import time
from dataclasses import dataclass
from typing import Any, Generic

from bruce.graph_optimizer.assets.base import A, Asset, P, S
from bruce.graph_optimizer.settings import get_logger
from bruce.graph_optimizer.transitions_matrix import get_transition_matrix


@dataclass(frozen=True)
class PriceQuantityPair:
    price_mwh: float
    quantity_kwh: float


class Node(Generic[S]):
    def __init__(self, state: S, time_step: int, mode: int = 0):
        self.state = state
        self.time_step = time_step
        self.mode = mode
        self.pathcost = 1e9
        self.best_edge: Edge[S, Any] | None = None
        self.shortest_path_elec: list[float] = []

    def __repr__(self):
        return f"[{self.time_step}|{self.mode}]{self.state}"


class Edge(Generic[S, A]):
    """A transition to one or more heads; ``cost`` is the weighted cost over the heads."""

    def __init__(
        self,
        tail: Node[S],
        heads: tuple[Node[S], ...],
        weights: tuple[float, ...],
        cost: float,
        action: A,
        elec_used_kwh: float,
    ):
        self.tail: Node[S] = tail
        self.heads: tuple[Node[S], ...] = heads
        self.weights: tuple[float, ...] = weights
        self.cost = cost
        self.action = action
        self.elec_used_kwh = elec_used_kwh

    @property
    def head_pathcost(self) -> float:
        return sum(w * head.pathcost for head, w in zip(self.heads, self.weights, strict=True))

    @property
    def value(self) -> float:
        return self.cost + self.head_pathcost

    def __repr__(self):
        return f"Edge[{self.tail} --{round(self.cost, 3)}--> {self.heads}]"


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
        """For every time step, create a layer of nodes for each (state, mode) pair."""
        modes = self.asset.get_modes()
        self.nodes: dict[int, list[Node[S]]] = {}
        self.nodes_by: dict[int, dict[tuple[int, int], Node[S]]] = {}
        for time_step in range(self.N + 1):
            layer: list[Node[S]] = []
            by_index: dict[tuple[int, int], Node[S]] = {}
            for mode in modes:
                for state in self.asset.state_space:
                    node = Node(state, time_step, mode)
                    layer.append(node)
                    by_index[(state.index, mode)] = node
            self.nodes[time_step] = layer
            self.nodes_by[time_step] = by_index

        for node in self.nodes[self.N]:
            node.pathcost = self.asset.terminal_node_cost(node.state)

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

                available_actions = self.asset.get_available_actions(node.state, time_step, node.mode)

                for action, next_mode in available_actions:
                    variant = self.asset.transition_variant(node.state, action, time_step)
                    successors = self.transition_matrix.successors(action.index, node.state.index, variant)
                    next_states = [self.asset.state_space[index] for index, _ in successors]
                    if not all(
                        self.asset.allow_transition(node.state, next_state, action, time_step)
                        for next_state in next_states
                    ):
                        continue
                    heads = tuple(self.nodes_by[time_step + 1][(index, next_mode)] for index, _ in successors)
                    weights = tuple(weight for _, weight in successors)
                    cost = sum(
                        weight * self.asset.cost(node.state, next_state, action, time_step, node.mode)
                        for next_state, weight in zip(next_states, weights, strict=True)
                    )
                    elec_used_kwh = self.asset.elec_used_kwh(node.state, action, time_step)
                    edge = Edge(node, heads, weights, cost, action, elec_used_kwh)
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
                best_edge = min(self.edges[node], key=lambda e: e.value)
                node.pathcost = best_edge.value
                node.best_edge = best_edge

    def rollout(self, first_edge: Edge[S, A], max_steps: int | None = None) -> list[tuple[Edge[S, A], Node[S]]]:
        """The plan from ``first_edge`` on: each step's edge and the head the plan moves to.

        Each step moves to the head that keeps the plan's ``rollout_quantity`` closest to the
        weighted one, so the plan does not drift from it (or to the heaviest head if the asset
        has no such quantity).
        """
        steps: list[tuple[Edge[S, A], Node[S]]] = []
        drift = 0.0
        edge: Edge[S, A] | None = first_edge
        while edge is not None and (max_steps is None or len(steps) < max_steps):
            quantities = [self.asset.rollout_quantity(head.state) for head in edge.heads]
            known = [q for q in quantities if q is not None]
            if len(edge.heads) == 1 or len(known) < len(quantities):
                head = max(zip(edge.heads, edge.weights, strict=True), key=lambda hw: hw[1])[0]
            else:
                target = sum(w * q for w, q in zip(edge.weights, known, strict=True))
                head, q = min(
                    zip(edge.heads, known, strict=True),
                    key=lambda hq: abs(drift + hq[1] - target),
                )
                drift += q - target
            steps.append((edge, head))
            edge = head.best_edge
        return steps

    def _populate_shortest_path_elec(self) -> None:
        if not hasattr(self, "nodes") or 1 not in self.nodes:
            return
        path_steps = getattr(self.params, "stability_penalty_horizon_hours", self.N)
        path_steps = min(int(path_steps), self.N)
        for node in self.nodes[1]:
            node.shortest_path_elec = []
            if node.best_edge is None:
                continue
            for edge, _ in self.rollout(node.best_edge, max_steps=path_steps):
                node.shortest_path_elec.append(round(edge.elec_used_kwh, 1))

    def find_initial_node(self) -> Node[S]:
        closest = self.asset.closest_state(self.asset.initial_state())
        mode = self.asset.initial_mode()
        nodes_by_0 = getattr(self, "nodes_by", None)
        if nodes_by_0 is not None:
            return nodes_by_0[0][(closest.index, mode)]
        for node in self.bid_nodes:
            if node.state == closest and node.mode == mode:
                return node
        raise RuntimeError(f"No step-0 node for initial state {closest} and mode {mode}")

    def trim_graph_for_waiting(self) -> None:
        """Keep only nodes and edges for the first time step to generate a bid later."""
        start = time.perf_counter()
        if self.N >= 2:
            for node in self.nodes[2]:
                node.best_edge = None
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
                for n in (edge.tail, *edge.heads):
                    if n is not None and id(n) not in all_node_ids:
                        all_node_ids.add(id(n))
                        nodes_to_clear.append(n)

        for node in getattr(self, "bid_nodes", None) or []:
            if id(node) not in all_node_ids:
                all_node_ids.add(id(node))
                nodes_to_clear.append(node)

        i = 0
        while i < len(nodes_to_clear):
            best_edge = nodes_to_clear[i].best_edge
            if best_edge is not None:
                for nxt in best_edge.heads:
                    if id(nxt) not in all_node_ids:
                        all_node_ids.add(id(nxt))
                        nodes_to_clear.append(nxt)
            i += 1

        for node in nodes_to_clear:
            node.best_edge = None
        for edge in edges_to_clear:
            edge.tail = None  # pyright: ignore[reportAttributeAccessIssue]
            edge.heads = ()

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
        # Walk the price domain from the market ceiling down. The first pair is
        # always at bid_max, so the pairs partition [bid_min, bid_max] into
        # segments: pair i holds for prices in (price_{i+1}, price_i].
        price_range_mwh = sorted(
            set(range(bid_min, bid_max + 1)) | {forecast_price_mwh}, reverse=True
        )
        forecast_price_kwh = forecast_price_mwh / 1000

        pq_pairs: list[PriceQuantityPair] = []

        for trial_price in price_range_mwh:
            trial_price_kwh = float(trial_price) / 1000

            best_edge = min(
                bid_edges,
                key=lambda e: (
                    e.head_pathcost
                    + e.elec_used_kwh * trial_price_kwh # Cost of elec used at trial price
                    + (e.cost - e.elec_used_kwh * forecast_price_kwh) # Cost of penalties
                ),
            )
            best_quantity = max(0, best_edge.elec_used_kwh)

            if not pq_pairs or abs(best_quantity - pq_pairs[-1].quantity_kwh) > 0.01:
                pq_pairs.append(
                    PriceQuantityPair(
                        price_mwh=float(trial_price),
                        quantity_kwh=best_quantity,
                    )
                )

        best_at_forecast = min(bid_edges, key=lambda e: e.value)
        initial_node.pathcost = best_at_forecast.value
        initial_node.best_edge = best_at_forecast
        self.initial_node = initial_node

        self.logger.info(f"Done ({len(pq_pairs)} PQ pairs found).")
        return pq_pairs

    def best_edge_at_price(self, price_mwh: float) -> Edge[S, A]:
        """The best hour-0 edge at clearing price, also set as ``initial_node.best_edge``."""
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
                e.head_pathcost
                + e.elec_used_kwh * trial_price_kwh
                + (e.cost - e.elec_used_kwh * forecast_price_kwh)
            ),
        )
        initial_node.best_edge = best_edge
        return best_edge
