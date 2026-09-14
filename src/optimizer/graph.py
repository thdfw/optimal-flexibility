import time
from typing import Generic

from assets.base import A, Asset, P, S
from optimizer.transitions_table import get_transitions_table


class Node(Generic[S]):
    def __init__(self, state: S, time_step: int):
        self.state = state
        self.time_step = time_step
        self.pathcost = 1e9
        self.next_node: Node[S] | None = None

    def __repr__(self):
        return f"[{self.time_step}]{self.state}"


class Edge(Generic[S, A]):
    def __init__(self, tail: Node[S], head: Node[S], cost: float, action: A):
        self.tail: Node[S] = tail
        self.head: Node[S] = head
        self.cost = cost
        self.action = action

    def __repr__(self):
        return f"Edge[{self.tail} --{round(self.cost, 3)}--> {self.head}]"


class Graph(Generic[S, A, P]):
    def __init__(self, asset: Asset[S, A, P]):
        self.asset = asset
        self.params = asset.params
        self.N = asset.params.horizon
        self.transitions = get_transitions_table(asset)
        self._time_and_log(self.create_nodes, "nodes")
        self._time_and_log(self.create_edges, "edges")
        self._time_and_log(self.find_shortest_path, "shortest_path")

    def _time_and_log(self, func, step: str) -> None:
        start = time.perf_counter()
        func()
        elapsed = round(time.perf_counter() - start, 1)
        if step == "nodes":
            print(f"Created nodes in {elapsed} seconds")
        elif step == "edges":
            print(f"Created edges in {elapsed} seconds")
        elif step == "shortest_path":
            print(f"Found shortest path in {elapsed} seconds")

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

        for time_step in range(self.N):
            print(f"Building edges for time step {time_step}...")
            for node in self.nodes[time_step]:
                self.edges[node] = []

                available_actions = self.asset.get_available_actions(node.state, time_step)

                for action in available_actions:
                    next_state = self.transitions[(node.state, action)]
                    if not self.asset.allow_transition(node.state, next_state, action, time_step):
                        continue
                    next_node = self.nodes_by[time_step + 1][next_state]
                    cost = self.asset.cost(node.state, next_state, action, time_step)
                    self.edges[node].append(Edge(node, next_node, cost, action))

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
