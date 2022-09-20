import networkx as nx
from dataclasses import dataclass
import numpy as np
import random
from networkx.drawing.nx_pydot import graphviz_layout
from dataclasses import dataclass, field
from typing import List


@dataclass
class GraphPath:
    source: int
    target: int
    path: List[int]
    length: float


@dataclass
class UncertaintyCond:
    p: float  # probability
    u: float  # additive
    index: np.int32
    counter: np.int32 = np.int32(1e6)

    def __init__(self, p, u):
        self.p = p
        self.u = u
        self.res_u = np.array([])
        self.res_p = np.array([])
        self.reinit()

    def sample(self, size=1):
        if size > self.counter:
            p = UncertaintyCond.sample_p(size)
            u = np.ones(shape=(size,))
            u[p < self.p] = self.u
            return u, p
        if self.index + size > self.counter:
            self.reinit()

        ru, rp = self.res_u[self.index:(self.index+size)],\
            self.res_p[self.index:(self.index+size)]
        self.index += size
        return ru, rp

    def reinit(self):
        self.res_p = UncertaintyCond.sample_p(self.counter)
        self.res_u = np.ones(shape=(self.counter,))
        self.res_u[self.res_p < self.p] = self.u
        self.index = 0

    def sample_p(size):
        return np.random.uniform(size=size)


@dataclass
class SimulationSettings:
    unc: UncertaintyCond
    source: int
    target: int
    need_show: bool = False
    need_print: bool = False


def show_graph(G, gpath: GraphPath = None):
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos)

    if gpath:
        path = gpath.path
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(
            G, pos, nodelist=[path[0]], node_color='b')
        nx.draw_networkx_nodes(
            G, pos, nodelist=[path[-1]], node_color='g')
        nx.draw_networkx_nodes(
            G, pos, nodelist=path[1:-1], node_color='r')
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, edge_color='r', width=2)


class Simulator:
    def __init__(self, igraph, ss: SimulationSettings) -> None:
        self.settings = ss
        self.igraph = igraph
        pass

    def createGraphWithUncertainty(igraph, unc: UncertaintyCond):
        new_graph = nx.Graph()
        weight, p = unc.sample(len(igraph.edges))
        new_edges = [(edge[0], edge[1], weight[i])
                     for i, edge in enumerate(igraph.edges)]
        new_graph.add_nodes_from(igraph)
        new_graph.add_weighted_edges_from(new_edges)

        return new_graph

    def gready_search(self):
        ss = self.settings
        if ss.source == ss.target:
            return [ss.source]
        n_curr = ss.source
        path = [n_curr]
        length = 0
        while n_curr != ss.target:
            adj = self.igraph.adj[n_curr]
            next_nodes = list(adj.keys())
            weights, p = ss.unc.sample(len(next_nodes))
            min_weight = weights.min()

            min_positions = np.where(weights == min_weight)
            n_icurr = random.choice(min_positions[0])
            p_n_curr = next_nodes[n_icurr]

            if len(self.igraph.adj[p_n_curr]) > 1 or p_n_curr == ss.target:
                n_curr = p_n_curr
                path.append(n_curr)
                length = length + min_weight

        return GraphPath(ss.source, ss.target, path, length)

    def dijkstra_search(self):
        ss = self.settings
        dijkstra_path = nx.dijkstra_path(
            self.igraph, source=ss.source, target=ss.target)

        igraph_unc = Simulator.createGraphWithUncertainty(self.igraph, ss.unc)

        length = nx.path_weight(igraph_unc, dijkstra_path, 'weight')

        return GraphPath(ss.source, ss.target, dijkstra_path, length)

    def choose_source_target(igraph):
        node_from, node_to = 0, 0
        while node_from == node_to:
            source_nodes = list(igraph.nodes)
            target_nodes = list(igraph.nodes)
            node_from = random.choice(source_nodes)
            node_to = random.choice(target_nodes)

        return node_from, node_to

    def run(self):
        dj_path = self.dijkstra_search()
        gr_path = self.gready_search()

        if self.settings.need_print:
            print("_____Result______")
            print("Dijkstra path length: ", dj_path.length)
            print("Gready path length: ", gr_path.length)

        if self.settings.need_show:
            show_graph(self.igraph, dj_path)

        return dj_path.length, gr_path.length
