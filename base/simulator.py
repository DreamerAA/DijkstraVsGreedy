import networkx as nx
from dataclasses import dataclass
import numpy as np
import random
from networkx.drawing.nx_pydot import graphviz_layout
from dataclasses import dataclass, field
from typing import List
import xarray as xr
import os.path
from joblib import Parallel, delayed

from utils.timer import Timer


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
    need_save_path: bool = False
    


class SimulationRunnerSettings:
    greedy_mfpt_integral: bool = False



class Simulator:
    def __init__(self, igraph, ss: SimulationSettings, srs:SimulationRunnerSettings) -> None:
        self.settings = ss
        self.igraph = igraph
        self.sim_run_settings = srs
        pass

    def sim_wrap(wrap, srs:SimulationRunnerSettings):
        count, igraph, p, u = wrap
        res = []
        for _ in range(count):
            source, target = Simulator.choose_source_target(igraph)
            unc = UncertaintyCond(p, u)
            ss = SimulationSettings(unc, need_print=False,
                                    source=source, target=target)
            simulator = Simulator(igraph, ss, srs)
            res.append(simulator.run())
        return res

    def sim(igraph: nx.Graph, unc: UncertaintyCond):
        source, target = Simulator.choose_source_target(igraph)
        ss = SimulationSettings(unc, need_print=False,
                                source=source, target=target)
        simulator = Simulator(igraph, ss)
        return simulator.run()

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
        # path = [n_curr]
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
                # if ss.need_save_path:
                # path.append(n_curr)
                length = length + min_weight

        return GraphPath(ss.source, ss.target, [], length)
        # return GraphPath(ss.source, ss.target, path, length)

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

        return dj_path.length, gr_path.length

    def simulation_up(graph, a_u, a_p, fname, run_count:int=100, srs:SimulationRunnerSettings = SimulationRunnerSettings()):
        num_proc = 10
        step = int(run_count/num_proc)

        res_shape = (len(a_u), len(a_p))
        a_z = np.empty(shape=res_shape)
        a_z[:] = np.nan

        if os.path.isfile(fname):
            df = xr.load_dataset(fname)
            a_z = df.ln10_z.to_numpy()

        def save():
            coord_names = ["log10(u)", "p"]
            df = xr.Dataset({
                "ln10_z": (coord_names, a_z),
            },
                coords={
                coord_names[0]: np.log10(a_u),
                coord_names[1]: a_p,
            })
            df.to_netcdf(fname)

        for i, u in enumerate(a_u):
            for j, p in enumerate(a_p):
                if ~np.isnan(a_z[i][j]):
                    print(" --- Skip")
                    print(" --- u:", u)
                    print(" --- p:", p)
                    print(' --- Ln (z):', a_z[i][j])
                    print('')
                    continue

                t = Timer()
                t.start()

                args = [(step, graph.copy(), p, u) for i in range(num_proc)]
                sim_results = Parallel(n_jobs=num_proc)(delayed(Simulator.sim_wrap)(arg, srs)
                                                        for arg in args)

                dj_lengths = [r[0]
                              for res_chunk in sim_results for r in res_chunk]
                gr_lengths = [r[1]
                              for res_chunk in sim_results for r in res_chunk]

                dj_mfpt = sum(dj_lengths)/run_count
                gr_mfpt = sum(gr_lengths)/run_count
                a_z[i][j] = np.log10(gr_mfpt / dj_mfpt)

                save()
                print(" --- Result")
                print(" --- u:", u)
                print(" --- p:", p)
                print(' --- Ln (z):', a_z[i][j])
                t.stop()

                print('')

    def simulation_ch(cd, cc, u, p, findGraph, fname):
        count = 1000
        num_proc = 12
        step = int(count/num_proc)

        res_shape = (len(cd), len(cc))
        a_z = np.empty(shape=res_shape)
        a_z[:] = np.nan

        if os.path.isfile(fname):
            df = xr.load_dataset(fname)
            a_z = df.ln10_z.to_numpy()

        def save():
            coord_names = ["Degree", "Diameter"]
            df = xr.Dataset({
                "ln10_z": (coord_names, a_z),
            },
                coords={
                coord_names[0]: cc,
                coord_names[1]: cd,
            })
            df.to_netcdf(fname)

        for i, c in enumerate(cc):
            for j, d in enumerate(cd):

                if ~np.isnan(a_z[i][j]):
                    print(" --- Skip")
                    print(" --- c:", c)
                    print(" --- d:", d)
                    print(' --- Ln (z):', a_z[i][j])
                    print('')
                    continue

                t = Timer()
                t.start()

                graph = findGraph(c, d)
                if graph is None:
                    continue

                args = [(step, graph.copy(), p, u) for i in range(num_proc)]
                sim_results = Parallel(n_jobs=num_proc)(delayed(Simulator.sim_wrap)(arg)
                                                        for arg in args)

                dj_lengths = [r[0]
                              for res_chunk in sim_results for r in res_chunk]
                gr_lengths = [r[1]
                              for res_chunk in sim_results for r in res_chunk]

                dj_mfpt = sum(dj_lengths)/count
                gr_mfpt = sum(gr_lengths)/count
                a_z[i][j] = np.log10(gr_mfpt / dj_mfpt)

                save()
                print(" --- Result")
                print(" --- c:", c)
                print(" --- d:", d)
                print(' --- Ln (z):', a_z[i][j])
                t.stop()

                print('')
