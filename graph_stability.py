from graphviz import Graph
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed
import xarray as xr
from os import listdir
from os.path import isfile, join
from GraphCreator import GraphCreator
from Simulator import Simulator, SimulationSettings, UncertaintyCond
import pandas as pd
from Timer import Timer
import os.path
from multiprocessing import Process, Pool
from dataclasses import dataclass
from typing import List, Union, Callable
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


@dataclass
class Regular:
    degree: int
    count_nodes: int


@dataclass
class SmallWorld:
    count_nodes: int
    count_neighbours: int
    probability: float


@dataclass
class DirectedScaleFree:
    count_nodes: int
    alpha: float
    beta: float
    gamma: float


@dataclass
class BarabasiAlbert:
    count_nodes: int
    count_edges: int


@dataclass
class ErdosRenyi:
    count_nodes: int
    probability: float


@dataclass
class GeneratorDTO:
    info: List[Union[Regular, SmallWorld,
                     DirectedScaleFree, BarabasiAlbert, ErdosRenyi]] = None
    generator: Callable = None
    color: str = "r"


@dataclass
class Task:
    generators: List[GeneratorDTO]
    unc: UncertaintyCond
    count_process: int = 10


@dataclass
class GraphInfo:
    graph: nx.Graph = None
    degree: int = 0
    diameter: int = 0
    count_nodes: int = 0


def generate_regular_graph(regular: Regular) -> GraphInfo:
    graph = nx.random_regular_graph(regular.degree, regular.count_nodes)
    diameter = nx.diameter(graph)
    return GraphInfo(graph, regular.degree,
                     diameter, regular.count_nodes)


def generate_small_world_graph(sw: SmallWorld):
    graph = nx.watts_strogatz_graph(
        n=sw.count_nodes, k=sw.count_neighbours, p=sw.probability)
    degree = GraphCreator.extractAvareageDegree(graph)
    diameter = nx.diameter(graph)
    return GraphInfo(graph.copy(), degree,
                     diameter, sw.count_nodes)


def generate_directed_scale_free_graph(sf: DirectedScaleFree):
    graph = nx.scale_free_graph(
        sf.count_nodes,
        alpha=sf.alpha,
        beta=sf.beta,
        gamma=sf.gamma,
        delta_in=1,
        delta_out=1)

    graph = nx.Graph(graph)

    if not GraphCreator.connected(graph):
        print(' --- Skip graph (a, b, g):', sf.alpha,
              sf.beta, sf.gamma)
        return None

    degree = GraphCreator.extractAvareageDegree(graph)
    diameter = nx.diameter(graph)
    return GraphInfo(graph.copy(), degree,
                     diameter, sf.count_nodes)


def generate_barabasi_graph(ab: BarabasiAlbert):
    graph = nx.barabasi_albert_graph(
        ab.count_nodes, ab.count_edges)
    graph = nx.Graph(graph)
    degree = GraphCreator.extractAvareageDegree(graph)

    if not GraphCreator.connected(graph):
        print(' --- Skip graph (ba):', ab.count_nodes, ab.count_edges)
        return None

    diameter = nx.diameter(graph)
    print(' --- Barabasi: ', ab.count_nodes, ab.count_edges)
    print(' --- Barabasi results: ', degree, diameter)
    return GraphInfo(graph.copy(), degree,
                     diameter, ab.count_nodes)


def generate_erdos_renyi_graph(er: ErdosRenyi) -> GraphInfo:
    graph = nx.erdos_renyi_graph(er.count_nodes, er.probability)
    degree = GraphCreator.extractAvareageDegree(graph)

    if not GraphCreator.connected(graph):
        print(' --- Skip graph (er):', er.count_nodes, er.probability)
        return None

    diameter = nx.diameter(graph)
    return GraphInfo(graph.copy(), degree,
                     diameter, er.count_nodes)


# 0 - Dijkstra, 1 - Greedy
def extractLnZ(sim_result):
    return np.log10(sim_result[1]/sim_result[0])


def plot(info, color):
    x = [p.degree for p in info]
    y = [p.diameter for p in info]
    plt.scatter(x, y, color=color, s=100)


def main(task: Task):

    fig, ax = plt.subplots()
    full_results = []
    for gen in task.generators:
        infos = []
        for info in gen.info:
            graph_info = gen.generator(info)
            if graph_info is None:
                continue
            infos.append(graph_info)
        full_results.append((infos, gen))

    for info, gen in full_results:
        plot(info, gen.color)
    return full_results


def constant_n_task():
    n = 6**4

    lgrg = []
    for c in range(3, 10):
        lgrg.append(Regular(c, n))
    reg_dto = GeneratorDTO(lgrg, generate_regular_graph, 'r')

    lgsmw = []
    for cn in [4, 6, 8, 10, 12]:
        for prob in [0.1, 0.5, 0.8, 1.]:
            # for prob in [0.1]:
            lgsmw.append(SmallWorld(n, cn, prob))
    sw_dto = GeneratorDTO(lgsmw, generate_small_world_graph, 'g')

    lgsf = []
    for a, b, g in [(0.02, 0.49, 0.49), (0.49, 0.02, 0.49),
                    (0.49, 0.49, 0.02),  (0.9, 0.05, 0.05),
                    (0.05, 0.9, 0.05), (0.05, 0.05, 0.9),
                    (0.33, 0.33, 0.34)]:
        lgsf.append(DirectedScaleFree(n, a, b, g))
    sf_dto = GeneratorDTO(
        lgsf, generate_directed_scale_free_graph, 'blue')

    lgba = []
    for a in range(1, 4):
        for b in range(1, 4):
            lgba.append(BarabasiAlbert(n, (2**a) * (3**b) - 1))
    ab_dto = GeneratorDTO(lgba, generate_barabasi_graph, 'm')

    lger = []
    for p in np.arange(0.1, 1.1, 0.1):
        lger.append(ErdosRenyi(n, p))
    er_dto = GeneratorDTO(lger, generate_erdos_renyi_graph, 'black')

    u = 50000
    p = 0.15
    unc = UncertaintyCond(p, u)
    task = Task([er_dto, reg_dto, sw_dto, sf_dto, ab_dto], unc)

    plt.figure()
    full_results = main(task)
    x = np.arange(2, 1e3, 0.1)
    tmp = 0.5*(u*p+1)/(u*(p**x) + 1)
    y = 2*np.log10(tmp)/np.log10(x)
    plt.plot(x, y, color='black', linewidth=3)
    plt.xscale('log')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # plt.figure()
    # info, gen = full_results[2]
    # plot(info, gen.color)

    x = np.arange(2, 50, 0.1)
    tmp = 0.5*(u*p+1)/(u*(p**x) + 1)
    y = 2*np.log10(tmp)/np.log10(x)
    plt.plot(x, y, color='black', linewidth=3)
    plt.xscale('log')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)


def z(u,p,c,h):
    return 2*(c**h)*(u*(p**c) + 1)/(u*p + 1)

def h(u,p,c):
    return np.log((u*p+1)/(2*(u*p**c + 1)))/np.log(c)

def h_by_n(n,c):
    return np.log(n)/np.log(c)

def c_crit(u,p,n):
    return np.log((u*p + 1)/(2*u*n) - 1/u)/np.log(p)
    

def test_n_increasing_1():
    top_h = 20
    bottom_h = 1

    c = np.arange(2,10,0.01)
    u = 1e9
    p = 0.00001
    plt.plot(c, h(u,p,c), color='black',linewidth=3)
    plt.plot(c, h_by_n(100,c), linestyle='--', color='r', linewidth=4)
    plt.plot(c, h_by_n(1000,c), linestyle='--', color='r', linewidth=4)
    plt.plot(c, h_by_n(100000,c), linestyle='-', color='r', linewidth=4)
    # plt.fill_between(c, top_h, color='#ffffff') 
    # plt.fill_between(c, h(u,p,c), color='#25548a') 
    plt.xlim(2,10)
    plt.ylim(2,10)
    plt.show()


def test_n_increasing_2():
    top_h = 30
    bottom_h = 1

    c = np.arange(2,10,0.01)
    u = 30000
    p = 0.15
    n1 = 150
    n2 = 1e3
    
    print('z1 =',z(u,p,10,4))
    print('z2 =',z(u,p,10,3))

    c1_crit = c_crit(u,p,n1)
    c2_crit = c_crit(u,p,n2)

    i_c1 = np.argmin(np.abs(c - c1_crit))
    i_c2 = np.argmin(np.abs(c - c2_crit))
    
    print(c1_crit, c2_crit)
    print(i_c1, i_c2)

    def plot_wrap(x, n, s):
        plt.plot(x, h_by_n(n,x), linestyle=s, color='r', linewidth=4)

    plt.plot(c, h(u,p,c), color='black',linewidth=3)
    plot_wrap(c[:i_c1], n1, '-')
    plot_wrap(c[i_c1:], n1, '--')

    plot_wrap(c[:i_c2], n2, '-')
    plot_wrap(c[i_c2:], n2, '--')

    plt.plot(c, h_by_n(5e3,c), linestyle='-', color='r', linewidth=4)
    # plt.fill_between(c, top_h, color='#ffffff') 
    # plt.fill_between(c, h(u,p,c), color='#25548a') 
    plt.xlim(2,10)
    plt.ylim(2,5)
    plt.show()

def test_scale_increasing_part1():
    cmax = 25
    c = np.arange(2,cmax,0.01)
    u = 30000
    p = 0.15

    plt.plot(c, h(u,p,c), color='r',linewidth=3)
    plt.xlim(2,cmax)
    plt.ylim(2,5)
    plt.show()

def test_scale_increasing_part2():
    xmax = 10
    a = 4
    x = np.arange(0.5,xmax - 0.5,0.01)
    y = (x/a)**2 + 5
    plt.plot(x, [7]*x.shape[0], color='r',linewidth=3)
    plt.plot(x, y, color='black',linewidth=3)
    plt.xlim(0,xmax)
    plt.ylim(0,15)

    plt.show()

if __name__ == '__main__':

    # constant_n_task()
    # test_n_increasing_1()
    # test_n_increasing_2()
    test_scale_increasing_part1()
    # test_scale_increasing_part2()


