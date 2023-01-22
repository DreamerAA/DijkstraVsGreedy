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
from GraphCreator import GraphCreator
from visualizer import Visualizer


dfile = {}


def fillDict(path="../Results/barabasi_albert_graphs/"):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for file in onlyfiles:
        params = [p.split("=")[1] for p in file.split("_")]
        rc, rd, _ = [int(p) for p in params]
        dfile[(rc, rd)] = path + file


def findGraph(c, d):
    lk = list(dfile.keys())
    t = (c, d)
    if lk.count(t) == 0:
        return None
    return nx.read_pajek(dfile[lk[lk.index(t)]])


fillDict()


def generate_graphs(path="../Results/barabasi_albert_graphs/"):
    keys = list(dfile.keys())

    def create(n, e):
        graph = GraphCreator.generateBarabasiAlbertGraph(n, e)
        if not GraphCreator.connected(graph):
            return
        degree = GraphCreator.extractIntAvareageDegree(graph)
        if degree > 50:
            return

        diameter = nx.diameter(graph)

        if keys.count((degree, diameter)) != 0:
            return
        nx.write_pajek(
            graph, path + f"c={int(degree)}_d={diameter}_n={n}")

    for n in range(600, 2, -1):
        Parallel(n_jobs=11)(delayed(create)(n, e)
                            for e in np.arange(1, n))
        print(" --- Count nodes:", n)


def simulation_zch():
    cd = np.arange(2, 9)
    cc = np.arange(2, 9)
    u = 1e4
    p = 0.01

    fname = f"../Results/BarabasiAlbertResults/c_from_{cc[0]}_to_{cc[-1]}_d_from_{cd[0]}_to_{cd[-1]}_u={u}_p={p}.nc"
    Simulator.simulation_ch(cd, cc, u, p, findGraph, fname)


def checkExistedGraphs(c_max, d_max):
    keys = list(dfile.keys())
    for c in range(2, c_max):
        for d in range(2, d_max):
            if keys.count((c, d)) == 0:
                print(f" --- Graph does not exist: c = {c}, d = {d}")


def simulation_zup():
    rc, rd = 6, 6
    graph = findGraph(rc, rd)

    # a_u = np.array([1e5, 1e4], dtype=np.float64)
    # a_p = np.array([0.2, 0.3], dtype=np.float64)
    a_u = np.array([1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6,
                   5e6, 1e7, 5e7, 1e8, 5e8, 1e9], dtype=np.float64)
    # a_u = np.array([1e2, 1e3, 1e4, 1e5, 1e6,
    #                 1e7, 1e8, 1e9], dtype=np.float64)
    a_p = np.arange(0, 0.41, 0.01, dtype=np.float64)

    fname = f"../Results/BarabasiAlbertResults/u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}_c={rc}_D={rd}.nc"

    print("C, D:", GraphCreator.extractAvareageDegree(graph), rd)
    Simulator.simulation_up(graph, a_u, a_p, fname)


def main():

    # generate_graphs()

    # checkExistedGraphs(10, 10)

    # simulation_zup()
    fname = "u_from_10.0_to_1000000000.0_p_from_0.0_to_0.4_c=6_D=6.nc"
    Visualizer.showRegularResult(
        f"../Results/BarabasiAlbertResults/{fname}", field="ln10_z", xticks=[0, 0.1, 0.2, 0.3, 0.4])
    Visualizer.add_critical_u(3, 0.4)

    # simulation_zch()
    # fname = "c_from_2_to_8_d_from_2_to_8_u=10000.0_p=0.01.nc"
    # Visualizer.showRegularResult(
    #     f"../Results/BarabasiAlbertResults/{fname}", field="ln10_z")

    # graph = findGraph(7, 7)
    # graph = GraphCreator.generateErdosRenyiGraph(1000, 1.)
    # Visualizer.draw_hist(graph, mrange=(1, 1000))
    # Visualizer.showGraph(graph, size_node=1, size_edge=0.1, layout="spring")
    plt.show()


if __name__ == '__main__':
    main()
