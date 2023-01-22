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


def fillDict(path="../Results/erdos_renyi_graphs/"):
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


def generate_graphs(path="../Results/erdos_renyi_graphs/"):
    keys = list(dfile.keys())

    def create(n, p):
        graph = GraphCreator.generateErdosRenyiGraph(n, p)
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

    for n in range(600, 400, -1):
        print(" --- Count nodes:", n)
        Parallel(n_jobs=11)(delayed(create)(n, 0.01*mp)
                            for mp in range(101))


def simulation_zch():

    count = 1000
    num_proc = 12
    step = int(count/num_proc)

    cd = np.arange(2, 8)
    cc = np.arange(2, 8)
    res_shape = (len(cd), len(cc))
    u = 1e4
    p = 0.01

    fname = f"../Results/ErdosRenyiResults/c_from_{cc[0]}_to_{cc[-1]}_d_from_{cd[0]}_to_{cd[-1]}_u={u}_p={p}.nc"
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

            dj_lengths = [r[0] for res_chunk in sim_results for r in res_chunk]
            gr_lengths = [r[1] for res_chunk in sim_results for r in res_chunk]

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


def checkExistedGraphs(c_max, d_max):
    keys = list(dfile.keys())
    for c in range(2, c_max):
        for d in range(2, d_max):
            if keys.count((c, d)) == 0:
                print(f" --- Graph does not exist: c = {c}, d = {d}")


def simulation_zup():
    rc, rd = 7, 7
    graph = findGraph(rc, rd)

    num_proc = 10
    count = 1000  # [41, 44] sec
    step = int(count/num_proc)

    # a_u = np.array([1e5, 1e4], dtype=np.float64)
    # a_p = np.array([0.2, 0.3], dtype=np.float64)
    a_u = np.array([1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6,
                   5e6, 1e7, 5e7, 1e8, 5e8, 1e9], dtype=np.float64)
    # a_u = np.array([1e2, 1e3, 1e4, 1e5, 1e6,
    #                 1e7, 1e8, 1e9], dtype=np.float64)
    a_p = np.arange(0, 0.4, 0.01, dtype=np.float64)

    res_shape = (len(a_u), len(a_p))

    fname = f"../Results/ErdosRenyiResults/u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}_c={rc}_D={rd}.nc"
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

    print("C, D:", GraphCreator.extractAvareageDegree(graph), rd)
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
            sim_results = Parallel(n_jobs=num_proc)(delayed(Simulator.sim_wrap)(arg)
                                                    for arg in args)

            dj_lengths = [r[0] for res_chunk in sim_results for r in res_chunk]
            gr_lengths = [r[1] for res_chunk in sim_results for r in res_chunk]

            dj_mfpt = sum(dj_lengths)/count
            gr_mfpt = sum(gr_lengths)/count
            a_z[i][j] = np.log10(gr_mfpt / dj_mfpt)

            save()
            print(" --- Result")
            print(" --- u:", u)
            print(" --- p:", p)
            print(' --- Ln (z):', a_z[i][j])
            t.stop()

            print('')


def main():

    # generate_graphs()

    # checkExistedGraphs(10, 10)

    # simulation_zup()
    # simulation_zch()

    fname = "u_from_10.0_to_1000000000.0_p_from_0.0_to_0.39_c=7_D=7.nc"
    Visualizer.showRegularResult(
        f"../Results/ErdosRenyiResults/{fname}", field="ln10_z", xticks=[0, 0.1, 0.2, 0.3])
    Visualizer.add_critical_u(3.5, 0.39)
    plt.show()

    # graph = findGraph(7, 7)
    # graph = GraphCreator.generateErdosRenyiGraph(1000, 1.)
    # Visualizer.draw_hist(graph, mrange=(1, 1000))
    # plt.show()
    # Visualizer.showGraph(graph, size_node=1, size_edge=0.1, layout="spring")


if __name__ == '__main__':
    main()
