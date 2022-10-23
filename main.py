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


def sim_wrap(wrap):
    count, igraph, p, u = wrap
    res = []
    for _ in range(count):
        source, target = Simulator.choose_source_target(igraph)
        unc = UncertaintyCond(p, u)
        ss = SimulationSettings(unc, need_print=False,
                                source=source, target=target)
        simulator = Simulator(igraph, ss)
        res.append(simulator.run())
    return res


def sim(igraph, p, u):
    source, target = Simulator.choose_source_target(igraph)
    unc = UncertaintyCond(p, u)
    ss = SimulationSettings(unc, need_print=False,
                            source=source, target=target)
    simulator = Simulator(igraph, ss)
    return simulator.run()


def genarate_grid(dim, diameter, periodic, path="../Results/grid_graphs/"):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for file in onlyfiles:
        fdim, fdiam, fn, rc, rd = [int(p.split("=")[1])
                                   for p in file.split("_")]
        if fdim == dim and fdiam == diameter:
            return rc, rd, nx.read_pajek(path + file)

    dims = diameter*np.ones(shape=(dim,), dtype=int)
    gr = nx.grid_graph(dim=list(dims), periodic=periodic)

    print(' --- Calculate: connectivity, diameter')
    adc = nx.average_degree_connectivity(gr)
    rc = int(np.sum(list(adc.values())))
    rd = nx.diameter(gr)

    nx.write_pajek(
        gr, path + f"dim={dim}_diam={diameter}_n={nx.number_of_nodes(gr)}_rc={rc}_rd={rd}")
    return rc, rd, gr


def find_regular_graph(c, d, new_graph=False, path="./regular_graphs/"):

    if not new_graph:
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for file in onlyfiles:
            sfc, sfd, sfn = file.split("_")
            fc, fd, fn = [int(p.split("=")[1]) for p in [sfc, sfd, sfn]]
            if fc == c and fd == d:
                return nx.read_pajek(path + file)
    print(f"Start find graph c={c}, d={d}")
    rg = generate_regular_graph2(c, d)
    nx.write_pajek(
        rg, path + f"c={c}_d={d}_n={nx.number_of_nodes(rg)}")
    print(f"Graph created c={c}, d={d}")
    print(f"")
    return rg


def generate_small_world(count_nodes, count_neighbours):
    igraph = nx.watts_strogatz_graph(n=count_nodes, k=count_neighbours, p=0.5)
    return igraph


def simulation_on_regular_graph_zch():
    a_diams = np.arange(2, stop=9, dtype=int)
    a_connect = np.arange(2, stop=9, dtype=int)

    res_shape = (len(a_diams), len(a_connect))
    mfpt_dj = np.zeros(shape=res_shape)
    mfpt_gr = np.zeros(shape=res_shape)

    count = 1000
    u = 4e2
    p = 0.2

    ar_p = p*np.ones(shape=res_shape)
    ar_u = u*np.ones(shape=res_shape)

    for i, diam in enumerate(a_diams):
        for j, connect in enumerate(a_connect):

            print(' --- Create graph')
            igraph = find_regular_graph(connect, diam)

            print(' --- Simulation start')
            sim_results = Parallel(n_jobs=12)(delayed(sim)(igraph, p, u)
                                              for i in range(count))

            dj_lengths = [r[0] for r in sim_results]
            gr_lengths = [r[1] for r in sim_results]

            dj_mfpt = sum(dj_lengths)/count
            gr_mfpt = sum(gr_lengths)/count

            mfpt_dj[i, j] = dj_mfpt
            mfpt_gr[i, j] = gr_mfpt

            print(" --- Result")
            print(' --- Probability(p):', p)
            print(' --- Cost(u):', u)
            print(' --- Diameter(diam):', diam)
            print(' --- Connectivity(connect):', connect)
            print(' --- MFPT Dijkstra:', dj_mfpt)
            print(' --- MFPT Greedy:', gr_mfpt)
            print('')

    coord_names = ["Diameter", "Degree"]

    df = xr.Dataset({
        "up": (coord_names, ar_p),
        "uv": (coord_names, ar_u),
        "dj_mfpt": (coord_names, mfpt_dj),
        "gr_mfpt": (coord_names, mfpt_gr),
        "lnz": (coord_names, np.log10(np.divide(mfpt_gr, mfpt_dj))),
    },
        coords={
        coord_names[0]: a_diams,
        coord_names[1]: a_connect
    })

    df.to_netcdf(
        f"Regular_graph_u={u}_p={p}_max-diam={a_diams[-1]}_max-degree={a_connect[-1]}.nc")

    df.lnz.plot()
    plt.show()


def simulation_on_regular_graph_zup():
    c, H = 6, 6
    count = 1000
    igraph = find_regular_graph(c, H)

    a_u = np.array([1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3,
                    1e4, 5e4, 1e5, 5e5, 1e6, 5e6], dtype=np.float64)
    a_p = np.arange(0, 0.41, 0.01, dtype=np.float64)

    res_shape = (len(a_u), len(a_p))
    mfpt_dj = np.zeros(shape=res_shape)
    mfpt_gr = np.zeros(shape=res_shape)
    a_c = c*np.ones(shape=res_shape)
    a_h = H*np.ones(shape=res_shape)

    for i, u in enumerate(a_u):
        for j, p in enumerate(a_p):
            sim_results = Parallel(n_jobs=15)(delayed(sim)(igraph, p, u)
                                              for i in range(count))
            dj_lengths = [r[0] for r in sim_results]
            gr_lengths = [r[1] for r in sim_results]

            dj_mfpt = sum(dj_lengths)/count
            gr_mfpt = sum(gr_lengths)/count
            mfpt_dj[i, j] = dj_mfpt
            mfpt_gr[i, j] = gr_mfpt
            print("_____Result______")
            print('Probability(p):', p)
            print('Cost(u):', u)
            print('MFPT Dijkstra:', dj_mfpt)
            print('MFPT Greedy:', gr_mfpt)
            print('ln (z):', np.log10(gr_mfpt/dj_mfpt))

    coord_names = ["log10(u)", "p"]

    df = xr.Dataset({
        "c": (coord_names, a_c),
        "Diameter": (coord_names, a_h),
        "dj_mfpt": (coord_names, mfpt_dj),
        "gr_mfpt": (coord_names, mfpt_gr),
        "lnz": (coord_names, np.log10(np.divide(mfpt_gr, mfpt_dj))),
    },
        coords={
        coord_names[0]: np.log10(a_u),
        coord_names[1]: a_p
    })

    df.to_netcdf(
        f"./RegularGraphResults/Regular_graph_u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}_c={c}_D={H}.nc")


def simulation_on_grid_graph_zch():
    count = 1000

    a_diameters = np.arange(2, stop=7, dtype=int)
    a_dimensions = np.arange(2, stop=7, dtype=int)

    res_shape = (len(a_diameters), len(a_dimensions))
    mfpt_dj = np.zeros(shape=res_shape)
    mfpt_gr = np.zeros(shape=res_shape)
    real_diameters = np.zeros(shape=res_shape, dtype=int)
    real_coonectivity = np.zeros(shape=res_shape, dtype=int)

    u = 4e2
    p = 0.2
    periodic = True

    ar_p = p*np.ones(shape=res_shape)
    ar_u = u*np.ones(shape=res_shape)

    for i, diam in enumerate(a_diameters):
        for j, dim in enumerate(a_dimensions):
            print(' --- Generate graph')
            print(f" --- Diameter: {diam}")
            print(f" --- Dimension: {dim}")
            rc, rd, igraph = genarate_grid(dim, diam, periodic)

            print(' --- Simulation start')
            sim_results = Parallel(n_jobs=12)(delayed(sim)(igraph, p, u)
                                              for i in range(count))

            dj_lengths = [r[0] for r in sim_results]
            gr_lengths = [r[1] for r in sim_results]

            dj_mfpt = sum(dj_lengths)/count
            gr_mfpt = sum(gr_lengths)/count

            mfpt_dj[i, j] = dj_mfpt
            mfpt_gr[i, j] = gr_mfpt
            real_diameters[i, j] = rd
            real_coonectivity[i, j] = rc

            print(" --- Result")
            print(' --- Probability(p):', p)
            print(' --- Cost(u):', u)
            print(' --- Diameter:', rd)
            print(' --- Connectivity:', rc)
            print(' --- MFPT Dijkstra:', dj_mfpt)
            print(' --- MFPT Greedy:', gr_mfpt)
            print('')

    coord_names = ["diameter", "dimension"]

    df = xr.Dataset({
        "up": (coord_names, ar_p),
        "uv": (coord_names, ar_u),
        "rdiameter": (coord_names, real_diameters),
        "rcoonectivity": (coord_names, real_coonectivity),
        "gr_mfpt": (coord_names, mfpt_gr),
        "dj_mfpt": (coord_names, mfpt_dj),
        "lnz": (coord_names, np.log10(np.divide(mfpt_gr, mfpt_dj))),
    },
        coords={
        "diameter": a_diameters,
        "dimension": a_dimensions,
    })

    df.to_netcdf(
        f"../Results/GridGraphResults/grid_u={u}_p={p}_max-dims={a_dimensions[-1]}_max-diams={a_diameters[-1]}_periodic={periodic}.nc")

    df.lnz.plot()
    plt.show()


def simulation_on_grid_graph_zup():
    count = 1000
    C, D, igraph = genarate_grid(5, 5, True)

    a_u = np.array([1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3,
                    1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7], dtype=np.float64)
    # a_p = np.arange(0, 0.41, 0.01, dtype=np.float64)

    # a_u = np.array([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=np.float64)
    a_p = np.arange(0, 0.71, 0.01, dtype=np.float64)

    res_shape = (len(a_u), len(a_p))
    mfpt_dj = np.zeros(shape=res_shape)
    mfpt_gr = np.zeros(shape=res_shape)
    print("C, D:", C, D)
    for i, u in enumerate(a_u):
        for j, p in enumerate(a_p):
            sim_results = Parallel(n_jobs=12)(delayed(sim)(igraph, p, u)
                                              for _ in range(count))
            dj_lengths = [r[0] for r in sim_results]
            gr_lengths = [r[1] for r in sim_results]

            dj_mfpt = sum(dj_lengths)/count
            gr_mfpt = sum(gr_lengths)/count
            mfpt_dj[i, j] = dj_mfpt
            mfpt_gr[i, j] = gr_mfpt
            print(" --- Result")
            print(" --- u:", u)
            print(" --- p:", p)
            print(' --- MFPT Dijkstra:', dj_mfpt)
            print(' --- MFPT Greedy:', gr_mfpt)
            print(' --- Ln (z):', np.log10(gr_mfpt/dj_mfpt))
            print('')

    coord_names = ["log10(u)", "p"]

    df = xr.Dataset({
        "dj_mfpt": (coord_names, mfpt_dj),
        "gr_mfpt": (coord_names, mfpt_gr),
        "lnz": (coord_names, np.log10(np.divide(mfpt_gr, mfpt_dj))),
    },
        coords={
        coord_names[0]: np.log10(a_u),
        coord_names[1]: a_p
    })

    df.to_netcdf(
        f"../Results/GridGraphResults/grid_u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}_c={C}_D={D}.nc")

    df.lnz.plot()
    plt.show()


def simulation_on_small_world_graph_zch():  # not checked
    count = 1000

    a_nodes = np.arange(5, stop=500, step=5, dtype=int)
    a_neigh = np.arange(2, stop=20, step=1, dtype=int)

    res_shape = (len(a_nodes), len(a_neigh))
    mfpt_dj = np.zeros(shape=res_shape)
    mfpt_gr = np.zeros(shape=res_shape)
    real_diameters = np.zeros(shape=res_shape)
    real_coonectivity = np.zeros(shape=res_shape)

    u = 1e3
    p = 0.1
    periodic = False

    ar_p = p*np.ones(shape=res_shape)
    ar_u = u*np.ones(shape=res_shape)

    for i, cnodes in enumerate(a_nodes):
        for j, cneigh in enumerate(a_neigh):
            if cnodes < cneigh:
                continue
            print(f"try cnodes={cnodes}, cneigh={cneigh}")
            ncc = 2
            while ncc != 1:
                igraph = generate_small_world(cnodes, cneigh)
                ncc = nx.number_connected_components(igraph)

            real_connect = nx.average_degree_connectivity(igraph)
            real_diameter = nx.diameter(igraph)
            av_connect = sum(
                list(real_connect.values()))/len(real_connect)

            sim_results = Parallel(n_jobs=15)(delayed(sim)(igraph, p, u)
                                              for i in range(count))

            dj_lengths = [r[0] for r in sim_results]
            gr_lengths = [r[1] for r in sim_results]

            dj_mfpt = sum(dj_lengths)/count
            gr_mfpt = sum(gr_lengths)/count

            mfpt_dj[i, j] = dj_mfpt
            mfpt_gr[i, j] = gr_mfpt
            real_diameters[i, j] = real_diameter
            real_coonectivity[i, j] = av_connect

            print("_____Result______")
            print('Probability(p):', p)
            print('Cost(u):', u)
            print('Nodes:', cnodes)
            print('Neighbors:', cneigh)
            print('Real Diameter:', real_diameter)
            print('Real Connectivity:', av_connect)
            print('MFPT Dijkstra:', dj_mfpt)
            print('MFPT Greedy:', gr_mfpt)

    coord_names = ["cnodes", "cneigh"]

    df = xr.Dataset({
        "up": (coord_names, ar_p),
        "uv": (coord_names, ar_u),
        "real_diameters": (coord_names, real_diameters),
        "real_coonectivity": (coord_names, real_coonectivity),
        "gr_mfpt": (coord_names, mfpt_gr),
        "gr_mfpt": (coord_names, mfpt_gr),
        "lnz": (coord_names, np.log10(np.divide(mfpt_gr, mfpt_dj))),
    },
        coords={
        "cnodes": a_nodes,
        "cneigh": a_neigh,
    })

    df.to_netcdf(
        f"Small-world(Regular-Lattice)_u={u}_p={p}_max-count-nodes={a_nodes[-1]}_max-count-neighbors={a_neigh[-1]}.nc")

    df.lnz.plot()
    plt.show()


def simulation_on_small_world_graph_zup():  # not checked
    count = 1000
    graph = GraphCreator.generateSmallWorld(5000, 10, 0.5)

    adc = nx.average_degree_connectivity(graph)
    rc = int(np.sum(list(adc.values())))
    # rd = nx.diameter(graph)

    a_u = np.array([1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3,
                    1e4, 5e4, 1e5, 5e5, 1e6, 5e6], dtype=np.float64)
    a_p = np.arange(0, 0.71, 0.01, dtype=np.float64)

    res_shape = (len(a_u), len(a_p))
    mfpt_dj = np.zeros(shape=res_shape)
    mfpt_gr = np.zeros(shape=res_shape)
    print("C, D:", rc, rd)
    for i, u in enumerate(a_u):
        for j, p in enumerate(a_p):
            sim_results = Parallel(n_jobs=15)(delayed(sim)(graph, p, u)
                                              for i in range(count))
            dj_lengths = [r[0] for r in sim_results]
            gr_lengths = [r[1] for r in sim_results]

            dj_mfpt = sum(dj_lengths)/count
            gr_mfpt = sum(gr_lengths)/count
            mfpt_dj[i, j] = dj_mfpt
            mfpt_gr[i, j] = gr_mfpt
            print(" --- Result")
            print(" --- u:", u)
            print(" --- p:", p)
            print(' --- MFPT Dijkstra:', dj_mfpt)
            print(' --- MFPT Greedy:', gr_mfpt)
            print(' --- Ln (z):', np.log10(gr_mfpt/dj_mfpt))
            print('')

    coord_names = ["log10(u)", "p"]

    df = xr.Dataset({
        "dj_mfpt": (coord_names, mfpt_dj),
        "gr_mfpt": (coord_names, mfpt_gr),
        "lnz": (coord_names, np.log10(np.divide(mfpt_gr, mfpt_dj))),
    },
        coords={
        coord_names[0]: np.log10(a_u),
        coord_names[1]: a_p
    })

    df.to_netcdf(
        f"../Results/SmallWorldGraphResults/u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}_c={rc}_D={rd}.nc")

    df.lnz.plot()
    plt.show()


def simulation_scale_free_graph_zup():

    graph = GraphCreator.generateScaleFreeGraph(1e3, 0.3, 0.3, 0.4, 1, 1)
    rc = GraphCreator.extractAvareageDegree(graph)
    rd = nx.diameter(graph)

    count = 300
    a_u = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8], dtype=np.float64)
    a_p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
    # a_u = np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8,
    #                1e9, 5e9, 1e10, 5e10, 1e11, 5e11, 1e12, 5e12], dtype=np.float64)
    # a_p = np.arange(0, 0.71, 0.005, dtype=np.float64)

    res_shape = (len(a_u), len(a_p))

    fname = f"../Results/ScaleFreeGraphResults/free_scale_u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}_c={rc}_D={rd}.nc"
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
        df.to_netcdf(
            f"../Results/ScaleFreeGraphResults/free_scale_u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}_c={rc}_D={rd}.nc")

    print("C, D:", rc, rd)
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
            sim_results = Parallel(n_jobs=16)(delayed(sim)(graph, p, u)
                                              for i in range(count))
            dj_lengths = [r[0] for r in sim_results]
            gr_lengths = [r[1] for r in sim_results]

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


def simulation_scale_free_graph_zch():
    a_diams = np.arange(3, stop=9, dtype=int)
    a_connect = np.arange(3, stop=9, dtype=int)

    res_shape = (len(a_diams), len(a_connect))
    mfpt_dj = np.zeros(shape=res_shape)
    mfpt_gr = np.zeros(shape=res_shape)

    count = 1000
    u = 1e6
    p = 0.08
    num_proc = 10
    step = int(count/num_proc)

    fname = f"../Results/ScaleFreeGraphResults/sfgr_d_from_{a_diams[0]}_to_{a_diams[-1]}_c_from_{a_connect[0]}_to_{a_connect[-1]}_u={u}_p={p}.nc"
    a_z = np.empty(shape=res_shape)
    a_z[:] = np.nan

    if os.path.isfile(fname):
        df = xr.load_dataset(fname)
        a_z = df.ln10_z.to_numpy()

    def save():
        coord_names = ["D", "c"]
        df = xr.Dataset({
            "ln10_z": (coord_names, a_z),
        },
            coords={
            coord_names[0]: a_diams,
            coord_names[1]: a_connect,
        })
        df.to_netcdf(fname)

    for i, diam in enumerate(a_diams):
        for j, connect in enumerate(a_connect):
            if ~np.isnan(a_z[i][j]):
                print(" --- Skip")
                print(" --- c:", connect)
                print(" --- d:", diam)
                print(' --- Ln (z):', a_z[i][j])
                print('')
                continue

            t = Timer()
            t.start()

            print(' --- Create graph')
            _, _, graph = GraphCreator.getScaleFreeGraph(connect, diam)

            print(' --- Simulation start')
            args = [(step, graph.copy(), p, u) for i in range(num_proc)]
            sim_results = Parallel(n_jobs=num_proc)(delayed(sim_wrap)(arg)
                                                    for arg in args)

            dj_lengths = [r[0] for res_chunk in sim_results for r in res_chunk]
            gr_lengths = [r[1] for res_chunk in sim_results for r in res_chunk]

            dj_mfpt = sum(dj_lengths)/count
            gr_mfpt = sum(gr_lengths)/count
            a_z[i][j] = np.log10(gr_mfpt / dj_mfpt)

            save()
            print(" --- Result")
            print(" --- c:", connect)
            print(" --- d:", diam)
            print(' --- Ln (z):', a_z[i][j])
            t.stop()


def simulation_on_facebook_graph_zup():

    # real_diameter=8
    # av_connect=117.63000341148297
    graph = GraphCreator.getFacebook(
        "../../dataset/ego_facebook/ego_facebook.csv")

    rc = 117.63000341148297
    rd = 8

    num_proc = 12
    count = 300  # [41, 44] sec
    step = int(count/num_proc)

    # a_u = np.array([1e5, 1e4], dtype=np.float64)
    # a_p = np.array([0.2, 0.3], dtype=np.float64)
    a_u = np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6,
                   5e6, 1e7, 5e7, 1e8, 5e8, 1e9], dtype=np.float64)
    a_p = np.arange(0, 0.405, 0.005, dtype=np.float64)

    res_shape = (len(a_u), len(a_p))

    fname = f"../Results/FacebookGraphResults/ego_facebook_u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}_c={rc}_D={rd}.nc"
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
        df.to_netcdf(
            f"../Results/FacebookGraphResults/ego_facebook_u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}_c={rc}_D={rd}.nc")

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
            sim_results = Parallel(n_jobs=num_proc)(delayed(sim_wrap)(arg)
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
    # simulation_on_regular_graph_zch()
    # simulation_on_regular_graph_zup()

    # simulation_on_grid_graph_zch()
    # simulation_on_grid_graph_zup()

    # simulation_on_small_world_graph_zup()

    # simulation_on_small_world_graph_zup()

    simulation_scale_free_graph_zup()
    # simulation_scale_free_graph_zch()

    # simulation_on_facebook_graph_zup()

    # print("t")

    # Tasks
    # 1) structures graphs (grid 2d,3d,4d,...)
    # 2) check for graph E.R. or
    # 3)


if __name__ == '__main__':
    main()
