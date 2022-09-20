import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed
import xarray as xr
from os import listdir
from os.path import isfile, join
from Simulator import Simulator, SimulationSettings, UncertaintyCond


def sim(igraph, p, u):
    source, target = Simulator.choose_source_target(igraph)
    unc = UncertaintyCond(p, u)
    ss = SimulationSettings(unc, need_print=False,
                            source=source, target=target)
    simulator = Simulator(igraph, ss)
    return simulator.run()


def genarate_grid(dim, diameter, periodic, path="./grid_graphs/"):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for file in onlyfiles:
        sfc, sfd, sfn = file.split("_")
        fdim, fdiam, fn = [int(p.split("=")[1]) for p in [sfc, sfd, sfn]]
        if fdim == dim and fdiam == diam:
            return nx.read_pajek(path + file)

    dims = diameter*np.ones(shape=(dim,), dtype=int)
    gr = nx.grid_graph(dim=list(dims), periodic=periodic)
    nx.write_pajek(
        gr, path + f"dim={dim}_diam={diam}_n={nx.number_of_nodes(gr)}")
    return gr


def generate_regular_graph(c, d):

    max_dist = 0
    dd = np.min([10, d-1])
    count_nodes = c*d
    adder = 1
    mult = 5
    rg = None
    num_iter = 1
    while max_dist != d:
        if (c*count_nodes) % 2 != 0:
            count_nodes += adder
            continue
        ntry = 1
        while ntry < 5 and max_dist != d:
            rg = nx.random_regular_graph(c, count_nodes)
            max_dist = nx.diameter(rg)
            ntry += 1
        if d + dd < max_dist:
            adder = -1
        elif max_dist < d - dd:
            adder = 1
        if num_iter % 10 == 0:
            print("count_nodes", count_nodes)
        num_iter += 1
        # count_nodes += adder
        count_nodes = count_nodes * mult
    return rg, max_dist


def generate_regular_graph2(c, d):
    count_nodes = 2*c*d

    count = 5

    def gen():
        rg = nx.random_regular_graph(c, count_nodes)
        max_dist = nx.diameter(rg)
        return rg, max_dist
    step = 5000
    state = 0
    while True:
        print(f"Current count nodes: {count_nodes}")
        print(f"Current state: {state}")
        print(f"Current step: {step}")
        sim_results = Parallel(n_jobs=count)(delayed(gen)()
                                             for i in range(count))
        diams = np.array([sr[1] for sr in sim_results])
        if np.all(diams < d):
            if state == 1:
                step *= 2
                state = 0
            count_nodes += step
            state = 1
        elif np.all(diams > d):
            if state == -1:
                step /= 2
                state = 0
            count_nodes -= step
            state = -1
        elif np.any(diams == d):
            ind = np.where(diams == d)[0][0]
            return sim_results[ind][0]
        else:
            state = 0


def get_regular_graph(c, d, new_graph=False, path="./regular_graphs/"):

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


def create_graphs():
    a_diams = np.arange(2, stop=10, dtype=int)
    a_connect = np.arange(2, stop=10, dtype=int)
    for i, diam in enumerate(a_diams):
        for j, connect in enumerate(a_connect):
            igraph = get_regular_graph(connect, diam)


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
            igraph = get_regular_graph(connect, diam)

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
    igraph = get_regular_graph(c, H)

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


def simulation_on_grid_graph():
    count = 10000

    a_diams = np.arange(2, stop=10, dtype=int)
    a_dims = np.arange(2, stop=5, dtype=int)

    res_shape = (len(a_diams), len(a_dims))
    mfpt_dj = np.zeros(shape=res_shape)
    mfpt_gr = np.zeros(shape=res_shape)
    real_diameters = np.zeros(shape=res_shape)
    real_coonectivity = np.zeros(shape=res_shape)

    u = 1e3
    p = 0.1
    periodic = False

    ar_p = p*np.ones(shape=res_shape)
    ar_u = u*np.ones(shape=res_shape)

    for i, diam in enumerate(a_diams):
        for j, dim in enumerate(a_dims):
            igraph = genarate_grid(dim, diam, periodic)

            connect = 2*dim
            real_connect = nx.average_degree_connectivity(igraph)
            real_diameter = nx.diameter(igraph)

            sim_results = Parallel(n_jobs=15)(delayed(sim)(igraph, p, u)
                                              for i in range(count))

            dj_lengths = [r[0] for r in sim_results]
            gr_lengths = [r[1] for r in sim_results]

            dj_mfpt = sum(dj_lengths)/count
            gr_mfpt = sum(gr_lengths)/count

            mfpt_dj[i, j] = dj_mfpt
            mfpt_gr[i, j] = gr_mfpt
            real_diameters[i, j] = real_diameter
            real_coonectivity[i, j] = list(real_connect.keys())[0]

            print("_____Result______")
            print('Probability(p):', p)
            print('Cost(u):', u)
            print('Diameter(diam):', diam)
            print('Connectivity(connect):', connect)
            print('Real Diameter(connect):', real_diameter)
            print('Real Connectivity(connect):', real_connect)
            print('MFPT Dijkstra:', dj_mfpt)
            print('MFPT Greedy:', gr_mfpt)

    coord_names = ["diameter", "dim"]

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
        "diameter": a_diams,
        "dim": a_dims,
    })

    df.to_netcdf(
        f"grid_u={u}_p={p}_max-dims={a_dims[-1]}_max-diams={a_diams[-1]}_periodic={periodic}.nc")

    df.lnz.plot()
    plt.show()


def simulation_on_small_world_graph():
    count = 10000

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


def main():
    # simulation_on_small_world_graph()
    # simulation_on_regular_graph_zch()
    # simulation_on_regular_graph_zup()

    #
    # Tasks
    # 1) structures graphs (grid 2d,3d,4d,...)
    # 2) check for graph E.R. or
    # 3)


if __name__ == '__main__':
    main()
