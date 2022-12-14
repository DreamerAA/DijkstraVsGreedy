from os import listdir
from os.path import isfile, join
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
import pandas as pd


class GraphCreator:
    def findRegularGraph(c, d, new_graph=False, path="./regular_graphs/"):
        if not new_graph:
            onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
            for file in onlyfiles:
                sfc, sfd, sfn = file.split("_")
                fc, fd, fn = [int(p.split("=")[1]) for p in [sfc, sfd, sfn]]
                if fc == c and fd == d:
                    return nx.read_pajek(path + file)
        return None

    def findGridGraph(dim, diameter, periodic, path="../Results/grid_graphs/"):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for file in onlyfiles:
            fdim, fdiam, fn, rc, rd = [int(p.split("=")[1])
                                       for p in file.split("_")]
            if fdim == dim and fdiam == diameter:
                return rc, rd, nx.read_pajek(path + file)

    def generateRegularGraph(c, d):
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

    def genarateGridGraph(dim, diameter, periodic, path="../Results/grid_graphs/", r_cd=True):
        dims = diameter*np.ones(shape=(dim,), dtype=int)
        gr = nx.grid_graph(dim=list(dims), periodic=periodic)

        rc, rd = -1, -1
        if r_cd:
            print(' --- Calculate: connectivity, diameter')
            adc = nx.average_degree_connectivity(gr)
            rc = int(np.sum(list(adc.values())))
            rd = nx.diameter(gr)
        return rc, rd, gr

    def getGridGraph(dim, diameter, periodic, path="../Results/grid_graphs/"):
        graph = GraphCreator.findGridGraph(dim, diameter, periodic, path)
        if graph is None:
            rc, rd, graph = GraphCreator.genarateGridGraph(
                dim, diameter, periodic, path)
            nx.write_pajek(
                graph, path + f"dim={dim}_diam={diameter}_n={nx.number_of_nodes(graph)}_rc={rc}_rd={rd}")
        return graph

    def getRegularGraph(c, d, new_graph=False, path="../Results/regular_graphs/"):
        graph = GraphCreator.findRegularGraph(c, d, new_graph, path)
        if graph is None:
            graph = GraphCreator.genarateRegularGraph(c, d)
            nx.write_pajek(
                graph, path + f"c={c}_d={d}_n={nx.number_of_nodes(graph)}")
        return graph

    def generateSmallWorld(count_nodes, count_neighbours, p=0.5):
        return nx.watts_strogatz_graph(n=count_nodes, k=count_neighbours, p=p)

    def findSmallWorldGraph(count_nodes, count_neighbours, p=0.5, path="../Results/small_world/"):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for file in onlyfiles:
            params = [p.split("=")[1]
                      for p in file.split("_")]
            rd, n, cn = [int(p) for p in params[1:-1]]
            rc = float(params[0])
            np = float(params[-1])
            if n == count_nodes and cn == count_neighbours and np == p:
                return rc, rd, nx.read_pajek(path + file)
        return 0, 0, None

    def getSmallWorldGraph(count_nodes, count_neighbours, p=0.5, path="../Results/small_world/"):
        rc, rd, graph = GraphCreator.findSmallWorldGraph(
            count_nodes, count_neighbours, p, path)
        if graph is None:
            graph = GraphCreator.generateSmallWorld(
                count_nodes, count_neighbours, p)
            rc = GraphCreator.extractAvareageDegree(graph)
            rd = nx.diameter(graph)
            nx.write_pajek(
                graph, path + f"c={rc}_d={rd}_n={count_nodes}_cn={count_neighbours}_p={p}")
        return rc, rd, graph

    def getFacebook(path):
        data = pd.read_csv(path, delimiter=' ')
        id1 = data["i1"].to_numpy()
        id2 = data["i2"].to_numpy()
        graph = nx.Graph()
        graph.add_edges_from(list(zip(id1, id2)))
        return graph

    def generateScaleFreeGraph(count_nodes, alpha, beta, gamma, delta_in=0.2, delta_out=0):
        mgraph = nx.scale_free_graph(
            count_nodes, alpha=alpha, beta=beta, gamma=gamma, delta_in=delta_in, delta_out=delta_out,)
        return nx.Graph(mgraph)

    def findScaleFreeGraph(c, d, path="../Results/scale_free_graphs/"):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for file in onlyfiles:
            params = [p.split("=")[1]
                      for p in file.split("_")]
            rc, rd, _ = [int(p) for p in params]
            if rc == c and rd == d:
                return rc, rd, nx.read_pajek(path + file)
        return 0, 0, None

    def getScaleFreeGraph(rc, rd, path="../Results/scale_free_graphs/"):
        _, _, graph = GraphCreator.findScaleFreeGraph(rc, rd, path)
        if graph is None:
            count_nodes = 276
            cc, cd = 0, 0
            while cc != rc or cd != rd:
                sum_d = []
                for alpha in np.arange(0.01, 0.99, 0.01):
                    for beta in np.arange(0.01, 0.99, 0.01):
                        if beta + alpha >= 0.99:
                            continue
                        gamma = 1 - alpha - beta
                        graph = GraphCreator.generateScaleFreeGraph(
                            count_nodes, alpha, beta, gamma, delta_in=1, delta_out=1)

                        components = nx.connected_components(graph)
                        comp = [c for c in components]
                        if len(comp) > 1:
                            continue

                        cc = GraphCreator.extractAvareageDegree(graph)
                        cd = nx.diameter(graph)
                        sum_d.append(cd)
                        if round(cc) == rc and cd == rd:
                            nx.write_pajek(
                                graph, path + f"c={int(rc)}_d={rd}_n={count_nodes}")
                            return rc, rd, graph
                # if count_nodes <= 1024 and count_nodes > 513:  # 370\2
                count_nodes -= 1

                # if sum(sum_d)/len(sum_d) > rd:
                #     if count_nodes <= 64:
                #         count_nodes -= 1
                #     else:
                #         count_nodes = int(count_nodes/2)
                # else:
                #     if count_nodes >= 64:
                #         count_nodes = int(count_nodes*2)
                #     else:
                #         count_nodes += 1

                print(f" --- count nodes: {count_nodes} ----")

        return 0, 0, graph

    def extractAvareageDegree(graph):
        degree = np.array([d[1] for d in graph.degree()])
        return degree.mean()
