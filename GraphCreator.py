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

    def removeOneDegreeNodes(graph):
        degrees = np.array([d for _, d in graph.degree()])
        while np.count_nonzero(degrees == 1) != 0:
            nodes = []
            edges = []
            for n, d in graph.degree():
                if d == 1:
                    nn = [cn for cn in graph.neighbors(n)]
                    nodes.append(n)
                    edges.append((n,nn[0]))

            graph.remove_edges_from(edges)
            graph.remove_nodes_from(nodes)
            degrees = np.array([d for _, d in graph.degree()])
            

    def getFacebook(path):
        data = pd.read_csv(path, delimiter=',')
        id1 = data["i1"].to_numpy()
        id2 = data["i2"].to_numpy()
        graph = nx.Graph()
        graph.add_edges_from(list(zip(id1, id2)))
        return graph

    def getVessel(fnodes,fedges,borders=np.array([[-np.inf,np.inf],[-np.inf,np.inf],[-np.inf,np.inf]])):
        ndata = pd.read_csv(fnodes, delimiter=';')
        edata = pd.read_csv(fedges, delimiter=';')
        
        id1 = edata["node1id"].to_numpy(dtype=np.int32)
        id2 = edata["node2id"].to_numpy(dtype=np.int32)

        xpos = ndata["pos_x"].to_numpy(dtype=float)
        ypos = ndata["pos_y"].to_numpy(dtype=float)
        zpos = ndata["pos_z"].to_numpy(dtype=float)

        pos = np.zeros(shape=(xpos.shape[0],3))
        pos[:,0] = xpos
        pos[:,1] = ypos
        pos[:,2] = zpos
        
        id1,id2, pos = GraphCreator.cutGraph(id1,id2,pos, borders)

        # nodes = [(i, {"x":pos[i,0],"y":pos[i,1],"z":pos[i,2]}) for i in range(pos.shape[0])]

        graph = nx.Graph()
        graph.add_edges_from(list(zip(id1, id2)))

        components = nx.connected_components(graph)
        lc = np.array([len(c) for c in components])
        max_comp = list(nx.node_connected_component(graph,lc.argmax()))

        nmask = np.zeros(shape=(pos.shape[0],),dtype=bool)
        nmask[max_comp] = True
        id1,id2, pos = GraphCreator.rmNodes(nmask, id1,id2, pos)

        graph = nx.Graph()
        graph.add_edges_from(list(zip(id1, id2)))

        return graph, pos

    def rmNodes(mask, i1, i2, opos):
        good_edges = np.logical_and(mask[i1], mask[i2])
        nip = np.cumsum(mask) - 1
        nl1 = nip[i1[good_edges]]
        nl2 = nip[i2[good_edges]]
        npos = opos[mask,:]
        return nl1, nl2, npos   

    def cutGraph(l1,l2,pos,borders):
        xi = np.logical_and(pos[:,0] > borders[0,0], pos[:,0] < borders[0,1])
        yi = np.logical_and(pos[:,1] > borders[1,0], pos[:,1] < borders[1,1])
        zi = np.logical_and(pos[:,2] > borders[2,0], pos[:,2] < borders[2,1])
        ipos = np.logical_and(np.logical_and(xi, yi), zi)

        nl1, nl2, npos = GraphCreator.rmNodes(ipos, l1, l2, pos)

        nmask = np.zeros(shape=(npos.shape[0],),dtype=bool)
        nmask[nl1] = True
        nmask[nl2] = True

        nl1, nl2, npos = GraphCreator.rmNodes(nmask, nl1, nl2, npos)

        return nl1, nl2, npos


    def generateDirectedScaleFreeGraph(count_nodes, alpha, beta, gamma, delta_in=1, delta_out=1):
        mgraph = nx.scale_free_graph(
            count_nodes, alpha=alpha, beta=beta, gamma=gamma, delta_in=delta_in, delta_out=delta_out)
        return nx.Graph(mgraph)

    def generateBarabasiAlbertGraph(count_nodes, count_edges):
        mgraph = nx.barabasi_albert_graph(count_nodes, count_edges)
        return nx.Graph(mgraph)

    def generateErdosRenyiGraph(count_nodes, probability):
        mgraph = nx.erdos_renyi_graph(count_nodes, probability)
        return nx.Graph(mgraph)

    def findDirectedScaleFreeGraph(c, d, path="../Results/directed_scale_free_graphs/"):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for file in onlyfiles:
            params = [p.split("=")[1]
                      for p in file.split("_")]
            rc, rd, _ = [int(p) for p in params]
            if rc == c and rd == d:
                return rc, rd, nx.read_pajek(path + file)
        return 0, 0, None

    def findScaleFreeGraphAB(c, d, path="../Results/scale_free_graphs_ab/"):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for file in onlyfiles:
            params = [p.split("=")[1]
                      for p in file.split("_")]
            rc, rd, _ = [int(p) for p in params]
            if rc == c and rd == d:
                return rc, rd, nx.read_pajek(path + file)
        return 0, 0, None

    def getScaleFreeGraphAB(rc, rd, path="../Results/scale_free_graphs_ab/"):
        _, _, graph = GraphCreator.findScaleFreeGraphAB(rc, rd, path)
        if graph is None:
            count_nodes = 150
            cc, cd = 0, 0
            while cc != rc or cd != rd:
                # sum_d = []
                print(f" --- next: n={count_nodes}")
                for m in np.arange(1, count_nodes-1):
                    graph = GraphCreator.generateBarabasiAlbertGraph(
                        count_nodes, m)

                    components = nx.connected_components(graph)
                    comp = [c for c in components]
                    if len(comp) > 1:
                        continue

                    cc = GraphCreator.extractAvareageDegree(graph)
                    cd = nx.diameter(graph)
                    # sum_d.append(cd)
                    if round(cc) == rc and cd == rd:
                        nx.write_pajek(
                            graph, path + f"c={int(rc)}_d={rd}_n={count_nodes}")
                        return rc, rd, graph

                count_nodes -= 2
        else:
            return rc, rd, graph

    def getDirectedScaleFreeGraph(rc, rd, path="../Results/directed_scale_free_graphs/"):
        _, _, graph = GraphCreator.findDirectedScaleFreeGraph(rc, rd, path)
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
                        graph = GraphCreator.generateDirectedScaleFreeGraph(
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

    def extractIntAvareageDegree(graph):
        return int(round(GraphCreator.extractAvareageDegree(graph)))

    def connected(graph):
        components = nx.connected_components(graph)
        comp = [c for c in components]
        return len(comp) == 1
