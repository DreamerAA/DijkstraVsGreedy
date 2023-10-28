import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import xarray as xr
from pathlib import Path
from numpy.polynomial import Polynomial
import pickle
import os
import sys
from os.path import realpath

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.simulator import Simulator
from base.graph_creator import GraphCreator
from base.community_extractor import CommunityExtractor
from utils.utils import get_cmap
from visualizer.visualizer import Visualizer





def simulation_zup(graph, count_sim:int = 100):
    # FOR TESTING
    a_u = np.array([1e2, 1e3, 1e4, 1e5, 1e6,
                    1e7,  1e8, 1e9], dtype=np.float64)
    a_p = np.arange(0.0, 0.051, 0.01, dtype=np.float64)

    # FULL
    # a_u = np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6,
    #                5e6, 1e7, 5e7, 1e8, 5e8, 1e9], dtype=np.float64)
    # a_p = np.arange(0.0, 0.051, 0.002, dtype=np.float64)

    fname = f"../data/Facebook/1_u={a_u[0]}:(*10):{a_u[-1]}_p={a_p[0]}:0.01:{a_p[-1]}_count_sim={count_sim}.nc"

    Simulator.simulation_up(graph, a_u, a_p, fname, count_sim)
    return fname 


def main():
    fname = "../data/Facebook/facebook_combined.csv"

    graph = GraphCreator.getFacebook(fname)

    extractor = CommunityExtractor(graph)

    lbl_prop_comm_fn:str = "../data/Facebook/lbl_prop_comm"
    if not os.path.isfile(lbl_prop_comm_fn):
        lbl_prop_comm = extractor.label_propagation()
        with open(lbl_prop_comm_fn, "wb") as fp:   #Pickling
            pickle.dump(lbl_prop_comm, fp)
    else:
        with open(lbl_prop_comm_fn, 'rb') as handle:
            lbl_prop_comm = pickle.load(handle)

    louvain_comm_fn = "../data/Facebook/louvain_comm"
    if not os.path.isfile(louvain_comm_fn):
        louvain_comm = extractor.louvain_communities() # good
        with open(louvain_comm_fn, "wb") as fp:   #Pickling
            pickle.dump(louvain_comm, fp)
    else:
        with open(louvain_comm_fn, 'rb') as handle:
            louvain_comm = pickle.load(handle)

    # res = lbl_prop_comm
    res = louvain_comm


    count_community = len(res)
    print(f"Count community: {count_community}")
    cmap = get_cmap(count_community)
    colors_data = {i:cmap(i) for i in range(count_community)}

    node_pos = np.load("../data/Facebook/facebook_pos.npy")
    corr = np.load("../data/Facebook/corr.npy")
    mean = node_pos.mean(axis=0)
    for i in range(node_pos.shape[0]):
        node_pos[i,:] = 10*(node_pos[i,:] - mean) + mean

    def have_link(c1:int, c2:int)->bool:
        for n1 in res[c1]:
            linked = np.array([u in graph.neighbors(n1) for u in res[c2]],dtype=np.bool_)
            if np.any(linked):
                return True
        return False

    linked_list = [(c1, c2) for c1 in range(count_community) for c2 in range(c1+1,count_community) if have_link(c1,c2)]

    cgraph = nx.Graph()
    cgraph.add_nodes_from([(i, {"type_id": i}) for i in range(count_community)])
    cgraph.add_edges_from(linked_list)

    print(f"Graph diameter: {nx.diameter(cgraph)}")
    print(f"Graph connectivity: {GraphCreator.extractAvareageDegree(cgraph)}")

    degrees = [d[1] for d in cgraph.degree()]
    uniq_deg = list(set(degrees))
    # uniq_deg = uniq_deg[1:]
    dens = [degrees.count(uq)/len(degrees) for uq in uniq_deg]
    print(" --- density: ", dens)
    print(" --- degrees: ", uniq_deg)

    # Visualizer.showGraph(cgraph, size_node=5., size_edge=1, layout='spring', colors_data=colors_data)
    # Visualizer.draw_hist(cgraph, mrange=(1, 20), rwidth=1, bins=20)# color='#8150ba'

    fname = simulation_zup(graph)

    Visualizer.showRegularResult(fname, field="ln10_z",xticks=[i*0.01 for i in range(6)])
    # Visualizer.add_critical_u(34, 0.05)

    plt.show()


if __name__ == '__main__':
    main()
