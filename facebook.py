import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import xarray as xr
from GraphCreator import GraphCreator
from Simulator import Simulator
from GraphCreator import GraphCreator
from visualizer import Visualizer
from numpy.polynomial import Polynomial
import pickle
from utils import st_time
from community_extractor import CommunityExtractor
import os


def get_cmap(n, name='gist_rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def simulation_zup(graph):
    a_u = np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6,
                   5e6, 1e7, 5e7, 1e8, 5e8, 1e9], dtype=np.float64)
    a_p = np.arange(0.0, 0.051, 0.002, dtype=np.float64)

    fname = f"/media/andrey/Samsung_T5/PHD/results/Facebook/1_u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}.nc"

    Simulator.simulation_up(graph, a_u, a_p, fname, 100)
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

    # ngraph = extractor.graph_with_community(res)

    # Visualizer.draw_nxvtk(ngraph, (node_pos,corr), size_node=0.4,size_edge=0.04,scale="one_ax_by_1",colors_data=colors_data)
    # Visualizer.showGraph(ngraph, size_node=1., size_edge=0.1, layout='spring', colors_data=colors_data)

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


    # node_pos = np.load("../data/Facebook/facebook_pos.npy")
    # corr = np.load("../data/Facebook/corr.npy")
    # mean = node_pos.mean(axis=0)
    # for i in range(node_pos.shape[0]):
    #     node_pos[i,:] = 10*(node_pos[i,:] - mean) + mean


    # Visualizer.draw_nxvtk(graph, (node_pos,corr), size_node=0.4,size_edge=0.04,scale="one_ax_by_1",animation=True)#
    # Visualizer.showGraph(graph, size_node=1,save_pos_path=("facebook_pos.npy", "corr.npy"),size_edge=0.3,layout='kamada')

    Visualizer.draw_hist(cgraph, mrange=(1, 20), rwidth=1, bins=20)# color='#8150ba'

    # cc = GraphCreator.extractAvareageDegree(graph)
    # cd = nx.diameter(graph)
    # print("diameter:", cd)
    # print("avarage degree:", cc)

    # fname = simulation_zup(graph)

    # Visualizer.showRegularResult(fname, field="ln10_z",xticks=[i*0.01 for i in range(6)])
    # Visualizer.add_critical_u(34, 0.05)

    plt.show()


if __name__ == '__main__':
    main()
