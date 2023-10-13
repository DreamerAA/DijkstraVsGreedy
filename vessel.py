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



def simulation_zup(graph):

    # a_u = np.array([1e5, 1e4], dtype=np.float64)
    # a_p = np.array([0.2, 0.3], dtype=np.float64)
    a_u = np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6,
                   5e6, 1e7, 5e7, 1e8, 5e8, 1e9], dtype=np.float64)
    # degree = [1+i for i in range(24)]
    # a_u = np.array([10**d for d in degree], dtype=np.float64)
    a_p = np.arange(0.0, 0.051, 0.002, dtype=np.float64)

    fname = f"/media/andrey/Samsung_T5/PHD/results/vessel/1_u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}.nc"

    Simulator.simulation_up(graph, a_u, a_p, fname, 100)
    return fname 


def extract_ndists(graph):
    one_degree_nodes = np.array([d[0] for d in graph.degree() if d[1] == 1])
    node_numbers = np.array([n for n in graph.nodes()])
    max_num_node = node_numbers.max() + 1
    od_nodes = np.zeros(shape=(max_num_node,),dtype=bool)
    od_nodes[one_degree_nodes] = True

    ndists = np.zeros(shape=(max_num_node,),dtype=float)

    for i in range(max_num_node):
        if od_nodes[i]:
            continue
        length = nx.single_source_dijkstra_path_length(graph, i)
        odn_lengthes = np.array([length[odn] for odn in one_degree_nodes])
        ndists[i] = odn_lengthes.min()
        assert(odn_lengthes.min() != 0)

    ndists = ndists/ndists.max()

    tdists = np.array([1 for e in graph.edges()])
    return ndists, tdists




def main():
    fnodes = '../data/VesselGraph/1_b_3_0_nodes_processed.csv'
    fedges = '../data/VesselGraph/1_b_3_0_edges_processed.csv'
    # fnodes = '../data/VesselGraph/C57BL_6-K20_b_3_0_nodes_processed.csv'
    # fedges = '../data/VesselGraph/C57BL_6-K20_b_3_0_edges_processed.csv'

    # b = np.array([[3500,7000],[3500,7000],[3000,7000]])
    # graph, node_pos = GraphCreator.getVessel(fnodes,fedges, borders=b)

    graph, node_pos = GraphCreator.getVessel(fnodes,fedges)

    # ndistst, tdists = extract_ndists(graph)


    # print(len(graph.nodes()))
    # print(len(graph.edges()))

    # GraphCreator.removeOneDegreeNodes(graph)

    kamada_node_pos = np.load("../data/VesselGraph/synt_kamada_pos.npy")
    kamada_corr = np.load("../data/VesselGraph/synt_kamada_corr.npy")

    # node_pos = {}
    # for n in graph.nodes(data=True):
    #     p = n[1]
    #     node_pos[n[0]] = (p['x'],p['y'],p['z'])

    corr = np.array([i for i in range(node_pos.shape[0])])
    # Visualizer.draw_nxvtk(graph,(node_pos,corr), size_node=1,size_edge=0.3,scale="one_ax_by_1")
    Visualizer.split_view((graph,node_pos,corr),
                          (graph,kamada_node_pos,kamada_corr), 
                          size_node=0.6,size_edge=0.3,scale="one_ax_by_1",animation=True)
    # Visualizer.draw_nxvtk(graph,node_pos, size_node=1,size_edge=0.3,scale="one_ax_by_1")
    # Visualizer.showGraph(graph, size_node=1,size_edge=0.3,
    #                      layout='kamada', 
    #                      save_pos_path=("../data/VesselGraph/synt_kamada_pos.npy","../data/VesselGraph/synt_kamada_corr.npy"),
    #                      animation=True)

    # Visualizer.draw_hist(graph, mrange=(1, 5))

    # cc = GraphCreator.extractAvareageDegree(graph)
    # cd = nx.diameter(graph)
    # print("diameter:", cd)
    # print("avarage degree:", cc)

    # fname = simulation_zup(graph)

    # data_name = "u_from_10.0_to_1e+24_p_from_0.0_to_0.05.nc"
    # fname = f"/media/andrey/Samsung_T5/PHD/results/vessel/{data_name}"

    # Visualizer.showRegularResult(fname, field="ln10_z",xticks=[i*0.01 for i in range(6)])
    # Visualizer.add_critical_u(34, 0.05)

    plt.show()


if __name__ == '__main__':
    main()
