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
    # fnodes = '../data/VesselGraph/1_b_3_0_nodes_processed.csv'
    # fedges = '../data/VesselGraph/1_b_3_0_edges_processed.csv'
    fnodes = '../data/VesselGraph/C57BL_6-K20_b_3_0_nodes_processed.csv'
    fedges = '../data/VesselGraph/C57BL_6-K20_b_3_0_edges_processed.csv'

    graph = GraphCreator.getVessel(fnodes,fedges)
    
    node_pos = {}
    for n in graph.nodes(data=True):
        p = n[1]
        node_pos[n[0]] = (p['x'],p['y'],p['z'])

    Visualizer.draw_nxvtk(graph,node_pos,size_node=1,size_edge=0.3,scale="one_ax_by_1")
    # Visualizer.draw_hist(graph, mrange=(1, 5))

    # generate_graphs()

    # checkExistedGraphs(10, 10)

    # simulation_zup()
    # fname = "vessel_graph_1_b_3_0.nc"
    # Visualizer.showRegularResult(
        # f"../Results/BarabasiAlbertResults/{fname}", field="ln10_z", xticks=[0, 0.1, 0.2, 0.3, 0.4])
    # Visualizer.add_critical_u(3, 0.4)

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
