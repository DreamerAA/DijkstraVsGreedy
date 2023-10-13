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


def main():
    fname = "../data/Facebook/facebook_combined.csv"

    graph = GraphCreator.getFacebook(fname)


    node_pos = np.load("../data/Facebook/facebook_pos.npy")
    corr = np.load("../data/Facebook/corr.npy")
    mean = node_pos.mean(axis=0)
    for i in range(node_pos.shape[0]):
        node_pos[i,:] = 10*(node_pos[i,:] - mean) + mean


    Visualizer.draw_nxvtk(graph, (node_pos,corr), size_node=0.4,size_edge=0.02,scale="one_ax_by_1",animation=True)#
    # Visualizer.showGraph(graph, size_node=1,save_pos_path=("facebook_pos.npy", "corr.npy"),size_edge=0.3,layout='kamada')

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
