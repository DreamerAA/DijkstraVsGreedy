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
import pandas as pd

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.simulator import Simulator
from base.graph_creator import GraphCreator
from base.community_extractor import CommunityExtractor
from utils.utils import get_cmap
from visualizer.visualizer import Visualizer



m_data_path = "../data/GitHub/"

def simulation_zup(graph, count_sim:int = 100):
    # FOR TESTING
    a_u = np.array([1e2, 1e3, 1e4, 1e5, 1e6,
                    1e7,  1e8, 1e9], dtype=np.float64)
    a_p = np.arange(0.0, 0.051, 0.01, dtype=np.float64)

    # FULL
    # a_u = np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6,
    #                5e6, 1e7, 5e7, 1e8, 5e8, 1e9], dtype=np.float64)
    # a_p = np.arange(0.0, 0.051, 0.002, dtype=np.float64)

    fname = m_data_path + f"/1_u={a_u[0]}:(*10):{a_u[-1]}_p={a_p[0]}:0.01:{a_p[-1]}_count_sim={count_sim}.nc"

    Simulator.simulation_up(graph, a_u, a_p, fname, count_sim)
    return fname 


def main():
    
    fname = m_data_path + "/git_web_ml/musae_git_edges.csv"
    
    data = pd.read_csv(fname, delimiter=',', on_bad_lines='skip')
    sources = data["id_1"].to_numpy(dtype=np.int32)
    targets = data["id_2"].to_numpy(dtype=np.int32)

    print(sources.min(), sources.max())
    print(targets.min(), targets.max()) 

    layout = "kamada"
    # layout = "spring"
    # layout = "spectral"

    save_pos_path= (True, m_data_path+f"./{layout}_full_layout_pos.npy",
                    m_data_path+f"./{layout}_full_layout_corr.npy")
    
    graph = nx.Graph()
    graph.add_edges_from(list(zip(sources, targets)))
    Visualizer.showGraph(graph, layout=layout, save_pos_path = save_pos_path)


if __name__ == '__main__':
    main()
