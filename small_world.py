import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import xarray as xr
from base.graph_creator import GraphCreator
from base.simulator import Simulator
from base.graph_creator import GraphCreator
from visualizer.visualizer import Visualizer



def simulation_zup(graph):

    a_u = np.array([1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3,
                    1e4, 5e4, 1e5, 5e5, 1e6, 5e6], dtype=np.float64)
    a_p = np.arange(0, 0.71, 0.01, dtype=np.float64)

    fname = f"/media/andrey/Samsung_T5/PHD/results/SmallWorld/u_from_{a_u[0]}_to_{a_u[-1]}_p_from_{a_p[0]}_to_{a_p[-1]}.nc"

    Simulator.simulation_up(graph, a_u, a_p, fname, 100)
    return fname 


def main():

    rc, rd ,graph = GraphCreator.getSmallWorldGraph(800, 10, p = 0.1, path="../data/Small-world/")

    print(rc, rd, len(graph.nodes()))

    # node_pos = np.load("../data/Facebook/facebook_pos.npy")
    # corr = np.load("../data/Facebook/corr.npy")
    # mean = node_pos.mean(axis=0)
    # for i in range(node_pos.shape[0]):
    #     node_pos[i,:] = 10*(node_pos[i,:] - mean) + mean


    # Visualizer.draw_nxvtk(graph, (node_pos,corr), size_node=0.4,size_edge=0.04,scale="one_ax_by_1",animation=True)#
    Visualizer.showGraph(graph, size_node=2,size_edge=0.2,layout='spectral')

    # Visualizer.draw_hist(graph, mrange=(1, 20), rwidth=5, bins=20)

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
