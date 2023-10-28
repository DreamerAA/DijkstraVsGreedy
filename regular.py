import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import xarray as xr
from GraphCreator import GraphCreator
from Simulator import Simulator
from GraphCreator import GraphCreator
from visualizer import Visualizer



def main():
   
    graph = GraphCreator.generateRegularGraph(3, 5, 4)

    # Visualizer.draw_nxvtk(graph, (node_pos, corr), size_node=0.4,size_edge=0.04,scale="one_ax_by_1",animation=True)#
    Visualizer.showGraph(graph, size_node=3,size_edge=0.6,layout='kamada')

    # Visualizer.draw_hist(graph, mrange=(1, 100), rwidth=1, bins=100)# color='#8150ba'

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
