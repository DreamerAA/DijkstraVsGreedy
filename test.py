from graphviz import Graph
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed
import xarray as xr
from os import listdir
from os.path import isfile, join
from base.graph_creator import GraphCreator
from base.simulator import Simulator, SimulationSettings, UncertaintyCond
import pandas as pd
from utils.timer import Timer
import os.path
from multiprocessing import Process, Pool
from base.graph_creator import GraphCreator
from visualizer.visualizer import Visualizer
import random
import pickle
from base.community_extractor import CommunityExtractor
import time

def random_color():
    return (random.randint(0, 255)/255., random.randint(0, 255)/255., random.randint(0, 255)/255., 1.)

def get_cmap(n, name='Paired'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

n = 4**4
graph = GraphCreator.generateSmallWorld(n, 6, 0.1)

extractor = CommunityExtractor(graph)

# res = extractor.girvan_newman(4)
# res = extractor.fluid_communities(3)
res = extractor.label_propagation()
# res = extractor.louvain_communities() # good

count_community = len(res)
print(f"Count community: {count_community}")

cmap = get_cmap(count_community)
colors_data = {i:cmap(i) for i in range(count_community)}

ngraph = extractor.graph_with_community(res)

# Visualizer.showGraph(ngraph, size_node=1., size_edge=0.1, layout='spring', colors_data=colors_data)

# def have_link(c1:int, c2:int)->bool:
#     for n1 in res[c1]:
#         linked = np.array([u in graph.neighbors(n1) for u in res[c2]],dtype=np.bool_)
#         if np.any(linked):
#             return True
#     return False

# linked_list = [(c1, c2) for c1 in range(count_community) for c2 in range(c1+1,count_community) if have_link(c1,c2)]

# cgraph = nx.Graph()
# cgraph.add_node_from([i for i in range(count_community)])
# cgraph.add_edges_from()

# with open("communities", "wb") as fp:   #Pickling
#     pickle.dump(res, fp)
# with open("communities", "rb") as fp:   #Pickling
#     r = pickle.load(fp)
#     print(r)