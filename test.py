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

n = 6**4
graph = GraphCreator.generateSmallWorld(n, 6, 0.1)
Visualizer.showGraph(graph, size_node=1., size_edge=0.1, layout='spectral')
