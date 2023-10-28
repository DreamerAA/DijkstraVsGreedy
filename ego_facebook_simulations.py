from copy import deepcopy
from posixpath import split
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
# from igraph import *
import networkx as nx
import pydot
import random
from networkx.drawing.nx_pydot import graphviz_layout
from dataclasses import dataclass, field
from joblib import Parallel, delayed
import xarray as xr
from os import listdir
from os.path import isfile, join
from base.simulator import Simulator, SimulationSettings, UncertaintyCond
import pandas as pd
import networkx as nx
from base.simulator import show_graph
from utils.timer import Timer


def sim(igraph, p, u: np.float64):
    source, target = Simulator.choose_source_target(igraph)
    unc = UncertaintyCond(p, u)
    ss = SimulationSettings(unc, need_print=False,
                            source=source, target=target)
    simulator = Simulator(igraph, ss)
    return simulator.run()


# real_diameter=8
# av_connect=117.63000341148297
data = pd.read_csv(
    "../data/Facebook/facebook_combined.csv", delimiter=' ')
id1 = data["0"].to_numpy()
id2 = data["1"].to_numpy()

igraph = nx.Graph()
igraph.add_edges_from(list(zip(id1, id2)))

# count=10 - 25sec
count = 700  # [41, 44] sec

# a_u = np.array([10**40, 10**45, 10**50], dtype=np.float64)
# a_p = np.arange(0.001, 0.2, step=0.002, dtype=np.float64)
a_u = np.array([10**(i) for i in range(5, 11)], dtype=np.float64)
a_p = np.arange(0, 0.011, step=0.0005, dtype=np.float64)
# a_u = np.array([1e5, 1e10], dtype=np.float64)
# a_p = np.array([0, 0.0005, 0.001], dtype=np.float64)

res_shape = (len(a_u), len(a_p))

mfpt_dj = np.zeros(shape=res_shape)
mfpt_gr = np.zeros(shape=res_shape)

for i, u in enumerate(a_u):
    for j, p in enumerate(a_p):
        timer = Timer()
        timer.start()

        sim_results = Parallel(n_jobs=15)(delayed(sim)(igraph, p, u)
                                          for i in range(count))
        dj_lengths = [r[0] for r in sim_results]
        gr_lengths = [r[1] for r in sim_results]

        dj_mfpt = sum(dj_lengths)/count
        gr_mfpt = sum(gr_lengths)/count
        mfpt_dj[i, j] = dj_mfpt
        mfpt_gr[i, j] = gr_mfpt

        print("_____Result______")
        print('Probability(p):', p)
        print('Cost(u):', u)
        print('MFPT Dijkstra:', dj_mfpt)
        print('MFPT Greedy:', gr_mfpt)
        print('ln (z):', np.log10(gr_mfpt/dj_mfpt))
        timer.stop()

coord_names = ["ln10_u", "p"]
df = xr.Dataset({
    "gr_mfpt": (coord_names, mfpt_gr),
    "dj_mfpt": (coord_names, mfpt_dj),
    "ln10_z": (coord_names, np.log10(np.divide(mfpt_gr, mfpt_dj))),
},
    coords={
    coord_names[0]: np.log10(a_u),
    coord_names[1]: a_p,
})
df.ln10_z.plot()

df.to_netcdf(
    f"ego_facebook_result_7.nc")


# show_graph(graph)
# nx.draw(graph)
plt.show()
