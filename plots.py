import matplotlib.pylab as plt
from matplotlib import pylab
import numpy as np
import xarray as xr
import networkx as nx
from matplotlib.ticker import MaxNLocator
from GraphCreator import GraphCreator
from vedo import *
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkFiltersCore import (
    vtkGlyph3D,
    vtkTubeFilter
)

from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleFlight,
    vtkInteractorStyleTrackballActor
)

from vtkmodules.vtkCommonDataModel import (
    vtkPolyData,
    vtkCellArray
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def critical_u(H):
    return H*np.exp(H + 1)


def add_vert(x, color=None, ymax=310):
    plt.plot([x, x], [0, ymax], linewidth=3, color=color)


def draw_hist(igraph, mrange=(1, 240), rwidth=1, bins=80, xticks=None):
    degree = np.array([d[1] for d in igraph.degree()], dtype=int)

    hist = np.zeros(shape=(degree.max()+1,), dtype=int)
    for d in degree:
        hist[d] += 1

    plt.hist(degree, bins=bins, histtype='bar', range=mrange, rwidth=rwidth)
    print(degree.mean())
    print(np.median(degree))
    print(hist.argmax())
    plt.xticks(xticks, fontsize=18)
    plt.yticks(None, fontsize=18)
    return degree, hist


def draw_nxvtk(G, node_pos, size_node=0.25, size_edge=0.02):
    """
    Draw networkx graph in 3d with nodes at node_pos.

    See layout.py for functions that compute node positions.

    node_pos is a dictionary keyed by vertex with a three-tuple
    of x-y positions as the value.

    The node color is plum.
    The edge color is banana.

    All the nodes are the same size.

    """
    mrange = 1e2
    positions = np.zeros(shape=(len(node_pos.keys()), 3), dtype=float)
    for i in node_pos.keys():
        k = int(i)
        positions[k, :] = node_pos[i]

    a_min = positions.min(axis=0)
    a_max = positions.max(axis=0)
    diff = a_max - a_min
    for i in range(3):
        positions[:, i] = (positions[:, i] - a_min[i]) * \
            2*mrange/diff[i] - mrange

    # set node positions
    colors = vtkNamedColors()
    nodePoints = vtkPoints()

    i = 0
    count_nodes = positions.shape[0]
    for (x, y, z) in positions:
        nodePoints.InsertPoint(i, x, y, z)
        i = i+1

    # Create a polydata to be glyphed.
    inputData = vtkPolyData()
    inputData.SetPoints(nodePoints)

    # Use sphere as glyph source.
    balls = vtkSphereSource()
    balls.SetRadius(size_node)
    balls.SetPhiResolution(20)
    balls.SetThetaResolution(20)

    glyphPoints = vtkGlyph3D()
    glyphPoints.SetInputData(inputData)
    glyphPoints.SetSourceConnection(balls.GetOutputPort())

    glyphMapper = vtkPolyDataMapper()
    glyphMapper.SetInputConnection(glyphPoints.GetOutputPort())

    glyph = vtkActor()
    glyph.SetMapper(glyphMapper)
    glyph.GetProperty().SetDiffuseColor(1., 0., 0.)
    glyph.GetProperty().SetSpecular(.3)
    glyph.GetProperty().SetSpecularPower(30)

    # Generate the polyline for the spline.
    points = vtkPoints()
    edgeData = vtkPolyData()

    # Edges

    lines = vtkCellArray()
    i = 0
    count_edges = len(G.edges())
    for u, v in G.edges():
        # The edge e can be a 2-tuple (Graph) or a 3-tuple (Xgraph)
        lines.InsertNextCell(2)
        for n in (u, v):
            (x, y, z) = positions[int(n), :]
            points.InsertPoint(i, x, y, z)
            lines.InsertCellPoint(i)
            i = i+1

    edgeData.SetPoints(points)
    edgeData.SetLines(lines)

    # Add thickness to the resulting line.
    Tubes = vtkTubeFilter()
    Tubes.SetNumberOfSides(16)
    Tubes.SetInputData(edgeData)
    Tubes.SetRadius(size_edge)
    #
    profileMapper = vtkPolyDataMapper()
    profileMapper.SetInputConnection(Tubes.GetOutputPort())

    #
    profile = vtkActor()
    profile.SetMapper(profileMapper)
    profile.GetProperty().SetDiffuseColor(0., 0., 1.)
    profile.GetProperty().SetSpecular(.3)
    profile.GetProperty().SetSpecularPower(30)

    # Now create the RenderWindow, Renderer and Interactor
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)

    style = vtkInteractorStyleTrackballCamera()
    # style = vtkInteractorStyleFlight()
    # style = vtkInteractorStyleTrackballActor()
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetInteractorStyle(style)

    # Add the actors
    ren.AddActor(glyph)
    ren.AddActor(profile)
    ren.SetBackground(colors.GetColor3d("White"))

    camera = ren.GetActiveCamera()
    camera.SetFocalPoint(0, 0, 0)
    camera.SetPosition(100, 100, 100)
    # renWin.SetSize(640, 640)

    renWin.Render()
    iren.Start()


def z(c, H, u, p):
    return 2*(c**H)*(u*(p**c)+1)/(u*p+1)


def z_p():
    c, H = 4, 4
    du = np.arange(3, 7)
    u = 10**du
    p = np.arange(0.0, 0.4, 0.05)
    for u_el in u:
        plt.plot(p, z(c, H, u_el, p))

    plt.plot()

# 'viridis', 'RdBu', 'Spectral', 'bwr', 'seismic'


def showRegularResult(data_path, xticks=None, yticks=None, field="lnz"):
    df = xr.load_dataset(data_path)
    res = df[field]

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    r = res.plot(cmap='seismic')
    # res = res.to_numpy()
    # res[res < -2] = -2
    # plt.imshow(res)

    lbls = list(df.dims.keys())

    plt.xlabel(lbls[0], fontsize=20)
    plt.ylabel(lbls[1], fontsize=20)
    plt.xticks(xticks, fontsize=18)
    plt.yticks(yticks, fontsize=18)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)


def showGridResult(data_path, xticks=None, yticks=None):
    df = xr.load_dataset(data_path)

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)

    sx, sy = 7, 7
    ad = np.arange(2, 2+sy)
    ac = np.arange(2, 2+sx)
    ares = np.empty((sy, sx))
    rc = df.rcoonectivity.to_numpy()
    rd = df.rdiameter.to_numpy()
    lnz = df.lnz.to_numpy()
    for i, d in enumerate(ad):
        for j, c in enumerate(ac):
            mask = np.logical_and(rc == c, rd == d)
            ares[i, j] = np.sum(lnz[mask])/np.sum(mask)

    coord_names = ["diameter", "connectivity"]
    res = xr.Dataset({
        "lnz": (coord_names, ares),
    },
        coords={
        coord_names[0]: ad,
        coord_names[1]: ac,
    })

    res.lnz.plot(cmap='seismic')
    # df.lnz.plot(cmap='seismic')

    # lbls = list(df.dims.keys())

    # plt.xlabel(lbls[0], fontsize=20)
    # plt.ylabel(lbls[1], fontsize=20)
    plt.xticks(xticks, fontsize=18)
    plt.yticks(yticks, fontsize=18)


def showGraph(G, size_node=0.25, size_edge=0.02, layout='kamada', **kwargs):
    graph = nx.Graph(G)

    print(f"sqrt={np.sqrt(len(graph.nodes()))}")
    edges = [(i, j, 1) for i, j in graph.edges()]
    graph.add_weighted_edges_from(edges)
    if layout == 'kamada':
        layout = nx.kamada_kawai_layout(graph, dim=3)
    elif layout == 'spring':
        layout = nx.spring_layout(
            graph, dim=3, **kwargs)
    elif layout == 'spectral':
        layout = nx.spectral_layout(graph, dim=3)
    draw_nxvtk(graph, layout, size_node, size_edge)


def showGridGraph(G, size_node=2, size_edge=0.2, layout='kamada', **kwargs):
    nodes = [n for n in G.nodes()]
    edges = [(nodes.index(i), nodes.index(j)) for i, j in G.edges()]

    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(len(nodes))])
    graph.add_edges_from(edges)
    if layout == 'kamada':
        layout = nx.kamada_kawai_layout(graph, dim=3)
    elif layout == 'spring':
        layout = nx.spring_layout(
            graph, dim=3, **kwargs)
    elif layout == 'spectral':
        layout = nx.spectral_layout(graph, dim=3)
    layout = {f"{i}": l for i, l in enumerate(layout.values())}
    draw_nxvtk(graph, layout, size_node, size_edge)


def add_critical_u(h, max_p=1):
    u = np.log10(critical_u(h))
    print(u)
    plt.plot([0, max_p], [u, u], color='darkgreen', linewidth=3.0)


def regular_graph_main():
    # Regular graph
    showRegularResult(
        "../Results/RegularGraphResults/Regular_graph_u_from_1.0_to_5000000.0_p_from_0.0_to_0.4_c=6_D=6.nc",
        xticks=[0, 0.1, 0.2, 0.3, 0.4])

    add_critical_u(3, 0.4)

    showRegularResult(
        "../Results/RegularGraphResults/Regular_graph_u=500.0_p=0.15_max-diam=8_max-degree=8.nc")

    G = GraphCreator.find_regular_graph(6, 6)
    showGraph(G)

    plt.show()


def grid_graph_main():
    pass
    # showGridResult(
    #     "../Results/GridGraphResults/grid_u=400.0_p=0.2_max-dims=6_max-diams=6_periodic=True.nc")

    # showRegularResult(
    #     "../Results/GridGraphResults/grid_u_from_1.0_to_50000000.0_p_from_0.0_to_0.7000000000000001_c=10_D=10.nc")

    # add_critical_u(5, 0.7)

    # igraph = nx.grid_graph(dim=[3, 4, 5], periodic=False)

    _, _, igraph = GraphCreator.genarateGridGraph(5, 5, True, r_cd=False)
    degree, hist = draw_hist(igraph, mrange=(
        8, 12), bins=6,)
    add_vert(degree.mean(), ymax=250, color=(
        1., 0.5, 14/255))

    # showGridGraph(igraph, size_node=1.5, size_edge=0.05, layout='spectral')


def small_world_graph_main():
    # showRegularResult(
    #     "../Results/SmallWorldGraphResults/u_from_1.0_to_5000000.0_p_from_0.0_to_0.7000000000000001_c=155_D=6.nc")

    # add_critical_u(3, 0.7)

    rc, rd, igraph = GraphCreator.getSmallWorldGraph(500, 10, 0.1)
    degree, hist = draw_hist(igraph, mrange=(
        1, 20), bins=20, xticks=[4, 8, 12, 16, 20])
    add_vert(degree.mean(), ymax=250, color=(
        1., 0.5, 14/255))
    # showGraph(igraph, size_node=2, size_edge=0.2,
    #           layout='spectral')  # spring kamada spectral


def facebook():
    # showRegularResult(
    #     "../Results/FacebookGraphResults/ego_facebook_u_from_100.0_to_5000000000000.0_p_from_0.0_to_0.705_c=117.63000341148297_D=8.nc", field="ln10_z")
    # showRegularResult(
    #     "../Results/FacebookGraphResults/ego_facebook_u_from_100.0_to_1000000000.0_p_from_0.0_to_0.7000000000000001_c=117.63000341148297_D=8.nc", field="ln10_z")
    # showRegularResult(
    #     "../Results/FacebookGraphResults/ego_facebook_u_from_100.0_to_1000000000.0_p_from_0.0_to_0.4_c=117.63000341148297_D=8.nc", field="ln10_z")
    # add_critical_u(4, 0.7)

    igraph = GraphCreator.getFacebook(
        "../../dataset/ego_facebook/ego_facebook.csv")
    print(GraphCreator.extractAvareageDegree(igraph))
    degree, hist = draw_hist(igraph)
    add_vert(degree.mean(), 150)
    add_vert(np.median(degree), 250)
    add_vert(hist.argmax())
    showGraph(igraph, 0.5, 0.0075, layout='spring', k=0.3)


def scale_free_er_graph_main():
    # showRegularResult(
    #     "../Results/ScaleFreeGraphResults/sfgr_d_from_3_to_8_c_from_3_to_8_u=1000000.0_p=0.08.nc", field="ln10_z")

    # showRegularResult(
    #     "../Results/ScaleFreeGraphResults/sfgr_u_from_100000.0_to_1e+20_p_from_0.0_to_0.1_c=3.198_D=12.nc",
    #     field="ln10_z", xticks=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1])

    graph = GraphCreator.generateScaleFreeGraphER(1000, 0.49, 0.49, 0.02)
    showGraph(graph, size_node=1.5, size_edge=0.25)


def scale_free_ab_graph_main():
    # fname = "u_from_1000.0_to_1000000000.0_p_from_0.0_to_0.02_c=4_D=8.nc"
    # showRegularResult(
    #     f"../Results/ScaleFreeGraphABResults/{fname}", field="ln10_z", xticks=[0.0, 0.005, 0.01, 0.015, 0.02])

    fname = "c_from_2_to_30_d_from_2_to_30_u=1000.0_p=0.1.nc"
    showRegularResult(
        f"../Results/ScaleFreeGraphABResults/{fname}", field="ln10_z")
    # xticks=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1]

    # _, _, graph = GraphCreator.getScaleFreeGraphAB(4, 8)
    # showGraph(graph, size_node=1, size_edge=0.1)
    # d = [d[1] for d in graph.degree()]
    # plt.hist(d, bins=100)


def main():
    # regular_graph_main()
    # grid_graph_main()
    # small_world_graph_main()
    facebook()
    # scale_free_er_graph_main()
    # scale_free_ab_graph_main()

    plt.show()


if __name__ == '__main__':
    main()
