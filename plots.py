import matplotlib.pylab as plt
from matplotlib import pylab
import numpy as np
import xarray as xr
import networkx as nx
from os import listdir
from os.path import isfile, join
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


def draw_nxvtk(G, node_pos):
    """ 
    Draw networkx graph in 3d with nodes at node_pos. 

    See layout.py for functions that compute node positions. 

    node_pos is a dictionary keyed by vertex with a three-tuple 
    of x-y positions as the value. 

    The node color is plum. 
    The edge color is banana. 

    All the nodes are the same size. 

    """
    # set node positions
    np = {}
    colors = vtkNamedColors()
    for n in G.nodes():
        try:
            np[n] = node_pos[n]
        except KeyError:
            raise nx.NetworkXError("node %s doesn't have position" % n)

    nodePoints = vtkPoints()

    i = 0
    for (x, y, z) in np.values():
        nodePoints.InsertPoint(i, x, y, z)
        i = i+1

    # Create a polydata to be glyphed.
    inputData = vtkPolyData()
    inputData.SetPoints(nodePoints)

    # Use sphere as glyph source.
    balls = vtkSphereSource()
    balls.SetRadius(.025)
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
    for u, v in G.edges():
        # The edge e can be a 2-tuple (Graph) or a 3-tuple (Xgraph)
        if v in node_pos and u in node_pos:
            lines.InsertNextCell(2)
            for n in (u, v):
                (x, y, z) = node_pos[n]
                points.InsertPoint(i, x, y, z)
                lines.InsertCellPoint(i)
                i = i+1

    edgeData.SetPoints(points)
    edgeData.SetLines(lines)

    # Add thickness to the resulting line.
    Tubes = vtkTubeFilter()
    Tubes.SetNumberOfSides(16)
    Tubes.SetInputData(edgeData)
    Tubes.SetRadius(.002)
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

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Add the actors
    ren.AddActor(glyph)
    ren.AddActor(profile)
    ren.SetBackground(colors.GetColor3d("White"))

    camera = ren.GetActiveCamera()
    camera.SetFocalPoint(0, 0, 0)
    camera.SetPosition(1, 1, 1)

    # renWin.SetSize(640, 640)

    renWin.Render()
    iren.Start()


def get_regular_graph(c, d, new_graph=False, path="./regular_graphs/"):

    if not new_graph:
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for file in onlyfiles:
            sfc, sfd, sfn = file.split("_")
            fc, fd, fn = [int(p.split("=")[1]) for p in [sfc, sfd, sfn]]
            if fc == c and fd == d:
                return nx.read_pajek(path + file)

    return None


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


def showResult(data_path, xticks=None, yticks=None):
    df = xr.load_dataset(data_path)

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)

    r = df.lnz.plot(cmap='seismic')

    lbls = list(df.dims.keys())

    plt.xlabel(lbls[0], fontsize=20)
    plt.ylabel(lbls[1], fontsize=20)
    plt.xticks(xticks, fontsize=18)
    plt.yticks(yticks, fontsize=18)


def showGraph(G):
    graph = nx.Graph(G)

    edges = [(i, j, 1) for i, j in graph.edges()]
    graph.add_weighted_edges_from(edges)

    layout = nx.kamada_kawai_layout(graph, dim=3)
    # layout = nx.spring_layout(graph, dim=3, k=0.5)
    # layout = nx.spectral_layout(graph, dim=3)

    draw_nxvtk(graph, layout)


def add_critical_u(h, max_p=1):
    u = np.log10(critical_u(h))
    print(u)
    plt.plot([0, max_p], [u, u], color='darkgreen', linewidth=3.0)


def regular_graph_main():
    # Regular graph
    showResult(
        "./RegularGraphResults/Regular_graph_u_from_1.0_to_5000000.0_p_from_0.0_to_0.4_c=6_D=6.nc",
        xticks=[0, 0.1, 0.2, 0.3, 0.4])

    add_critical_u(3, 0.4)

    showResult(
        "./RegularGraphResults/Regular_graph_u=500.0_p=0.15_max-diam=8_max-degree=8.nc")

    G = get_regular_graph(6, 6)
    showGraph(G)

    plt.show()


def grid_graph_main():
    pass


def main():
    regular_graph_main()
    grid_graph_main()


if __name__ == '__main__':
    main()
