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


class Visualizer:

    enumerate

    def critical_u(H):
        return H*np.exp(H + 1)

    def add_vert(x, color=None, ymax=310):
        plt.plot([x, x], [0, ymax], linewidth=3, color=color)

    def draw_hist(igraph, mrange=(1, 240), rwidth=1, bins=80, xticks=None):
        degree = np.array([d[1] for d in igraph.degree()], dtype=int)

        hist = np.zeros(shape=(degree.max()+1,), dtype=int)
        for d in degree:
            hist[d] += 1

        plt.hist(degree, bins=bins, histtype='bar',
                 range=mrange, rwidth=rwidth)
        print(degree.mean())
        print(np.median(degree))
        print(hist.argmax())
        plt.xticks(xticks, fontsize=18)
        plt.yticks(None, fontsize=18)
        return degree, hist

    def draw_nxvtk(G, node_pos, size_node=0.25, size_edge=0.02, scale="full_by_1"):
        """
        Draw networkx graph in 3d with nodes at node_pos.

        See layout.py for functions that compute node positions.

        node_pos is a dictionary keyed by vertex with a three-tuple
        of x-y positions as the value.

        The node color is plum.
        The edge color is banana.

        All the nodes are the same size.

        @todo to enumerate
        Scale: full_by_1, one_ax_by_1, no

        """
        mrange = 1e2
        positions = np.zeros(shape=(len(node_pos.keys()), 3), dtype=float)
        for i in node_pos.keys():
            k = int(i)
            positions[k, :] = node_pos[i]


        a_min = positions.min(axis=0)
        a_max = positions.max(axis=0)
        diff = a_max - a_min
        dmax = diff.max()
        if scale == "full_by_1" or scale == "one_ax_by_1":
            if scale == "full_by_1":
                for i in range(3):
                    positions[:, i] = (positions[:, i] - a_min[i]) * \
                    2*mrange/diff[i] - mrange
            elif scale == "one_ax_by_1":
                for i in range(3):
                    positions[:, i] = (positions[:, i] - a_min[i])*2*mrange/dmax - mrange
                
                

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
        Visualizer.draw_nxvtk(graph, layout, size_node, size_edge)

    # 'viridis', 'RdBu', 'Spectral', 'bwr', 'seismic'

    def showRegularResult(data_path, xticks=None, yticks=None, field="lnz"):
        df = xr.load_dataset(data_path)
        res = df[field]

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        r = res.plot(cmap='seismic',vmin=-1,vmax=1)

        lbls = list(df.dims.keys())

        plt.xlabel(lbls[0], fontsize=20)
        plt.ylabel(lbls[1], fontsize=20)
        plt.xticks(xticks, fontsize=18)
        plt.yticks(yticks, fontsize=18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

    def add_critical_u(h, max_p=1):
        u = np.log10(Visualizer.critical_u(h))
        print(u)
        plt.plot([0, max_p], [u, u], color='darkgreen', linewidth=3.0)
