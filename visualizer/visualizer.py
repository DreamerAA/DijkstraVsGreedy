import matplotlib.pylab as plt
from matplotlib import pylab
import numpy as np
import xarray as xr
import networkx as nx
from matplotlib.ticker import MaxNLocator
import math as m
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkDoubleArray
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
    vtkRenderer,
    vtkColorTransferFunction,
)
def Rx(theta):
    return np.matrix([[ 1, 0           , 0           ],
                    [ 0, m.cos(theta),-m.sin(theta)],
                    [ 0, m.sin(theta), m.cos(theta)]])

def Ry(theta):
    return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                    [ 0           , 1, 0           ],
                    [-m.sin(theta), 0, m.cos(theta)]])

def Rz(theta):
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                    [ m.sin(theta), m.cos(theta) , 0 ],
                    [ 0           , 0            , 1 ]])

class vtkTimerCallback():
    def __init__(self, steps, actors, cameras, iren):
        self.timer_count = 0
        self.steps = steps
        self.actors = actors
        self.cameras = cameras
        self.iren = iren
        self.timerId = None
        self.angle = 0.0
        self.cur_pos = np.array(cameras[0].GetPosition())

        self.astep = 0.15
        # a = self.astep*np.pi/180
        # self.Rxyz = Rz(a) * Ry(a) * Rx(a)

    def calcXYZ(self):
        a = self.angle*np.pi/180
        self.Rxyz = Ry(a) * Rx(a)
        return (self.cur_pos*self.Rxyz).T.A.squeeze()

    def execute(self, obj, event):
        step = 0
        while step < self.steps:
            self.angle += self.astep
            cur_pos = self.calcXYZ()
            for camera in self.cameras:
                camera.SetPosition(cur_pos[0],cur_pos[1],cur_pos[2])
                # actor.RotateWXYZ(1, 0.2, 0.2, 0.2)#self.timer_count / 100.0, self.timer_count / 100.0, 0
            iren = obj
            iren.GetRenderWindow().Render()
            self.timer_count += 1
            step += 1
        if self.timerId:
            iren.DestroyTimer(self.timerId)


class Visualizer:

    enumerate

    def critical_u(H):
        return H*np.exp(H + 1)

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

    def add_vert(x, color=None, ymax=310):
        plt.plot([x, x], [0, ymax], linewidth=3, color=color)

    def draw_hist(igraph, mrange=(1, 240), rwidth=1, bins=80, xticks=None):
        degree = np.array([d[1] for d in igraph.degree()], dtype=int)

        hist = np.zeros(shape=(degree.max()+1,), dtype=int)
        for d in degree:
            hist[d] += 1

        plt.hist(degree, bins=bins, histtype='bar',
                 range=mrange, rwidth=rwidth, color='#50ba81')##5081ba
        plt.xticks(xticks, fontsize=18)
        plt.yticks(None, fontsize=18)
        return degree, hist

    def draw_nxvtk(G, node_pos_corr, size_node=0.25, size_edge=0.02, save_pos_path='', scale="full_by_1", **kwargs):
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
        

        # Now create the RenderWindow, Renderer and Interactor
        ren = vtkRenderer()

        positions = None
        corr = None
        if type(dict()) == type(node_pos_corr):
            nums = np.array(list(node_pos_corr.keys()),dtype=int)
            positions = np.zeros(shape=(nums.shape[0], 3), dtype=float)
            corr = np.zeros(shape=(nums.max() + 1,), dtype=int)
            i = 0
            for k in node_pos_corr.keys():
                positions[i, :] = node_pos_corr[k]
                corr[int(k)] = i
                i = i + 1
        else:
            positions = node_pos_corr[0]
            corr = node_pos_corr[1]

        
        if G.nodes[0]['type_id'] is not None and "colors_data" in kwargs:
            data_for_vis = (G, positions, corr, kwargs["colors_data"])
        else:
            data_for_vis = (G, positions, corr)

        focal_pos, camera_pos = Visualizer.draw_graph(ren,data_for_vis,size_node,size_edge,save_pos_path,scale,**kwargs)

        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)

        style = vtkInteractorStyleTrackballCamera()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)

        camera = ren.GetActiveCamera()
        camera.SetFocalPoint(focal_pos[0], focal_pos[1], focal_pos[2])
        camera.SetPosition(camera_pos[0], camera_pos[1], camera_pos[2])

        renWin.Render()
        renWin.Render()
        iren.Initialize()

        if 'animation' in kwargs:
            # Sign up to receive TimerEvent
            cb = vtkTimerCallback(5000, [], [camera], iren)
            iren.AddObserver('TimerEvent', cb.execute)
            cb.timerId = iren.CreateRepeatingTimer(500)

        renWin.Render()
        renWin.SetSize(1900,1080)
        iren.Start()

    def split_view(data1, data2, size_node=0.25, size_edge=0.03, save_pos_path='', scale="full_by_1", **kwargs):
        xmins = [0, .5]
        xmaxs = [0.5, 1]
        ymins = [0]*2
        ymaxs = [1]*2

        rw = vtkRenderWindow()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(rw)
        cameras = []
        for i,data in enumerate([data1,data2]):
            ren = vtkRenderer()

            camera = ren.GetActiveCamera()
            camera.SetFocalPoint(0, 0, 0)
            camera.SetPosition(140, 140, 140)
            cameras.append(camera)

            rw.AddRenderer(ren)
            ren.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])
            Visualizer.draw_graph(ren, data, size_node,size_edge,save_pos_path,scale,**kwargs)


        if 'animation' in kwargs:
            # Sign up to receive TimerEvent
            cb = vtkTimerCallback(5000, [], cameras, iren)
            iren.AddObserver('TimerEvent', cb.execute)
            cb.timerId = iren.CreateRepeatingTimer(500)

        rw.SetSize(1900,1080)
        rw.Render()
        iren.Start()

    def draw_graph(ren, graph_pos_corr, size_node=0.25, size_edge=0.02, save_pos_path='', scale="full_by_1", **kwargs):
        mrange = 1e2
        i = 0

        ndcolors = None
        tdcolors = None
        if len(graph_pos_corr) == 3:
            graph, positions, corr = graph_pos_corr
        elif len(graph_pos_corr) == 4:
            graph, positions, corr, color_data = graph_pos_corr
            ndcolors = [n[1]["type_id"] for n in graph.nodes(data=True)]
        else:
            graph, positions, corr, ndcolors, tdcolors = graph_pos_corr

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
                    d = (positions[:, i] - a_min[i])/dmax
                    positions[:, i] = ((d - d.max()/2))*mrange

        print( f"min = {positions.min(axis=0)}, max = {positions.max(axis=0)}")

        if len(save_pos_path) != 0:
            with open(save_pos_path[0], 'wb') as f:
                np.save(f, positions)
            with open(save_pos_path[1], 'wb') as f:
                np.save(f, corr)
                

        # set node positions
        colors = vtkNamedColors()
        nodePoints = vtkPoints()
        if color_data is not None:
            color_transfer = vtkColorTransferFunction()
            for cd, color in color_data.items():
                color_transfer.AddRGBPoint(cd, color[0], color[1], color[2])     
        
        i = 0
        count_nodes = positions.shape[0]
        pore_data = vtkDoubleArray()
        pore_data.SetNumberOfValues(count_nodes)
        pore_data.SetName("data")
        for (x, y, z) in positions:
            nodePoints.InsertPoint(i, x, y, z)
            if ndcolors is not None:
                pore_data.SetValue(i, ndcolors[i])
            i = i+1
            

        # Create a polydata to be glyphed.
        inputData = vtkPolyData()
        inputData.SetPoints(nodePoints)
        inputData.GetPointData().AddArray(pore_data)

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
        glyphMapper.SetScalarModeToUsePointFieldData()
        glyphMapper.SelectColorArray(pore_data.GetName())
        glyphMapper.SetLookupTable(color_transfer)
        glyphMapper.Update()

        glyph = vtkActor()
        glyph.SetMapper(glyphMapper)
        glyph.GetProperty().SetDiffuseColor(1., 0., 0.)
        glyph.GetProperty().SetSpecular(.3)
        glyph.GetProperty().SetSpecularPower(30)
        

        edgeData = vtkPolyData()
        if size_edge != 0:
            points = vtkPoints()
            lines = vtkCellArray()
            i = 0
            for u, v in graph.edges():
                # The edge e can be a 2-tuple (Graph) or a 3-tuple (Xgraph)
                lines.InsertNextCell(2)
                for n in (u, v):
                    ni = corr[int(n)]
                    (x, y, z) = positions[ni, :]
                    points.InsertPoint(i, x, y, z)
                    lines.InsertCellPoint(i)
                    i = i+1
            edgeData.SetPoints(points)
            edgeData.SetLines(lines)

        # Add thickness to the resulting line.
        Tubes = vtkTubeFilter()
        Tubes.SetNumberOfSides(3)
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

        ren.AddActor(glyph)
        ren.AddActor(profile)
        ren.SetBackground(colors.GetColor3d("White"))
        
        mid = positions.mean(axis=0)
        a_min = positions.min(axis=0)
        a_max = positions.max(axis=0)
        diff = a_max - a_min
        return mid, mid + 2 * diff

    def showGraph(G, size_node=0.25, size_edge=0.02, layout='kamada',save_pos_path='', **kwargs):
        graph = nx.Graph(G)

        # print(f"sqrt={np.sqrt(len(graph.nodes()))}")
        edges = [(i, j, 1) for i, j in graph.edges()]
        graph.add_weighted_edges_from(edges)
        if layout == 'kamada':
            layout = nx.kamada_kawai_layout(graph, dim=3)
        elif layout == 'spring':
            layout = nx.spring_layout(
                graph, dim=3)
        elif layout == 'spectral':
            layout = nx.spectral_layout(graph, dim=3)
        Visualizer.draw_nxvtk(graph, layout, size_node, size_edge, save_pos_path=save_pos_path, **kwargs)

    # 'viridis', 'RdBu', 'Spectral', 'bwr', 'seismic'

    def showRegularResult(data_path, xticks=None, yticks=None, field="lnz", log_callback=None):
        df = xr.load_dataset(data_path)
        res = df[field]

        if log_callback != None:
            log_callback(res)

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        r = res.plot(cmap='seismic')

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
