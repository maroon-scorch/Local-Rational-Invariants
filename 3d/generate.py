import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
# import pylab as pl
from point3 import *
import numpy as np
import math

from surface import intersection_to_squares, square_to_voxel, Trig, Square

def triangle_to_voxel(trig_list):
    ax = a3.Axes3D(plt.figure())
    for sq in trig_list:
        vtx = np.array([sq.p1.vec, sq.p2.vec, sq.p3.vec])
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_alpha(0.5)
        tri.set_color(colors.rgb2hex(np.random.rand(3)))
        
        ax.scatter(sq.p1.x, sq.p1.y, sq.p1.z, c = '#FF0000')
        ax.scatter(sq.p2.x, sq.p2.y, sq.p2.z, c = '#FF0000')
        ax.scatter(sq.p3.x, sq.p3.y, sq.p3.z, c = '#FF0000')
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    plt.show()
    
def plot_points(point_list):
    ax = a3.Axes3D(plt.figure())
    for pt in point_list:
        ax.scatter(pt.x, pt.y, pt.z, c = '#FF0000')
    plt.show()

# Given two numbers x, y, find the integers between them inclusive
def int_between(x, y):
    if x < y:
        return range(math.ceil(x), math.floor(y) + 1)
    else:
        return range(math.ceil(y), math.floor(x) + 1)

def visualize_surface(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, z)
    print(surf.get_figure())
    print(ax.collections[0])

    plt.show()
    
def lst_to_mid(lst):
    edges = []
    for idx, point in enumerate(lst):
        if idx != len(lst) - 1:
            edges.append([point, lst[idx + 1]])
    mid = []
    for s, e in edges:
        mid.append((s + e)/2)
    return mid

def find_intersection(x, y, z, center):
    return []

def generate_mesh():
    R = 4
    r = 2
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2*np.pi, 50)
    # p: u, v
    f_x = lambda p: (R + r*math.cos(p[0]))*math.cos(p[1])
    f_y = lambda p: (R + r*math.cos(p[0]))*math.sin(p[1])
    f_z = lambda p: r*math.sin(p[0])
    px = []
    py = []
    for a in u:
        for b in v:
            px.append(a)
            py.append(b)
            
    indices = Triangulation(px, py).triangles
    
    mesh = []
    for i1, i2, i3 in indices:
        mesh.append([(px[i1], py[i1]), (px[i2], py[i2]), (px[i3], py[i3])])
    
    mesh_3d = list(map(lambda trig: [Point3(f_x(trig[0]), f_y(trig[0]), f_z(trig[0]))
                                     , Point3(f_x(trig[1]), f_y(trig[1]), f_z(trig[1])),
                                     Point3(f_x(trig[2]), f_y(trig[2]), f_z(trig[2]))], mesh))
    return mesh_3d

if __name__ == "__main__":
    # Sphere
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = 2.5 * np.outer(np.cos(u), np.sin(v))
    # y = 2.5 * np.outer(np.sin(u), np.sin(v))
    # z = 2.5 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Torus

    mesh_3d = generate_mesh()
    mesh_triangles = list(map(lambda trig: Trig(trig[0], trig[1], trig[2]), mesh_3d))
    print(len(mesh_triangles))
    
    x_list = list(map(lambda trigs: [trigs.p1.x, trigs.p2.x, trigs.p3.x], mesh_triangles))
    x_list = np.array(x_list).flatten()
    x_int = int_between(np.min(x_list), np.max(x_list))
    x_cen = lst_to_mid(x_int)
    
    y_list = list(map(lambda trigs: [trigs.p1.y, trigs.p2.y, trigs.p3.y], mesh_triangles))
    y_list = np.array(y_list).flatten()
    y_int = int_between(np.min(y_list), np.max(y_list))
    y_cen = lst_to_mid(y_int)
    
    z_list = list(map(lambda trigs: [trigs.p1.z, trigs.p2.z, trigs.p3.z], mesh_triangles))
    z_list = np.array(z_list).flatten()
    z_int = int_between(np.min(z_list), np.max(z_list))
    z_cen = lst_to_mid(z_int)
    
    center = []
    for a in x_cen:
        for b in y_cen:
            for c in z_cen:
                center.append(Point3(a, b, c))

    plot_points(center)
    sq_list = []
    for c in center:
        intersections = find_intersection(mesh_triangles, c)
        squares = intersection_to_squares(intersections)
        sq_list += squares
        
    square_to_voxel(sq_list)
    
    # y_int = int_between(np.min(y), np.max(y))
    # z_int = int_between(np.min(z), np.max(z))

    # x_cen = lst_to_mid(x_int)
    # y_cen = lst_to_mid(y_int)
    # z_cen = lst_to_mid(z_int)
    
    # triangle_to_voxel(mesh_triangles)
    # plt.figure()
    # plt.gca().set_aspect('equal')
    # plt.triplot(triang, 'bo-', lw=1)
    
    # x = np.outer(R + r*np.cos(u), np.cos(v))
    # y = np.outer(R + r*np.cos(u), np.sin(v))
    # z = np.outer(r*np.sin(u), np.ones(np.size(v)))
    
    # x, y, z = x.flatten(), y.flatten(), z.flatten()

    # visualize_surface(x, y, z)
    
    # u, v = symbols('u v')
    # test = plot3d_parametric_surface(
    #     (R + r*cos(u))*cos(v), (R + r*cos(u))*sin(v), r*sin(u),
    # (u, 0, 2 * math.pi), (v, 0, 2*math.pi))
    # print(test)
    # x_int = int_between(np.min(x), np.max(x))
    # y_int = int_between(np.min(y), np.max(y))
    # z_int = int_between(np.min(z), np.max(z))

    # x_cen = lst_to_mid(x_int)
    # y_cen = lst_to_mid(y_int)
    # z_cen = lst_to_mid(z_int)

    # # print(np.max(z))

    # center = []
    # for a in x_int:
    #     for b in y_int:
    #         for c in z_int:
    #             center.append(Point3(a, b, c))

    # plot_points(center)
    # sq_list = []
    # for c in center:
    #     intersections = find_intersection(x, y, z, c)
    #     squares = intersection_to_squares(intersections)
    #     sq_list += squares
        
    # square_to_voxel(sq_list)