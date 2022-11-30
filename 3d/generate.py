import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
# import pylab as pl
from point3 import *
import numpy as np
import math

from sympy import symbols
from sympy.plotting import plot3d

from surface import intersection_to_squares, square_to_voxel


from statistics import mean

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
    ax.plot_surface(x, y, z)
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

if __name__ == "__main__":
    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 2.5 * np.outer(np.cos(u), np.sin(v))
    y = 2.5 * np.outer(np.sin(u), np.sin(v))
    z = 2.5 * np.outer(np.ones(np.size(u)), np.cos(v))

    visualize_surface(x, y, z)

    x_int = int_between(np.min(x), np.max(x))
    y_int = int_between(np.min(y), np.max(y))
    z_int = int_between(np.min(z), np.max(z))

    x_cen = lst_to_mid(x_int)
    y_cen = lst_to_mid(y_int)
    z_cen = lst_to_mid(z_int)

    print(np.max(z))

    center = []
    for a in x_int:
        for b in y_int:
            for c in z_int:
                center.append(Point3(a, b, c))

    plot_points(center)
    sq_list = []
    for c in center:
        intersections = find_intersection(x, y, z, c)
        squares = intersection_to_squares(intersections)
        sq_list += squares
        
    square_to_voxel(sq_list)