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

import sys, itertools

class Square:
    def __init__(self, p1, p2, p3, p4):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.lst = [p1, p2, p3, p4]
        # The String Representation of the value of the Literal
        self.value = str(self.lst)

    def __repr__(self):
        return str(self.lst)

    def __eq__(self, other):
        if type(other) != Square:
            return False
        return self.p1 == other.p1 and self.p2 == other.p2 and self.p3 == other.p3 and self.p4 == other.p4

    def __hash__(self):
      return hash((self.p1, self.p2, self.p3, self.p4))

class Trig:
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.lst = [p1, p2, p3]
        # The String Representation of the value of the Literal
        self.value = str(self.lst)

    def __repr__(self):
        return str(self.lst)

    def __eq__(self, other):
        if type(other) != Square:
            return False
        return self.p1 == other.p1 and self.p2 == other.p2 and self.p3 == other.p3

    def __hash__(self):
      return hash((self.p1, self.p2, self.p3))

def square_to_voxel(square_list):
    ax = a3.Axes3D(plt.figure())
    for sq in square_list:
        vtx = np.array([sq.p1.vec, sq.p2.vec, sq.p3.vec, sq.p4.vec])
        tri = a3.art3d.Poly3DCollection([vtx])
        ax.scatter(sq.p1.x, sq.p1.y, sq.p1.z, c = '#FF0000')
        ax.scatter(sq.p2.x, sq.p2.y, sq.p2.z, c = '#FF0000')
        ax.scatter(sq.p3.x, sq.p3.y, sq.p3.z, c = '#FF0000')
        ax.scatter(sq.p4.x, sq.p4.y, sq.p4.z, c = '#FF0000')
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    plt.show()
    
def triangle_to_voxel(trig_list):
    ax = a3.Axes3D(plt.figure())
    for sq in trig_list:
        vtx = np.array([sq.p1.vec, sq.p2.vec, sq.p3.vec])
        tri = a3.art3d.Poly3DCollection([vtx])
        # tri.set_opacity(0.7)
        # tri.set_color(colors.rgb2hex(np.random.rand(3)))
        tri.set_edgecolor('k')
        ax.scatter(sq.p1.x, sq.p1.y, sq.p1.z, c = '#FF0000')
        ax.scatter(sq.p2.x, sq.p2.y, sq.p2.z, c = '#FF0000')
        ax.scatter(sq.p3.x, sq.p3.y, sq.p3.z, c = '#FF0000')
        ax.add_collection3d(tri)
    plt.show()
    
def plot_points(point_list):
    ax = a3.Axes3D(plt.figure())
    for pt in point_list:
        ax.scatter(pt.x, pt.y, pt.z, c = '#FF0000')
    plt.show()
    
def midpoint_face(current_p1, current_p2):
    mid_x_1 = (math.floor(current_p1.x) + math.ceil(current_p1.x))/2
    mid_y_1 = (math.floor(current_p1.y) + math.ceil(current_p1.y))/2
    mid_z_1 = (math.floor(current_p1.z) + math.ceil(current_p1.z))/2
        
    mid_x_2 = (math.floor(current_p2.x) + math.ceil(current_p2.x))/2
    mid_y_2 = (math.floor(current_p2.y) + math.ceil(current_p2.y))/2
    mid_z_2 = (math.floor(current_p2.z) + math.ceil(current_p2.z))/2

    mp = midpoint_of(current_p1, current_p2)
    
    corner = Point3(math.floor(mp.x), math.floor(mp.y), math.floor(mp.z))
    corner_op = Point3(math.ceil(mp.x), math.ceil(mp.y), math.ceil(mp.z))
    center = midpoint_of(corner, corner_op)
    
    p1_center = Point3(mid_x_1, mid_y_1, mid_z_1)
    p2_center = Point3(mid_x_2, mid_y_2, mid_z_2)
    
    return p1_center, center, p2_center

def visualize_edges(grid_edge_list):
    ax = plt.figure().add_subplot(projection='3d')
    for i, ed in enumerate(grid_edge_list):
        start = ed[0]
        end = ed[1]
        ax.plot([start.x, end.x], [start.y, end.y], [start.z, end.z], 'k-')
        # plt.annotate(i, [(start.x + end.x)/2, (start.y + end.y)/2])
    plt.show()

def face_center_to_cube_center(c1, c2):
    mp = midpoint_of(c1, c2)
        
    corner = Point3(math.floor(mp.x), math.floor(mp.y), math.floor(mp.z))
    corner_op = Point3(math.ceil(mp.x), math.ceil(mp.y), math.ceil(mp.z))
    center = midpoint_of(corner, corner_op)
    
    return center

def is_square(v1, v2, v3, v4):
    standard = [1/2, 1/2, math.sqrt(2)/2]
    dl1 = [dist(v1, v2), dist(v1, v3), dist(v1, v4)]
    dl1.sort()
    
    dl2 = [dist(v2, v1), dist(v2, v3), dist(v2, v4)]
    dl2.sort()
    
    dl3 = [dist(v3, v1), dist(v3, v2), dist(v3, v4)]
    dl3.sort()
    
    dl4 = [dist(v4, v1), dist(v4, v2), dist(v4, v3)]
    dl4.sort()
    
    return dl1 == standard and dl2 == standard and dl3 == standard and dl4 == standard
    

def intersection_to_squares(intersection):
    """ Given a list of intersections contained in a unit voxel,
    convert it to a cubical approximation (roughly) """
    
    if intersection == []:
        return []
    
    cube_list = []
    edge_list = []
    center_list = []
    square_list = []
    for p1, p2 in itertools.combinations(intersection,2):
        c1, c2, c3 = midpoint_face(p1, p2)
        cube_list.append(c1)
        cube_list.append(c2)
        cube_list.append(c3)
        
        edge_list.append([c1, c2])
        edge_list.append([c2, c3])
        center_list.append(c2)
        
    cube_list = list(set(cube_list))
    center_list = list(set(center_list))
    print(cube_list)
    # visualize_edges(edge_list)
    
    # This is a list of vertices of the cubical faces
    cube_list.append(face_center_to_cube_center(center_list[0], center_list[1]))
    
    for v1, v2, v3, v4 in itertools.combinations(cube_list, 4):
        if is_square(v1, v2, v3, v4):
            # Scuffed swap
            new_sq = [v1, v2, v3, v4]
            new_sq.sort(key=lambda p: dist(v1, p))
            temp = new_sq[2]
            new_sq[2] = new_sq[3]
            new_sq[3] = temp      
            ns = Square(new_sq[0], new_sq[1], new_sq[2], new_sq[3])
            print(ns)      
            square_list.append(ns)
    
    
    return square_list
        

def run():
    tri_lst = [Trig(Point3(0, 0, 0), Point3(0, 0, 5), Point3(0, 5, 0))]
    triangle_to_voxel(tri_lst)
    sq_lst = [Square(Point3(0, 0, 0), Point3(0, 0, 1), Point3(0, 1, 1), Point3(0, 1, 0)), Square(Point3(0, 0, 0), Point3(0, 0, -1), Point3(0, -1, -1), Point3(0, -1, 0))]
    square_to_voxel(sq_lst)

# https://stackoverflow.com/questions/4622057/plotting-3d-polygons-in-python-matplotlib
# The main body of the code:
if __name__ == "__main__":
    input = [Point3(0, 0, 0.3), Point3(0, 0.3, 0), Point3(0.3, 0, 0)]
    plot_points(input)
    output = intersection_to_squares(input)
    square_to_voxel(output)
    
    # run()

# n_radii = 8
# n_angles = 36

# # Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
# radii = np.linspace(0.125, 1.0, n_radii)
# angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)[..., np.newaxis]

# # Convert polar (radii, angles) coords to cartesian (x, y) coords.
# # (0, 0) is manually added at this stage,  so there will be no duplicate
# # points in the (x, y) plane.
# x = np.append(0, (radii*np.cos(angles)).flatten())
# y = np.append(0, (radii*np.sin(angles)).flatten())

# # Compute z to make the pringle surface.
# z = np.sin(-x*y)

# ax = plt.figure().add_subplot(projection='3d')

# ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

# plt.show()