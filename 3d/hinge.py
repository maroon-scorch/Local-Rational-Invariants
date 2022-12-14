import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import numpy as np
from surface import Square
from point3 import *

def visualize_square(square_list, vertex):
    ax = a3.Axes3D(plt.figure())
    ax.scatter(vertex.x, vertex.y, vertex.z, c = '#FF0000')
    for sq in square_list:
        vtx = np.array([sq.p1.vec, sq.p2.vec, sq.p3.vec, sq.p4.vec])
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_alpha(0.5)
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    plt.show()
    
if __name__ == "__main__":
    
    # s = set()
    # s.add(Point3(0, 1, 2))
    
    vertex = Point3(-1, -8, -1,)
    squares = [[[ 0, -8, -1,], [-1, -8, -1,], [-1, -9, -1,], [ 0, -9, -1,]],
               [[ 0, -8, -1,], [-1, -8, -1,], [-1, -9, -1,], [ 0, -9, -1,]],
               [[-1, -8, -1,], [-1, -9, -1,], [-2, -9, -1,], [-2, -8, -1,]],
               [[-1, -8, -1,], [-2, -8, -1,], [-2, -9, -1,], [-1, -9, -1,]],
               [[-2, -7, -1,], [-2, -8, -1,], [-1, -8, -1,], [-1, -7, -1,]],
               [[-2, -7, -1,], [-2, -8, -1,], [-1, -8, -1,], [-1, -7, -1,]],
               [[ 0, -8, -1,], [-1, -8, -1,], [-1, -7, -1,], [ 0, -7, -1,]],
               [[ 0, -8, -1,], [-1, -8, -1,], [-1, -7, -1,], [ 0, -7, -1,]],
               [[-1, -7, -2,], [-1, -7, -1,], [-1, -8, -1,], [-1, -8, -2,]],
               [[-1, -7, -2,], [-1, -8, -2,], [-1, -8, -1,], [-1, -7, -1,]]]

    sq_list = []
    
    for sq in squares:
        p1 = Point3(sq[0][0], sq[0][1], sq[0][2])
        p2 = Point3(sq[1][0], sq[1][1], sq[1][2])
        p3 = Point3(sq[2][0], sq[2][1], sq[2][2])
        p4 = Point3(sq[3][0], sq[3][1], sq[3][2])
        
        sq_list.append(Square(p1, p2, p3, p4))
        
    visualize_square(sq_list, vertex)