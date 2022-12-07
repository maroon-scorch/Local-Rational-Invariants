import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
# import pylab as pl
import sys
from point3 import *
from surface import Trig, square_to_voxel
from generate import *
    
# Run program using 'python main.py [directory to file]'
def read_input(inputFile):
    """ Read and parse the input file, returning the list of points and its dimension """
    trig_list = []
    with open(inputFile, "r") as f:
        dimension = int(f.readline())
        for line in f.readlines():
            tokens = line.strip().split()
            p1 = Point3(float(tokens[0]), float(tokens[1]), float(tokens[2]))
            p2 = Point3(float(tokens[3]), float(tokens[4]), float(tokens[5]))
            p3 = Point3(float(tokens[6]), float(tokens[7]), float(tokens[8]))
            tri = Trig(p1, p2, p3)
            trig_list.append(tri)
    return trig_list, dimension

if __name__ == "__main__":
    input_file = sys.argv[1]
    triangles, dimension = read_input(input_file)
    print("Number of Triangles: ", len(triangles))
    triangle_to_voxel(triangles)
    squares = solve(triangles)
    square_to_voxel(squares)