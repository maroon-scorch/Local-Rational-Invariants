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
from surface import Trig, square_to_voxel, visualize_edges_lst
from generate import *
    
# Run program using 'python main.py [directory to file]'
def read_input(inputFile):
    """ Read and parse the input file, returning the list of triangles and its dimension """
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

def clean_input(squares):
    clean_list = []
    for sq in squares:
        new_sq = Square(scale_point3(sq.p1, 2), scale_point3(sq.p2, 2),  scale_point3(sq.p3, 2), scale_point3(sq.p4, 2))
        clean_list.append(new_sq)
    return clean_list

def append_item(dict, square):
    points = square.lst
    for p in points:
        key = str(p)
        if key in dict:
            dict[key].append(square)
        else:
            dict[key] = [p, square]

def directions(vec):
    vec = vec.tolist()
    if vec == [1, 0, 0]:
        return 1
    elif vec == [-1, 0, 0]:
        return 2
    elif vec == [0, 1, 0]:
        return 3
    elif vec == [0, -1, 0]:
        return 4
    elif vec == [0, 0, 1]:
        return 5
    elif vec == [0, 0, -1]:
        return 6
    else:
        return 100

def vert_link(vertex, squares):
    """ Finds the link of the vertex in the complex """
    # print("Vertex: ", vertex)
    vert = vertex.vec
    # print(squares)
    
    # List of opposite edges - if ordered properly
    # this will be the link to the vertex
    edges = []
    for sq in squares:
        center = sq.center.vec
        opp_vert = 2*(center - vert) + vert
        
        # Find neighboring vertex of opp_vert
        neighbors = []
        sign_vector = np.sign(vert - center).tolist()
        for idx, sign in enumerate(sign_vector):
            if sign != 0:
                dir = np.zeros((3,))
                dir[idx,] = sign
                new_pt = opp_vert + dir
                neighbors.append(new_pt)
        
        assert len(neighbors) == 2 
        edges.append([neighbors[0].tolist(), opp_vert.tolist()])
        edges.append([neighbors[1].tolist(), opp_vert.tolist()])
    
    # Unordered list of edges in the sink, retreiving its vertices
    vert_dict = {}
    for start, end in edges:
        key_s = str(start)
        key_e = str(end)
        if key_s in vert_dict:
            vert_dict[key_s].append(end)
        else:
            vert_dict[key_s] = [start, end]
        if key_e in vert_dict:
            vert_dict[key_e].append(start)
        else:
            vert_dict[key_e] = [end, start]
            
    vertices = [vert_dict[key][0] for key in vert_dict.keys()]
    # print("Number of Vertices ", len(vertices))
    order = [directions(np.asarray(vert) - vertex.vec) for vert in vertices]
    # print(order)
    min_index = order.index(min(order))
    # print(min_index)
    min_vertex = vertices[min_index]
    # print("Start ", min_vertex)
    
    next = min_vertex
    path = [min_vertex]
    # Make sure have traversed through the entire graph
    while len(path) != len(vertices):
        pathes = vert_dict[str(next)]
        for dir in pathes:
            if dir not in path:
                path.append(dir)
                next = dir
                break
    # This creates a proper traversal of the link
    # print(path)
    vec_path = []
    for idx, vert in enumerate(path):
        if idx != 0:
            diff = np.asarray(vert) - np.asarray(path[idx - 1])
            vec_path.append(diff)
    vec_path.append(np.asarray(path[0]) - np.asarray(path[-1]))
    # print(vec_path)
    order_path = [directions(elt) for elt in vec_path]
    # print(order_path)
    # print("-------------------------------------------")
    
    # Visualizing Link of Vertex
    # ordered_link = []
    # for idx, vert in enumerate(path):
    #     if idx != len(path) - 1:
    #         ordered_link.append([vert, path[idx + 1]])
    #     else:
    #         ordered_link.append([vert, path[0]])
    # visualize_edges_lst(ordered_link)
    
    return order_path

def order_to_string(order_list):
    output = ""
    for o in order_list:
        output += str(o)
    return output

def dict_to_polynomial(type_dict):
    string = ""
    variables = set()
    for key in type_dict.keys():
        item = str(type_dict[key]) + "*x_" + key + " + "
        string += item
        variables.add("x_" + key)
    num = 0
    # num = turning_number(points)
    string += "0 == "
    string += (str(2) + ",\n")
    return string, variables

def symmetrize_polynomials(variables):
    polynomial_list = []
    var = set()
    for v in variables:
        # print(v)
        var_index = v[2:][::-1]
        dual_variable = "x_"
        for char in var_index:
            c = int(char)
            if c % 2 == 0:
                dual_variable += str(c - 1)
            else:
                dual_variable += str(c + 1)
        var.add(v)
        var.add(dual_variable)
        # print(dual_variable)
        polynomial = v + " == " + dual_variable
        polynomial_list.append(polynomial)
    return polynomial_list, var

def variables_to_sage(variables):
    string = ""
    for v in variables:
        string += v + ", "
    return string

if __name__ == "__main__":
    input_file = sys.argv[1]
    triangles, dimension = read_input(input_file)
    print("Number of Triangles: ", len(triangles))
    # triangle_to_voxel(triangles)
    squares = solve(triangles)
    squares = clean_input(squares)
    # print(squares)
    square_to_voxel(squares)
    
    # Create a vertex dictionary and squares it is contained in
    vert_dict = {}
    for sq in squares:
        append_item(vert_dict, sq)
    
    type_dict = {}
    for k in vert_dict.keys():
        head = vert_dict[k][0]
        body = vert_dict[k][1:]
        order_list = vert_link(head, body)
        vertex_type = order_to_string(order_list)
        
        if vertex_type in type_dict:
            type_dict[vertex_type] += 1
        else:
            type_dict[vertex_type] = 1
    
    polynomial, variables = dict_to_polynomial(type_dict)
    print(polynomial)
    print(variables)
    print("-------------------------------")
    polynomial_list, variables = symmetrize_polynomials(variables)
    for p in polynomial_list:
        print(p)
    print(variables_to_sage(variables))