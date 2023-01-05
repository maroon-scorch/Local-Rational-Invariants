import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
# import pylab as pl
import random
from point3 import *
from surface import Trig, square_to_voxel, visualize_edges_lst
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

def is_2_regular(graph):
    """ Checks that the graph is two regular """
    for key in graph.keys():
        if len(graph[key][1]) != 2:
            return False
    return True

def is_connected(graph):
    """ Checks that the graph is connected """
    return True

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
            vert_dict[key_s][1].add(to_point(end))
        else:
            vert_dict[key_s] = (start, {to_point(end)})
        if key_e in vert_dict:
            vert_dict[key_e][1].add(to_point(start))
        else:
            vert_dict[key_e] = (end, {to_point(start)})
    # print("Dictionary: ", vert_dict)
    assert is_2_regular(vert_dict) and is_connected(vert_dict), "ERROR: Vertex Link at " + str(vertex) + " has to be a connected cycle graph --------------------------------"
    
    vertices = [vert_dict[key][0] for key in vert_dict.keys()]
    # print("Number of Vertices ", len(vertices))
    order = [directions(np.asarray(vert) - vertex.vec) for vert in vertices]
    # print(order)
    min_index = order.index(min(order))
    # print(min_index)
    min_vertex = vertices[min_index]
    # print(vert_dict)
    # print("Start ", min_vertex)
    
    next = min_vertex
    path = [min_vertex]
    
    # Make sure have traversed through the entire graph
    count = 0
    while count < len(vertices):
        count = count + 1
        pathes = vert_dict[str(next)][1]
        for dir in pathes:
            dir1 = to_list(dir)
            if dir1 not in path:
                path.append(dir1)
                next = dir1
                break
    if len(path) != len(vertices):
        assert False, "ERROR: The Vertex Link is not a Cycle Graph! --------------------------------------------"

    
    # Alternating encoding of vertex:
    path.append(min_vertex)
    vec_path = []
    for v in path:
        # print(v)
        diff = np.asarray(v) - vertex.vec
        vec_path.append(diff)
    order_path = [directions(elt) for elt in vec_path]
    order_path = [i for i in order_path if i != 100]
    
    # This creates a proper traversal of the link
    # # print(path)
    # vec_path = []
    # for idx, vert in enumerate(path):
    #     if idx != 0:
    #         diff = np.asarray(vert) - np.asarray(path[idx - 1])
    #         vec_path.append(diff)
    # vec_path.append(np.asarray(path[0]) - np.asarray(path[-1]))
    # # print(vec_path)
    # order_path = [directions(elt) for elt in vec_path]
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

def dict_to_polynomial(type_dict, num):
    string = ""
    variables = set()
    for key in type_dict.keys():
        item = str(type_dict[key]) + "*x_" + key + " + "
        string += item
        variables.add("x_" + key)
    # num = turning_number(points)
    string += "0 == "
    string += (str(num) + ",\n")
    return string, variables

# For path variables
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
        polynomial += ",\n"
        polynomial_list.append(polynomial)
    return polynomial_list, var

def symmetrize_polynomials_alt(variables):
    polynomial_list = []
    var = set()
    for v in variables:
        # print(v)
        var_index = v[2:][::-1]
        dual_variable = "x_"
        for char in var_index:
            dual_variable += char
        var.add(v)
        var.add(dual_variable)
        # print(dual_variable)
        polynomial = v + " == " + dual_variable
        polynomial += ",\n"
        polynomial_list.append(polynomial)
    return polynomial_list, var

def variables_to_sage(variables):
    string = ""
    for v in variables:
        string += v + ", "
    return string

def generate_T2(R, r):
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, 2*np.pi, 10)
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
    
    m_triangles = list(map(lambda trig: Trig(trig[0], trig[1], trig[2]), mesh_3d))
    mesh_triangles = []
    for trg in m_triangles:
        if not is_triangle_degenrate(trg):
            mesh_triangles.append(trg)
    return mesh_triangles

def generate_s2(A, B):
    r = A
    u = np.linspace(0, 2*np.pi, 8)
    v = np.linspace(0, np.pi, 8)
    
    # p: u, v
    f_x = lambda p: r*math.cos(p[0])*math.sin(p[1])
    f_y = lambda p: r*math.sin(p[0])*math.sin(p[1])
    f_z = lambda p: B*math.cos(p[1])
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
    
    m_triangles = list(map(lambda trig: Trig(trig[0], trig[1], trig[2]), mesh_3d))
    mesh_triangles = []
    for trg in m_triangles:
        if not is_triangle_degenrate(trg):
            mesh_triangles.append(trg)
    
    return mesh_triangles

def squares_to_polynomials(squares, num):
    vert_dict = {}
    for sq in squares:
        append_item(vert_dict, sq)
    # Create a vertex dictionary and squares it is contained in
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
    
    polynomial, variables = dict_to_polynomial(type_dict, num)
    return polynomial, variables

if __name__ == "__main__":
    iter = 50
    all_variables = set()
    file = open("polynomial.txt", "w+")
    
    for i in range(0): 
        
        print("Iteration: ", i)
        try:
            A = random.uniform(0.5, 5)
            B = random.uniform(0.5, 5)

            triangles = generate_s2(A, B)
            print("A: ", A)
            print("B: ", B)
            
            print("Number of Triangles: ", len(triangles))
            # triangle_to_voxel(triangles)
            squares = solve(triangles)
            squares = clean_input(squares)
            # square_to_voxel(squares)
            polynomial, variables = squares_to_polynomials(squares, 2)

            file.write(polynomial)
            all_variables.update(variables)
        except Exception as e:
            print(e)
            print("------------------------------------------")
            continue
        
    for i in range(iter):
        print("Iteration: ", i + iter)
        try:       
            R = random.uniform(2.5, 5)
            r = random.uniform(0.5, R/4)
            triangles = generate_T2(R, r)
            
            print("Number of Triangles: ", len(triangles))
            # triangle_to_voxel(triangles)
            squares = solve(triangles)
            squares = clean_input(squares)
            # square_to_voxel(squares)
            polynomial, variables = squares_to_polynomials(squares, 0)

            file.write(polynomial)
            all_variables.update(variables)
        except Exception as e:
            print(e)
            print("------------------------------------------")
            continue

    polynomial_list, all_variables = symmetrize_polynomials_alt(all_variables)
    for p in polynomial_list:
        file.write(p)
        
    sage_export = variables_to_sage(all_variables)
    file.write(sage_export)