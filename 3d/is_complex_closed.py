from surface import Square, square_to_voxel
from point3 import *
import sys
import numpy as np
from euler import vert_link, is_2_regular, is_connected

def append_item_alt(dict, square):
    points = square.lst
    for p in points:
        key = p
        if key in dict:
            dict[key].append(square)
        else:
            dict[key] = [square]

def load_squares(input_file):
    sq_list = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            p1 = Point3(float(tokens[0]), float(tokens[1]), float(tokens[2]))
            p2 = Point3(float(tokens[3]), float(tokens[4]), float(tokens[5]))
            p3 = Point3(float(tokens[6]), float(tokens[7]), float(tokens[8]))
            p4 = Point3(float(tokens[9]), float(tokens[10]), float(tokens[11]))
            sq = Square(p1, p2, p3, p4)
            sq_list.append(sq)
    
    return sq_list

def is_point_bad(vertex, squares):
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
    return not(is_2_regular(vert_dict) and is_connected(vert_dict))

if __name__ == "__main__":
    input_file = sys.argv[1]
    square_list = load_squares(input_file)
    
    vert_dict = {}
    for sq in square_list:
        append_item_alt(vert_dict, sq)
        
    type_dict = {}
    for k in vert_dict.keys():
        if is_point_bad(k, vert_dict[k]):
            print(k)
        