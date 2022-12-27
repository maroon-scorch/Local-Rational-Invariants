from surface import Square, square_to_voxel
from point3 import *
import math, itertools, random, sys
import numpy as np
from euler import vert_link, order_to_string, dict_to_polynomial, clean_input
from rigid_motion import apply_rigid_motion, translate, integerify_square

# ----------------------------------------------------------------------------------
# def append_item(dict, square):
#     points = square.lst
#     for p in points:
#         key = str(p)
#         if key in dict:
#             dict[key].append(square)
#         else:
#             dict[key] = [p, square]

# def squares_to_polynomials(squares, num):
#     vert_dict = {}
#     for sq in squares:
#         append_item(vert_dict, sq)
#     # Create a vertex dictionary and squares it is contained in
#     type_dict = {}
#     for k in vert_dict.keys():
#         head = vert_dict[k][0]
#         body = vert_dict[k][1:]
#         order_list = vert_link(head, body)
#         vertex_type = order_to_string(order_list)
        
#         if vertex_type in type_dict:
#             type_dict[vertex_type] += 1
#         else:
#             type_dict[vertex_type] = 1
    
#     polynomial, variables = dict_to_polynomial(type_dict, num)
#     return polynomial, variables


# ----------------------------------------------------------------------------------

def is_square_alt(v1, v2, v3, v4):
    standard = [1, 1, math.sqrt(2)]
    dl1 = [dist(v1, v2), dist(v1, v3), dist(v1, v4)]
    dl1.sort()
    
    dl2 = [dist(v2, v1), dist(v2, v3), dist(v2, v4)]
    dl2.sort()
    
    dl3 = [dist(v3, v1), dist(v3, v2), dist(v3, v4)]
    dl3.sort()
    
    dl4 = [dist(v4, v1), dist(v4, v2), dist(v4, v3)]
    dl4.sort()
    
    return dl1 == standard and dl2 == standard and dl3 == standard and dl4 == standard

def point_cloud_to_squares(point_list):
    square_list = []
    for v1, v2, v3, v4 in itertools.combinations(point_list, 4):
        if is_square_alt(v1, v2, v3, v4):
            # Scuffed swap
            new_sq = [v1, v2, v3, v4]
            new_sq.sort(key=lambda p: dist(v1, p))
            temp = new_sq[2]
            new_sq[2] = new_sq[3]
            new_sq[3] = temp      
            ns = Square(new_sq[0], new_sq[1], new_sq[2], new_sq[3])
            # print(ns)      
            square_list.append(ns)
    return square_list

# NEED TO OPTIMIZE THIS later
def generate_cube(a, b, c):
    point_list = [[], [], [], [], [], []]
    square_list = []
    vertice = []
    for i in range(0, b + 1):
        for j in range(0, c + 1):
            point_list[0].append(Point3(0, i, j))
            point_list[1].append(Point3(a, i, j))
            
    for i in range(0, a + 1):
        for j in range(0, c + 1):
            point_list[2].append(Point3(i, 0, j))
            point_list[3].append(Point3(i, b, j))
            
    for i in range(0, a + 1):
        for j in range(0, b + 1):
            point_list[4].append(Point3(i, j, 0))
            point_list[5].append(Point3(i, j, c))    
    
    for i in range(6):
        vertice = vertice + point_list[i]
        square_list += point_cloud_to_squares(point_list[i])

    center = np.asarray([0, 0, 0])
    for p in vertice:
        center += p.vec
    center = center / len(vertice)
    
    return square_list, vertice, center

def append_item_alt(dict, square):
    points = square.lst
    for p in points:
        key = p
        if key in dict:
            dict[key].append(square)
        else:
            dict[key] = [square]

def sq_to_axis(square):
    """ Given a square on the grid, find out which axis it is perpendicular to:
    0 - x, 1 - y, 2 - z"""
    point, center = square.p1.vec, square.p3.vec
    diff = (center - point).tolist()
    for idx, elt in enumerate(diff):
        if elt == 0:
            return idx

def push_square(square, axis, direction):
    vector = [0, 0, 0]
    vector[axis] = direction[axis]
    vector = np.asarray(vector)
        
    s1 = to_point(square.p1.vec + vector)
    s2 = to_point(square.p2.vec + vector)
    s3 = to_point(square.p3.vec + vector)
    s4 = to_point(square.p4.vec + vector)
    
    return Square(s1, s2, s3, s4)


def is_2_regular_alt(graph):
    """ Checks that the graph is two regular """
    for key in graph.keys():
        if len(graph[key][1]) != 2:
            return False
    return True

def is_vertex_problematic(vertex, squares):
    """ Finds if the vertex is problematic or not """
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
    
    return not is_2_regular_alt(vert_dict)

def find_problematic_vertice(square_list):
    # Cleaning out problematic vertices 
    vert_dict = {}
    for sq in square_list:
        append_item_alt(vert_dict, sq)
    
    problematic_vertice = []
    for vert in vert_dict.keys():
        if is_vertex_problematic(vert, vert_dict[vert]):
            problematic_vertice.append(vert)
    
    return problematic_vertice

# ------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------

def remove_vertex(square_list, vertex):
    neighbor = []
    index_list = []
    for idx, sq in enumerate(square_list):
        if vertex in sq.lst:
            index_list.append(idx)
            neighbor.append(sq)
    index_list.reverse()
    for i in index_list:
        square_list.pop(i)
    
    
    center = np.asarray([0, 0, 0])
    neighbor_vert = set()
    for sq in neighbor:
        neighbor_vert.add(sq.p1)
        neighbor_vert.add(sq.p2)
        neighbor_vert.add(sq.p3)
        neighbor_vert.add(sq.p4)
    neighbor_vert.remove(vertex)
    for vert in neighbor_vert:
        center += vert.vec
    center = center / len(neighbor_vert)
    
    direction = np.sign(center - vertex.vec)
    
    for sq in neighbor:
        axis = sq_to_axis(sq)
        square_list.append(push_square(sq, axis, direction))
    

    problematic_vertice = find_problematic_vertice(square_list)
    # Potentially faulty, need to check later    
    problematic_squares = point_cloud_to_squares(problematic_vertice)
    # square_to_voxel(problematic_squares)
    
    pop_list = []
    for sq in problematic_squares:
        if sq in square_list:
            square_list = [square for square in square_list if square != sq]
        else:
            square_list.append(sq)
    
    
    if find_problematic_vertice(square_list) != []:
        print(vertex)
        print(direction)
        square_to_voxel(square_list)
    
    assert find_problematic_vertice(square_list) == []
    
    
    return square_list
    # print(neighbor)

def scale_square_xyz(square_list, x_factor, y_factor, z_factor):    
    output = []
    factor_list = [x_factor, y_factor, z_factor]
    x_center = np.asarray([0, y_factor/2, z_factor/2])
    y_center = np.asarray([x_factor/2, 0, z_factor/2])
    z_center = np.asarray([x_factor/2, y_factor/2, 0])
    center_list = [x_center, y_center, z_center]
    point_list = [[], [], []]
    
    for i in range(0, y_factor + 1):
        for j in range(0, z_factor + 1):
            point_list[0].append(Point3(0, i, j))

    for i in range(0, x_factor + 1):
        for j in range(0, z_factor + 1):
            point_list[1].append(Point3(i, 0, j))
            
    for i in range(0, x_factor + 1):
        for j in range(0, y_factor + 1):
            point_list[0].append(Point3(i, j, 0))

    point_list[0] = point_cloud_to_squares(point_list[0])
    point_list[1] = point_cloud_to_squares(point_list[1])
    point_list[2] = point_cloud_to_squares(point_list[2])
    
    for sq in square_list:
        diff = (sq.p1.vec - sq.center.vec).tolist()
        axis = diff.index(0)
        
        vector = (factor_list[axis]*sq.center.vec - center_list[axis]).tolist()
        new_squares = translate(point_list[axis], vector[0], vector[1], vector[2])
        output += new_squares
    output = integerify_square(output)
    return output


def load_squares(input_file):
    sq_list = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            p1 = Point3(int(tokens[0]), int(tokens[1]), int(tokens[2]))
            p2 = Point3(int(tokens[3]), int(tokens[4]), int(tokens[5]))
            p3 = Point3(int(tokens[6]), int(tokens[7]), int(tokens[8]))
            p4 = Point3(int(tokens[9]), int(tokens[10]), int(tokens[11]))
            sq = Square(p1, p2, p3, p4)
            sq_list.append(sq)
    
    vertice = set()
    for sq in sq_list:
        vertice.add(sq.p1)
        vertice.add(sq.p2)
        vertice.add(sq.p3)
        vertice.add(sq.p4)
    vertice = list(vertice)
    
    center = np.asarray([0, 0, 0])
    for p in vertice:
        center += p.vec
    center = center / len(vertice)
    
    return sq_list, vertice, center

if __name__ == "__main__":
    
    input_file = sys.argv[1]
    
    # square_to_voxel(square_list)
    
    file = open("temp1.txt", "w+")
    iter = 1
    for i in range(iter):
        print("Iteration: ", i)
        square_list, vertice, center = load_squares(input_file)
        vertex = random.choice(vertice)
        square_list = remove_vertex(square_list, vertex)
        square_to_voxel(square_list)
        # square_list = scale_square_xyz(square_list, random.randint(2, 5))
        rigid_polynomials = apply_rigid_motion(square_list, 8, 0)
        for r in rigid_polynomials:
           file.write(r)
        
        vert_dict = {}
        for sq in square_list:
            append_item_alt(vert_dict, sq)
        
        type_dict = {}
        for k in vert_dict.keys():
            order_list = vert_link(k, vert_dict[k])
            vertex_type = order_to_string(order_list)
            
            if vertex_type in type_dict:
                type_dict[vertex_type] += 1
            else:
                type_dict[vertex_type] = 1
        
        # print(type_dict)
            
        polynomial, variables = dict_to_polynomial(type_dict, 0)
        # print(polynomial)
        file.write(polynomial)
    
    file.close()