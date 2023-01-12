from surface import Square, square_to_voxel
from point3 import *
import math, itertools, random, sys
import numpy as np
from euler import vert_link, order_to_string, dict_to_polynomial
from corner_cube import point_cloud_to_squares, generate_cube, remove_vertex

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
    
    return sq_list, vertice

def translate(square_list, x, y, z):
    new_squares = []
    vector = np.asarray([x, y, z])
    for sq in square_list:
        s1 = to_point(sq.p1.vec + vector)
        s2 = to_point(sq.p2.vec + vector)
        s3 = to_point(sq.p3.vec + vector)
        s4 = to_point(sq.p4.vec + vector)
        new_squares.append(Square(s1, s2, s3, s4))
        
    return new_squares

def rotate(array, theta):
    original_point = array.tolist()
    new_point = array.tolist()
    
    new_point[0] = original_point[0]*math.cos(theta) - original_point[2]*math.sin(theta)
    new_point[2] = original_point[0]*math.sin(theta) + original_point[2]*math.cos(theta)
    
    return np.asarray(new_point)

def rotate_phi(array, phi):
    original_point = array.tolist()
    new_point = array.tolist()
    
    new_point[1] = original_point[1]*math.cos(phi) - original_point[2]*math.sin(phi)
    new_point[2] = original_point[1]*math.sin(phi) + original_point[2]*math.cos(phi)
    
    return np.asarray(new_point)
    
    
def integerify_square(squares):
    clean_list = []
    for sq in squares:
        new_sq = Square(round_p3(sq.p1), round_p3(sq.p2),  round_p3(sq.p3), round_p3(sq.p4))
        clean_list.append(new_sq)
    return clean_list   

def rotate_squares_theta(square_list, theta):
    new_squares = []
    for sq in square_list:
        s1 = to_point(rotate(sq.p1.vec, theta))
        s2 = to_point(rotate(sq.p2.vec, theta))
        s3 = to_point(rotate(sq.p3.vec, theta))
        s4 = to_point(rotate(sq.p4.vec, theta))
        new_squares.append(Square(s1, s2, s3, s4))
        
    return new_squares

def rotate_squares_phi(square_list, phi):
    new_squares = []
    for sq in square_list:
        s1 = to_point(rotate_phi(sq.p1.vec, phi))
        s2 = to_point(rotate_phi(sq.p2.vec, phi))
        s3 = to_point(rotate_phi(sq.p3.vec, phi))
        s4 = to_point(rotate_phi(sq.p4.vec, phi))
        new_squares.append(Square(s1, s2, s3, s4))
        
    return new_squares

def remove_duplicate(square_list):
    output = []
    for sq in square_list:
        if sq not in output:
            output.append(sq)
    return output

def remove_repeat_squares(square_list):
    output = []
    remove = set()
    
    for sq in square_list:
        if sq not in output:
            output.append(sq)
        else:
            remove.add(sq)
    
    for r in remove:
        # print(r)
        output = [o for o in output if o != r]
    
    return output

def scale_square(square_list, factor):
    assert int(factor) == factor and factor > 0
    
    output = []
    x_center = np.asarray([0, factor/2, factor/2])
    y_center = np.asarray([factor/2, 0, factor/2])
    z_center = np.asarray([factor/2, factor/2, 0])
    center_list = [x_center, y_center, z_center]
    point_list = [[], [], []]
    for i in range(0, factor + 1):
        for j in range(0, factor + 1):
            point_list[0].append(Point3(0, i, j))
            point_list[1].append(Point3(i, 0, j))
            point_list[2].append(Point3(i, j, 0))
    point_list[0] = point_cloud_to_squares(point_list[0])
    point_list[1] = point_cloud_to_squares(point_list[1])
    point_list[2] = point_cloud_to_squares(point_list[2])
    
    for sq in square_list:
        diff = (sq.p1.vec - sq.center.vec).tolist()
        axis = diff.index(0)
        
        vector = (factor*sq.center.vec - center_list[axis]).tolist()
        new_squares = translate(point_list[axis], vector[0], vector[1], vector[2])
        output += new_squares
    output = integerify_square(output)
    return output


# --------------------------------------
# Removing a vertex

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

def is_vertex_problematic(vertex, squares):
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
    for key in vert_dict.keys():
        if len(vert_dict[key][1]) != 2:
            return True
    return False

def remove_vertex_local(square_list, vertex, direction):
    neighbor = []
    index_list = []
    for idx, sq in enumerate(square_list):
        if vertex in sq.lst:
            index_list.append(idx)
            neighbor.append(sq)
    
    index_list.reverse()
    for i in index_list:
        square_list.pop(i)
    
    print(direction)
    square_to_voxel(neighbor)
    
    for sq in neighbor:
        axis = sq_to_axis(sq)
        square_list.append(push_square(sq, axis, direction))
    
    square_to_voxel(square_list)
    
    for i in range(2):
        vert_dict = {}
        for sq in square_list:
            append_item_alt(vert_dict, sq)
        
        problematic_vertice = []
        for v in vert_dict.keys():
            if is_vertex_problematic(v, vert_dict[v]):
                problematic_vertice.append(v)
        
        # problematic_vertice = find_problematic_vertice(square_list)
        # Potentially faulty, need to check later    
        problematic_squares = point_cloud_to_squares(problematic_vertice)
        for sq in problematic_squares:
            if sq in square_list:
                square_list = [square for square in square_list if square != sq]
            else:
                square_list.append(sq)
    
    # if find_problematic_vertice(square_list) != []:
    #     print(vertex)
    #     print(direction)
    #     square_to_voxel(square_list)
    
    # assert find_problematic_vertice(square_list) == []
    
    
    return square_list
    # print(neighbor)

if __name__ == "__main__":
    angle = [0, math.pi/2, 3*math.pi/2, math.pi]
    
    # cube, _, _ = generate_cube(1, 1, 19)
    # cube1, _, _ = generate_cube(1, 1, 10)
    hedge, _, _ = generate_cube(1, 1, 10)
    h2, _, _ = generate_cube(10, 1, 1)
    h3, _, _ = generate_cube(1, 10, 1)
    
    squares = []
    
    unit, _, _ = generate_cube(1, 1, 1)
    squares = unit + translate(unit, 1, 0, 0)
    # This is how you glue
    squares = remove_repeat_squares(squares)
    
    
    # Crystal Structure
    # for i in range(10):
    #     squares += translate(hedge, i, 0, 0)
    #     squares = remove_repeat_squares(squares)
    # for i in range(10):
    #     squares += translate(h2, 1, i, -1)
    #     squares = remove_repeat_squares(squares)
    # for i in range(10):
    #     squares += translate(h3, 1, 1, i)
    #     squares = remove_repeat_squares(squares)
    
    # squares += translate(hedge, 0, 0, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(hedge, 0, 0, -10)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(h2, 1, 0, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(h2, -10, 0, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(h3, 0, 1, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(h3, 0, -10, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(squares, 21, 0, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(squares, 0, 0, 0)
    # squares = remove_repeat_squares(squares)
    
    # Two examples:
    # squares += translate(h2, 0, 0, 0)
    # squares += translate(h3, 0, 1, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(h3, 9, 1, 0)
    # squares = remove_repeat_squares(squares)
    
    # cop = integerify_square(rotate_squares_phi(squares, math.pi/2))
    # squares += translate(cop, 0, 12, -10)
    # squares = remove_repeat_squares(squares)
    
    # c1, _, _ = generate_cube(3, 3, 1)
    # squares += c1 + translate(c1, 0, 0, 20)
    # squares += translate(cube, 1, 0, 1)
    # squares += translate(cube1, 1, 1, 4)
    # squares += translate(cube, 1, 2, 1)
    # squares = remove_repeat_squares(squares)
    
    # Torus connected by a string:
    # squares += translate(h2, 0, 0, 0)
    # squares += translate(h3, 0, 1, 0)
    # squares = remove_repeat_squares(squares)
    # squares += translate(h3, 9, 1, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(h2, 0, 11, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(squares, 6, 8, 1)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(squares, 0, 0, 1) + translate(squares, 0, 0, 2)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(squares, 11, 0, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(hedge, -1, 5, 2)
    # squares = remove_repeat_squares(squares)
    # squares += translate(hedge, 21, 5, 2)
    # squares = remove_repeat_squares(squares)
    
    # c4, _, _ = generate_cube(21, 1, 1)
    # squares += translate(c4, 0, 5, 11)
    # squares = remove_repeat_squares(squares)
    
    # Stair case:
    # for i in range(5):
    #     for j in range(i):
    #         squares += translate(h3, i, 0, j)
    #         squares = remove_repeat_squares(squares)
            
    # for i in range(11):
    #     squares += translate(h3, i, 0, i)
    #     squares = remove_repeat_squares(squares)
    #     squares += translate(h3, i, 0, i+1)
    #     squares = remove_repeat_squares(squares)
    # squares += translate(h2, 1, 2, 0)
    # squares = remove_repeat_squares(squares)
    # squares += translate(hedge, 10, 2, 0)
    # squares = remove_repeat_squares(squares)
    # squares += translate(h3, 9, 1, 0)
    # squares = remove_repeat_squares(squares)
    
    # cop = integerify_square(rotate_squares_phi(squares, math.pi/2))
    # squares += translate(cop, 0, 12, -10)
    # squares = remove_repeat_squares(squares)
    
    # Cube
    # unit, _, _ = generate_cube(1, 1, 1)
    # squares += unit
    # squares += translate(unit, 1, 0, 0) + translate(unit, 2, 0, 0)
    # squares += translate(unit, 0, 2, 0) + translate(unit, 1, 2, 0) + translate(unit, 2, 2, 0)
    # squares += translate(unit, 0, 1, 0) + translate(unit, 2, 1, 0)
    # squares += translate(unit, 0, 0, 1) + translate(unit, 1, 0, 1) + translate(unit, 2, 0, 1)
    # squares += translate(unit, 0, 2, 1) + translate(unit, 1, 2, 1) + translate(unit, 2, 2, 1)
    # squares += translate(unit, 0, 0, 2) + translate(unit, 1, 0, 2) + translate(unit, 2, 0, 2)
    # squares += translate(unit, 0, 2, 2) + translate(unit, 1, 2, 2) + translate(unit, 2, 2, 2)
    # #squares += translate(unit, 0, 1, 1) + translate(unit, 2, 1, 1)
    # squares += translate(unit, 0, 1, 2) + translate(unit, 2, 1, 2)
    # # squares += translate(unit, 1, 1, 1)
    # squares = remove_repeat_squares(squares)
    
    # Torus with corner digged
    # r1, _, _ = generate_cube(3, 1, 1)
    # unit, _, _ = generate_cube(1, 1, 1)
    # squares += r1 + translate(r1, 0, 2, 0)
    # # squares = remove_repeat_squares(squares)
    # squares += translate(unit, 0, 1, 0) + translate(unit, 2, 1, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(unit, 0, 0, 1) + translate(unit, 2, 0, 1) + translate(unit, 0, 2, 1) + translate(unit, 2, 2, 1)
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(unit, 1, 1, 0)
    # squares = remove_repeat_squares(squares)
    # squares = scale_square(squares, 2)
    # square_to_voxel(squares)
    # squares = remove_vertex_local(squares, Point3(2, 2, 0), np.asarray([-1, -1, 1]))
    
    
    # Stacking a torus inside and out
    # unit, _, _ = generate_cube(1, 1, 1)
    
    # r1, _, _ = generate_cube(3, 1, 1)
    # unit, _, _ = generate_cube(1, 1, 1)
    # squares += r1 + translate(r1, 0, 2, 0)
    # squares = remove_repeat_squares(squares)
    # squares += translate(unit, 0, 1, 0) + translate(unit, 2, 1, 0)
    # squares = remove_repeat_squares(squares)
    
    # squares = scale_square(squares, 3)
    # squares = remove_repeat_squares(squares)
    
    # r2, _, _ = generate_cube(1, 1, 2)
    # # squares += translate(r2,3, 3, 1)
    # squares += translate(r2, 4, 3, 2)
    # squares += translate(r2, 4, 4, 2)
    # # squares += translate(r2, 4, 5, 2)
    # # squares += translate(r2, 3, 4, 2)
    # # squares += translate(r2, 5, 4, 2)
    # squares += translate(r2, 3, 4, 2)
    # # squares += translate(r2, 3, 5, 2)
    # # squares += translate(r2, 5, 3, 2)
    # # squares += translate(r2, 5, 5, 2)
    # squares = remove_repeat_squares(squares)
    
    # Punctured Ball
    # c3, _, _ = generate_cube(3, 3, 3)
    # c5, _, center = generate_cube(4, 5, 4)
    # unit, _, _ = generate_cube(1, 1, 1)
    
    # squares += c5 # + translate(c3, 1, 1, 1)
    # squares = remove_repeat_squares(squares)

    # for i in range(3):
    #     vert_dict = {}
    #     for sq in squares:
    #         append_item_alt(vert_dict, sq)
    #     squares = remove_vertex(squares, random.choice(list(vert_dict.keys())), center)
    
    # c3, _, _ = generate_cube(1, 5, 5)
    # c4, _, _ = generate_cube(1, 5, 2)
    # c5, _, _ = generate_cube(2, 1, 1)
    # c6, _, _ = generate_cube(1, 1, 7)
    # squares += c3
    # squares += translate(c3, 1, 0, 4)
    # squares += translate(c4, 1, 5, 4)
    # squares += translate(c5, 1, 1, 0)
    # squares += translate(c6, 2, 0, 0)
    
    # squares = remove_repeat_squares(squares)
    
    # -----------------------------
    # Twisted Torus
    # squares += translate(hedge, -1, 1, 0)
    # squares += translate(hedge, 5, 0, 0)
    
    # squares += translate(h2, 3, -1, 0)
    # squares += translate(h2, 0, 1, 7)
    
    # d, _, _ = generate_cube(1, 4, 8)
    
    # squares = remove_repeat_squares(squares)
    
    # squares += translate(d, 10, -2, 3)
    # squares = remove_repeat_squares(squares)
    
    # -------------
    # Torus with hole at same direction
    # unit, _, _ = generate_cube(1, 1, 1)
    # c3, _, _ = generate_cube(3, 3, 3)
    # c5, _, _ = generate_cube(5, 5, 5)
    
    # squares += translate(c3, 1, 1, 1) + c5
    # squares += translate(unit, 1, 2, 4)
    # squares += translate(unit, 3, 3, 4)
    # squares = remove_repeat_squares(squares)
    
    # f, _, _ = generate_cube(3, 10, 1)
    # unit, _, _ = generate_cube(1, 1, 1)
    # squares += translate(h3, 0, 0, 1) # + translate(h3, 2, 0, 1)
    # squares += translate(unit, 0, 0, 0) + translate(unit, 0, 2, 0) + translate(unit, 0, 4, 0) + translate(unit, 0, 6, 0) + translate(unit, 0, 8, 0)
    # squares += translate(h3, 1, 0, 0)
    # squares += translate(h3, 1, 0, 1)
    # squares = remove_repeat_squares(squares)
    # squares = remove_repeat_squares(squares)
    
    
    # CASTLE
    # base, _, _ = generate_cube(5, 5, 1)
    # g, _, _ = generate_cube(3, 3, 1)
    
    # squares += base + translate(g, 1, 1, 1)
    # squares = remove_repeat_squares(squares)
    
    # unit, _, _ = generate_cube(1, 1, 1)
    # squares += translate(unit, 1, 1, 2)
    # squares += translate(unit, 3, 1, 2)
    # squares += translate(unit, 1, 3, 2)
    # squares += translate(unit, 3, 3, 2)
    # squares = remove_repeat_squares(squares)
    
    ## Tunnel
    # base, _, _ = generate_cube(5, 5, 1)
    # unit, _, _ = generate_cube(1, 1, 1)
    
    # trench = unit + translate(unit, 1, 0, 0) + translate(unit, 1, 1, 0) + translate(unit, 1, 2, 0) + translate(unit, 0, 2, 0)
    # trench = remove_repeat_squares(trench)
    
    # t1 = trench + translate(unit, 0, 1, 0)
    # t1 = remove_repeat_squares(t1)
    # # g, _, _ = generate_cube(3, 3, 1)
    
    # squares += base + translate(base, 0, 0, 10)
    # # for i in range(1, 3):
    # #     squares += translate(t1, 1, 1, i)
    # # for i in range(3, 6):
    # #     squares += translate(trench, 1, 1, i)
    # # for i in range(6, 8):
    # #     squares += translate(t1, 1, 1, i)
    # for i in range(1, 10):
    #     squares += translate(trench, 1, 1, i)
    # squares = remove_repeat_squares(squares)
    
    # unit, _, _ = generate_cube(1, 1, 1)
    # squares += translate(unit, 1, 1, 2)
    # squares += translate(unit, 3, 1, 2)
    # squares += translate(unit, 1, 3, 2)
    # squares += translate(unit, 3, 3, 2)
    # squares = remove_repeat_squares(squares)
    
    square_to_voxel(squares)


    file = open("glue.txt", "w+")
    for i in range(1):
        print("Iteration: ", i)
        dt = translate(squares, 0, 0, 0)
        
        vert_dict = {}
        for sq in dt:
            append_item_alt(vert_dict, sq)
        
        type_dict = {}
        variables = set()
        for k in vert_dict.keys():
            order_list = vert_link(k, vert_dict[k])
            vertex_type = order_to_string(order_list)
            
            if vertex_type in type_dict:
                type_dict[vertex_type] += 1
            else:
                type_dict[vertex_type] = 1
        
        polynomial, variables = dict_to_polynomial(type_dict, 2)
        file.write(polynomial)
    
    file.close()