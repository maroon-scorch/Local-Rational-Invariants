from surface import Square, square_to_voxel
from point3 import *
import math, itertools, random, sys
import numpy as np
from euler import vert_link, order_to_string, dict_to_polynomial
from corner_cube import point_cloud_to_squares, generate_cube

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

if __name__ == "__main__":
    angle = [0, math.pi/2, 3*math.pi/2, math.pi]
    
    # cube, _, _ = generate_cube(1, 1, 19)
    # cube1, _, _ = generate_cube(1, 1, 10)
    hedge, _, _ = generate_cube(1, 1, 10)
    h2, _, _ = generate_cube(10, 1, 1)
    h3, _, _ = generate_cube(1, 10, 1)
    
    squares = []
    
    # Crystal Structure
    for i in range(10):
        squares += translate(hedge, i, 0, 0)
        squares = remove_repeat_squares(squares)
    for i in range(10):
        squares += translate(h2, 1, i, -1)
        squares = remove_repeat_squares(squares)
    for i in range(10):
        squares += translate(h3, 1, 1, i)
        squares = remove_repeat_squares(squares)
    
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
    
    
    
    square_to_voxel(squares)


    file = open("glue.txt", "w+")
    for i in range(1):
        print("Iteration: ", i)
        dt = translate(squares, 0, 0, 0)
        
        vert_dict = {}
        for sq in dt:
            append_item_alt(vert_dict, sq)
        
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