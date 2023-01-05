from surface import Square, square_to_voxel
from point3 import *
import math, itertools, random, sys
import numpy as np
from euler import vert_link, order_to_string, dict_to_polynomial, clean_input
from corner_cube import find_problematic_vertice, point_cloud_to_squares

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

def apply_rigid_motion(square_list, iter):
    polynomial_list = []
    angle = [0, math.pi/2, 3*math.pi/2, math.pi]
    for i in range(iter):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        
        square_list = translate(square_list, a, b, c)
        square_list = rotate_squares_theta(square_list, random.choice(angle))
        square_list = rotate_squares_phi(square_list, random.choice(angle))
        square_list = integerify_square(square_list)
        
        
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
        polynomial_list.append(polynomial)
    return polynomial_list

def integerify_square(squares):
    clean_list = []
    for sq in squares:
        new_sq = Square(round_p3(sq.p1), round_p3(sq.p2),  round_p3(sq.p3), round_p3(sq.p4))
        clean_list.append(new_sq)
    return clean_list  

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

if __name__ == "__main__":
    
    input_file = sys.argv[1]
    angle = [0, math.pi/2, 3*math.pi/2, math.pi]
    
    
    square_list, vertice = load_squares(input_file)
    square_list = integerify_square(square_list)
    square_list = remove_duplicate(square_list)
    
    # square_to_voxel(square_list)
    right_squares = integerify_square(translate(square_list, 5, 0, 0))
    left_squares = integerify_square(translate(square_list, -5, 0, 0))
    double_torus = remove_duplicate(right_squares) + remove_duplicate(left_squares)
    # square_to_voxel(double_torus)
    print(len(double_torus))
    # double_torus = list(set(double_torus))
    
    double_torus = remove_repeat_squares(double_torus)
    
    d2 = integerify_square(translate(double_torus, 10, 10, 0))
    double_torus += d2
    double_torus = remove_repeat_squares(double_torus)
    
    # a = random.randint(-10, 10)
    # b = random.randint(-10, 10)
    # c = random.randint(-10, 10)
        
    # double_torus = translate(double_torus, a, b, c)
    # double_torus = rotate_squares_theta(double_torus, random.choice(angle))
    # double_torus = rotate_squares_phi(double_torus, random.choice(angle))
    # double_torus = integerify_square(double_torus)
    
    # print(len(double_torus))
    
    square_to_voxel(double_torus)
    
    vert_dict = {}
    for sq in double_torus:
        append_item_alt(vert_dict, sq)
    
    type_dict = {}
    for k in vert_dict.keys():
        order_list = vert_link(k, vert_dict[k])
        vertex_type = order_to_string(order_list)
        
        if vertex_type in type_dict:
            type_dict[vertex_type] += 1
        else:
            type_dict[vertex_type] = 1
        
    polynomial, variables = dict_to_polynomial(type_dict, -2)
    
    print(polynomial)
    
    # 77*x_13241 + 12*x_15231 + 9*x_13251 + 24*x_16251 + 18*x_15261 + 14*x_16231 + 61*x_14231 + 9*x_14251 + 12*x_15241 + 7*x_14261 + 19*x_36453 + 6*x_15361 + 4*x_1351 + 12*x_14531 + 3*x_1631 + 8*x_13641 + 7*x_13261 + 3*x_145231 + 6*x_25462 + 3*x_132641 + 4*x_142351 + 5*x_16351 + 4*x_24532 + 4*x_142361 + 14*x_16241 + 9*x_23642 + 3*x_154231 + 3*x_16451 + 3*x_164231 + 4*x_135241 + 3*x_26352 + 2*x_136241 + 3*x_1451 + 8*x_15461 + 4*x_1461 + 2*x_153241 + 11*x_23542 + 11*x_35463 + 2*x_163241 + 6*x_24632 + 2*x_2452 + 5*x_26452 + 4*x_2462 + 2*x_2532 + 2*x_2632 + 3*x_132451 + 2*x_142531 + 8*x_25362 + 4*x_142631 + 7*x_14631 + 2*x_1541 + 1*x_1641 + 2*x_1361 + 3*x_2352 + 3*x_2362 + 3*x_132461 + 3*x_132541 + 3*x_2542 + 1*x_1531 + 3*x_13541 + 3*x_146231 + 1*x_2642 + 0 == -2,
    