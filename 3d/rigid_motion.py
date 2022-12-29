from surface import Square, square_to_voxel
from point3 import *
import math, itertools, random, sys
import numpy as np
from euler import vert_link, order_to_string, dict_to_polynomial, clean_input


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

def rotate_psi(array, phi):
    original_point = array.tolist()
    new_point = array.tolist()
    
    new_point[0] = original_point[0]*math.cos(phi) - original_point[1]*math.sin(phi)
    new_point[1] = original_point[0]*math.sin(phi) + original_point[1]*math.cos(phi)
    
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

def rotate_squares_psi(square_list, psi):
    new_squares = []
    for sq in square_list:
        s1 = to_point(rotate_psi(sq.p1.vec, psi))
        s2 = to_point(rotate_psi(sq.p2.vec, psi))
        s3 = to_point(rotate_psi(sq.p3.vec, psi))
        s4 = to_point(rotate_psi(sq.p4.vec, psi))
        new_squares.append(Square(s1, s2, s3, s4))
        
    return new_squares

def apply_rigid_motion(square_list, iter, num):
    polynomial_list = []
    angle = [0, math.pi/2, 3*math.pi/2, math.pi]
    for i in range(iter):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        
        square_list = translate(square_list, a, b, c)
        square_list = rotate_squares_theta(square_list, random.choice(angle))
        square_list = rotate_squares_phi(square_list, random.choice(angle))
        square_list = rotate_squares_psi(square_list, random.choice(angle))
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
            
        polynomial, variables = dict_to_polynomial(type_dict, num)
        # print(polynomial)
        polynomial_list.append(polynomial)
    return polynomial_list

if __name__ == "__main__":
    
    input_file = sys.argv[1]
    angle = [0, math.pi/2, 3*math.pi/2, math.pi]
    
    file = open("polynomial.txt", "w+")
    iter = 25
    for i in range(iter):
        print("Iteration: ", i)
        square_list, vertice = load_squares(input_file)
        
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        
        square_list = translate(square_list, a, b, c)
        square_list = rotate_squares_theta(square_list, random.choice(angle))
        square_list = rotate_squares_phi(square_list, random.choice(angle))
        square_list = rotate_squares_psi(square_list, random.choice(angle))
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
        file.write(polynomial)
    
    file.close()