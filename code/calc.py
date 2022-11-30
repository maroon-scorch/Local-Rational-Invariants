import sys, itertools
from copy import copy, deepcopy
from main import intersection_point, is_crossing_stable, run, turning_number
from data.point import *
from util import *
# from projection import *
import matplotlib.pyplot as plt
import math

from util import int_between

min_range = 160/180*math.pi
max_range = 200/180*math.pi
    
def custom_cuve(curve_x, curve_y, start, stop, num_points, scale):
    # Given a parameterized function for a curve, produces its polygonal approximation
    samples = np.linspace(start, stop, num=num_points).tolist()
    points = []
    for t in samples:
        current_x = round(scale*curve_x(t))
        current_y = round(scale*curve_y(t))
        points.append(Point(current_x, current_y))
        
    return points

def parse_solution(vertices):
    adj_list = []
    for idx, vert in enumerate(vertices):
        if idx != len(vertices) - 1:
            item = (vert, [vertices[idx - 1], vertices[idx + 1]])
            adj_list.append(item)
        else:
            item = (vert, [vertices[idx - 1], vertices[0]])
            adj_list.append(item)
    return adj_list

def scuffed_index(vec):
    lst = vec.tolist()
    if lst == [0, 1]:
        return 1
    elif lst == [1, 0]:
        return 2
    elif lst == [0, -1]:
        return 3
    elif lst == [-1, 0]:
        return 4
    else:
        print(lst)
        print("QQQQQQQQQQQQQQQQQQQQQQ")
        return 0

def label_this_vertex_ignore(start, end, vertex):
    vec_1 = start.vec - vertex.vec
    vec_2 = end.vec - vertex.vec
    ind_1 = scuffed_index(vec_1)
    ind_2 = scuffed_index(vec_2)
    
    if ind_1 <= ind_2:
        index = str(ind_1) + str(ind_2)
    else:
        index = str(ind_2) + str(ind_1)
    
    return index

def label_this_vertex_order(start, end, vertex):
    vec_1 = start.vec - vertex.vec
    vec_2 = end.vec - vertex.vec
    ind_1 = scuffed_index(vec_1)
    ind_2 = scuffed_index(vec_2)
    index = str(ind_1) + str(ind_2)
    
    return index

def label_vertex(adj_list):
    vertex_dict = {}
    for vert, neighbor in adj_list:
        n0 = neighbor[0]
        n1 = neighbor[1]
        label_v = label_this_vertex_ignore(n0, n1, vert)
        # label_v = label_this_vertex_order(n0, n1, vert)
        
        if vertex_dict.get(label_v) == None:
            vertex_dict[label_v] = [vert]
        else:
            vertex_dict.get(label_v).append(vert)
    return vertex_dict

def process_solution(solution):
    adj_list = parse_solution(solution)

    vertex_dict = label_vertex(adj_list)
    count_list = []
    for key in vertex_dict.keys():
        count_list.append([key, len(vertex_dict[key])])
    # print(vertex_dict.keys())
    # print(count_list)
    # print(len(count_list))    
    return count_list

def pretty_print(count_list, points):
    string = ""
    for idx, count in count_list:
        item = str(count) + "*x_" + idx + " + "
        string += item
    num = 0
    # num = turning_number(points)
    string += "0 == "
    string += (str(num) + ",\n")
    return string

def rotate(f_x, f_y, theta):
    t_x = lambda t: f_x(t)*math.cos(theta) - f_y(t)*math.sin(theta)
    t_y = lambda t: f_x(t)*math.sin(theta) + f_y(t)*math.cos(theta)
    return t_x, t_y

def rotate_curve(points, index):
    points_list = []
    
    if index == 1:
        for pt in points: 
            t_x =  -pt.y
            t_y = pt.x
            points_list.append(Point(t_x, t_y))
    elif index == 2:
        for pt in points: 
            t_x =  -pt.x
            t_y = -pt.y
            points_list.append(Point(t_x, t_y))
    elif index == 3:
        for pt in points: 
            t_x =  pt.y
            t_y = -pt.x
            points_list.append(Point(t_x, t_y))
    else:
        points_list = points        
    return points_list
    
# points = generate_curve(-10, 10, 10, 2)
# # points = getpts(-10, 10, 3)
# visualize(points, "Title", True)

# Return a list of specific grid curves to test for
def specific_curve():
    c0 = [Point(0, 0), Point(1, 0), Point(2, 0), Point(2, 1), Point(1, 1), Point(1, 2), Point(0, 2), Point(0, 1)]
    curves = [c0]
    output = []
    for c in curves:
        c1, c2, c3 = rotate_curve(c, 1), rotate_curve(c, 2), rotate_curve(c, 3)
        output += [c, c1, c2, c3]
    return output

if __name__ == "__main__": 
    polynomial_list = []
    parameter_list = []
    grid_curve_list = []
    
    for i in range(10):
        a = random.randint(-5, -1)
        b = random.randint(1, 5)
        f_x = lambda t: math.cos(a*t) + 1
        f_y = lambda t: math.sin(b*t)
        f1_x, f1_y = rotate(f_x, f_y, math.pi/2)
        f2_x, f2_y = rotate(f_x, f_y, math.pi)
        f3_x, f3_y = rotate(f_x, f_y, 3*math.pi/2)
        
        functions_list = [(f_x, f_y, (a, b)), (f1_x, f1_y, (a, b)), (f2_x, f2_y, (a, b)), (f3_x, f3_y, (a, b))]
        
        for x, y, param in functions_list:
            parameter_list.append(param)
            points = custom_cuve(x, y, 0, 2*math.pi, 100, 30)
            solution = run(points, 2, False)
            # grid_curve_list.append(solution)
    
    grid_curve_list += specific_curve()
     
    for curve in grid_curve_list:
        count_list = process_solution(curve)
        polynomial_list.append([count_list, curve])
    print(len(polynomial_list))
    print("-------------------------------------------------------------")
    with open('poly.txt', 'w') as f:
        for c_lst, solution in polynomial_list:
            txt = pretty_print(c_lst, solution)
            f.write(txt)
        for p in parameter_list:
            f.write(str(p) + "\n")
