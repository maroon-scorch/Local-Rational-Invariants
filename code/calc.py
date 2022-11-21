import sys, itertools
from copy import copy, deepcopy
from main import intersection_point, is_crossing_stable, run
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
    print(samples)
    
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
        return 0

def label_this_vertex(start, end, vertex):
    vec_1 = start.vec - vertex.vec
    vec_2 = end.vec - vertex.vec
    ind_1 = scuffed_index(vec_1)
    ind_2 = scuffed_index(vec_2)
    
    if ind_1 <= ind_2:
        index = str(ind_1) + str(ind_2)
    else:
        index = str(ind_2) + str(ind_1)
    
    return index


def label_vertex(adj_list):
    vertex_dict = {}
    for vert, neighbor in adj_list:
        n0 = neighbor[0]
        n1 = neighbor[1]
        label_v = label_this_vertex(n0, n1, vert)
        
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

def pretty_print(count_list):
    string = ""
    for idx, count in count_list:
        item = str(count) + "*x_" + idx + " + "
        string += item
    string += " 0 = 2"
    return string
    
    
# points = generate_curve(-10, 10, 10, 2)
# # points = getpts(-10, 10, 3)
# visualize(points, "Title", True)

polynomial_list = []
for i in range(20):
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    f_x = lambda t: math.cos(a*t)
    f_y = lambda t: math.sin(b*t)
    points = custom_cuve(f_x, f_y, 0, 2*math.pi, 100, 30)
    solution = run(points, 2, False)
    count_list = process_solution(solution)
    polynomial_list.append(count_list)
    
print("-------------------------------------------------------------")
for c_lst in polynomial_list:
    print(pretty_print(c_lst))
# print(vertex_dict)

# visualize(solution, "Refined Input", False)
