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
        # label_v = label_this_vertex_ignore(n0, n1, vert)
        label_v = label_this_vertex_ignore(n0, n1, vert)
        
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
    # num = 0
    num = turning_number(points)
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

# Generate a legal curve
def legal_curve():
    start = Point(0, 0)
    directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    grid_curve = [start]
    
    max_run = 10
    prev = start
    next = Point(0, 1)
    edge_list = [[start, next]]
    
    failed = False
    
    while next != start and max_run > 0:
        grid_curve.append(next)
        dir = (prev.vec - next.vec).tolist()
        prev = next
        legal_directions = directions.copy()
        legal_directions.remove(dir)
        
        is_illegal = True       
        while is_illegal:
            next_dir = random.choice(legal_directions)
            next_vec = prev.vec + np.array(next_dir)
            next = Point(next_vec[0], next_vec[1])
            
            is_illegal = [prev, next] in edge_list or [next, prev] in edge_list
            if is_illegal:
                legal_directions.remove(next_dir)
                # If there are no places to go
                if legal_directions == []:
                    break
        
        edge_list.append([prev, next])
        max_run = max_run - 1
    
    # Closing the grid curve:
    if grid_curve[0] != grid_curve[-1]:
        x_start = grid_curve[-1].x
        x_end = grid_curve[0].x
        x_sign = 1 if x_end > x_start else -1
        
        y_start = grid_curve[-1].y
        y_end = grid_curve[0].y
        y_sign = 1 if y_end > y_start else -1
        
        while y_start != y_end:
            y_start += y_sign
            new_point = Point(x_start, y_start)
            new_edge = [grid_curve[-1], new_point]
            op_new_edge = [new_point, grid_curve[-1]]
            
            if new_edge in edge_list or op_new_edge in edge_list:
                failed = True
                break
            else:
                grid_curve.append(new_point)
                edge_list.append(new_edge)
            
        while x_start != x_end:
            x_start += x_sign
            new_point = Point(x_start, y_start)
            new_edge = [grid_curve[-1], new_point]
            op_new_edge = [new_point, grid_curve[-1]]
            
            if new_edge in edge_list or op_new_edge in edge_list:
                failed = True
                break
            else:
                grid_curve.append(new_point)
                edge_list.append(new_edge)
                
    if failed:
        return None
    else:
        grid_curve.pop(-1)
        
    # visualize(grid_curve, "", False)
    return grid_curve

def rectangle(w, h):
    curve = [Point(0, 0)]
    
    for i in range(1, w + 1):
        curve.append(Point(i, 0))
    for i in range(1, h + 1):
        curve.append(Point(w, i))
    for i in range(1, w + 1):
        curve.append(Point(w - i, h))
    for i in range(1, h + 1):
        curve.append(Point(0, h - i))
    # visualize(curve, "", False)
    curve.pop(-1)    
    return curve

def counter_example():
    
    curve = [Point(-1, 0), Point(0, 0)]
    for i in range(50):
        curve.append(Point(i, i+1))
        if i != 49:
            curve.append(Point(i+1, i+1))
    for i in range(99):
        curve.append(Point(50-i-2, 50))
    
    for i in range(49):
        curve.append(Point(-50+i, 50-i-1))
        curve.append(Point(-50+i+1, 50-i-1))
    
    # visualize(curve, "", False)
    return curve

# Return a list of specific grid curves to test for
def specific_curve():
    c0 = [Point(0, 0), Point(1, 0), Point(2, 0), Point(2, 1), Point(1, 1), Point(1, 2), Point(0, 2), Point(0, 1)]
    c1 = counter_example()
    curves = [c0, c1]
    
    # for i in range(100):
    #     a = random.randint(1, 50)
    #     b = random.randint(1, 50)
    #     curves.append(rectangle(a, b))
    
    for i in range(1000):
        current = legal_curve()
        if current != None:
            reverse_current = current.copy()
            reverse_current.append(current[0])
            reverse_current.reverse()
            reverse_current.pop(-1)
            
            curves.append(current)
            curves.append(reverse_current)
    
    output = []
    for c in curves:
        c1, c2, c3 = rotate_curve(c, 1), rotate_curve(c, 2), rotate_curve(c, 3)
        output += [c, c1, c2, c3]
    return output

if __name__ == "__main__": 
    polynomial_list = []
    parameter_list = []
    grid_curve_list = []
    
    for i in range(1):
        a = random.randint(-5, -1)
        b = random.randint(1, 5)
        f_x = lambda t: a*math.cos(t)
        f_y = lambda t: b*math.sin(t)
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
    # grid_curve_list += [counter_example()]
    
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
