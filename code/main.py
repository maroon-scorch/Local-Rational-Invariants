#!/bin/python3
import sys, itertools
from copy import copy, deepcopy
from data.point import *
# from projection import *
from functools import reduce
import math
import sympy as sp
from util import *
from solution import solve_half_length

epsilon = 0.1

# Run program using 'python main.py [directory to file]'
def read_input(inputFile):
    """ Read and parse the input file, returning the list of points and its dimension """
    poly_points = []
    with open(inputFile, "r") as f:
        dimension = int(f.readline())
        for line in f.readlines():
            tokens = line.strip().split()
            x_bool = bool(int(tokens[1]))
            y_bool = bool(int(tokens[3]))
            x_pos =  int(tokens[0]) if x_bool else float(tokens[0])
            y_pos =  int(tokens[2]) if y_bool else float(tokens[2])
            new_point = Point(x_pos, y_pos)
            poly_points.append(new_point)
    
    return poly_points, dimension

def validate_input(points):
    """ Checks if the list of points matches the input specification
    """
    # Avoid Non-degenerate lines
    if len(points) < 3:
        print("Error: Need At Least 3 points to indicate a polygonal curve")
        sys.exit(0)
        
    # Every point needs to have one coordinate to be an integer
    for pt in points:
        if not isinstance(pt.x, int) and not isinstance(pt.y, int):
            print("Error: " + str(pt) + " does not have integer coordinate")
            sys.exit(0)
            
    # Angle of any three consecutive points are 160-200 degree
    faulty_ver_list = bad_vertices(points)
    # for vert in faulty_ver_list:
    #     print("Error: " + str(vert) + " has angle out of range.")
            # sys.exit(0)
            
def scale_input(points, n):
    """ Scales the points by a factor of n (integer) """
    result = list(map(lambda pt: Point(n * pt.x, n * pt.y), points))
    return result

# --------------------------------------------------------
#               Refine Inputs
# --------------------------------------------------------
# This needs some work to generalize
def index_segment(point_1, point_2):
    """ Given two end points, returns a list of points on the line connecting the two points
    that meets the integer lattice """
    # If they are the same point, this is degenerate line
    if point_1 == point_2:
        return [point_1]
    
    x_start = point_1.x
    y_start = point_1.y
    x_end = point_2.x
    y_end = point_2.y
    
    x_int = int_between(x_start, x_end) # List of integers the x-value passes through
    y_int = int_between(y_start, y_end) # List of integers the y-value passes through
    
    x_int_cord = []
    y_int_cord = []
    
    if x_start != x_end:
        slope = (y_end - y_start)/(x_end - x_start)
        x_int_cord = list(map(lambda x: Point(x, y_start + slope*(x - x_start)), x_int)) 
    if y_start != y_end:
        inverse_slope = (x_end - x_start)/(y_end - y_start)
        y_int_cord = list(map(lambda y: Point(inverse_slope*(y - y_start) + x_start, y), y_int))
        
    # Only want unique results
    result = list(set(x_int_cord + y_int_cord))
    # Sort the result in order of distance to the starting point
    result = sorted(result, key=lambda pt: dist(pt, point_1))
    
    return result

# Note that this does not affect a straight horizontal line?
def avoid_grid(points):
    result = []
    for pt in points:
        is_x_int = abs(int(pt.x) - pt.x) < 0.00001
        is_y_int = abs(int(pt.y) - pt.y) < 0.00001
        if is_x_int and is_y_int:
            result.append(Point(pt.x + epsilon, pt.y))
        else:
            result.append(pt)
    return result

def refine_input(points):
    # Step 1: Start from one of the end point, index each line segment cut by the grid
    adj_list = []
    for idx, point in enumerate(points):
        if idx != len(points) - 1:
            adj_list.append([point, points[idx + 1]])
    # FUTURE Mattie - might wanna double check this, did you add?
    adj_list.insert(0, [])
    refined_points = reduce(lambda prev, next: prev + index_segment(next[0], next[1]), adj_list)
    refined_points = [k for k, g in itertools.groupby(refined_points)]
    
    # Avoid Grid Points by Perturbing it
    # refined_points = avoid_grid(refined_points)
    return refined_points

# --------------------------------------------------------
#               Smooth Inputs
# --------------------------------------------------------
def smooth(points):
    label_list = list(map(lambda p: label_index(p), points))
    
    triple_list = []
    for idx, pt in enumerate(label_list):
        if idx < len(points) - 2:
            triple_list.append((idx + 1, [label_list[idx], label_list[idx + 1], label_list[idx + 2]]))
            
    triple_list.append((len(points) - 1, [label_list[-2], label_list[-1], label_list[0]]))
    triple_list.insert(0, (0, [label_list[-1], label_list[0], label_list[1]]))
    
    for idx, label in triple_list:
        if label == [0, 1, 0]:
            if points[idx - 1].x == points[idx + 1].x:
                print("Too sharp at index ", idx)
                new_point = Point(points[idx-1].x, (points[idx-1].y + points[idx + 1].y)/2)
                points[idx] = new_point
                return False  
        if label == [1, 0, 1]:
            if points[idx - 1].y == points[idx + 1].y:
                print("Too sharp at index ", idx)
                new_point = Point((points[idx-1].x + points[idx + 1].x)/2, points[idx-1].y)
                points[idx] = new_point
                return False
        # ang = angle(points[idx - 1], points[idx], points[idx + 1]);
        # if not (min_range <= ang and ang <= max_range):
        #     print("Too sharp at index ", idx)
        #     new_point = Point((points[idx-1].x + points[idx + 1].x)/2, (points[idx-1].y + points[idx + 1].y)/2)
        #     points[idx] = new_point
        #     points = scale_input(points, 2)
        #     return False
            
            
    print("No sharp points!")
    return True
    
# --------------------------------------------------------
#               Checking Intersections
# --------------------------------------------------------
def intersection_point(edge_1, edge_2):
    """Given two edges, find its intersection points (hopefully our code won't return some parallel lines, I am not checking this tentatively) """
    
    p1, p2, p3, p4 = to_sp(edge_1[0]), to_sp(edge_1[1]), to_sp(edge_2[0]), to_sp(edge_2[1])
    s1 = sp.Segment(p1, p2)
    s2 = sp.Segment(p3, p4)
    showIntersection = s1.intersection(s2)
    
    # print(showIntersection)
    return showIntersection
      
def find_intersection(points):
    """ Given a list of points, find its intersections """
    edges = vert_to_edges(points)
    points = list(map(lambda x: [x, False], points))
    intersections = []
    
    for idx1, ed in enumerate(edges):
        for i in range(0, idx1 - 1):
            current_edge = edges[i]
            result = intersection_point(ed, current_edge)
            if result != []:
                if idx1 != len(edges) - 1 and i != 0:
                    for pt in result:
                        cross_pnt = Point(float(pt.x), float(pt.y))
                        if points[idx1] == cross_pnt:
                            points[idx1] = [cross_pnt, True]
                        elif points[idx1 + 1] == cross_pnt:
                            points[idx1 + 1] = [cross_pnt, True]
                        else:
                            points.insert(idx1 + 1, [cross_pnt, True])
                        intersections.append(cross_pnt)
    # intersections = list(set(intersections))
    points = [k for k, g in itertools.groupby(points)]
    
    index_list = []
    for idx, pt in enumerate(points):
        if pt[1]:
            index_list.append(idx)
            
    points = list(map(lambda x: x[0], points))
    
    print(points)
    print(intersections)
    visualize(points, "With Intersection Input", True)
    print("----------------------------")
    
    return intersections, index_list
            
def is_stable(points, index):
    """ Given a list of points and an index of self intersection, discern if intersection is stable """
    intersection = points[index]
    # indices = [i for i, x in enumerate(points) if x == intersection]
    # print(points)
    # print(indices)
    duplicate_1 = deepcopy(points)
    duplicate_2 = deepcopy(points)
    duplicate_3 = deepcopy(points)
    duplicate_4 = deepcopy(points)
    
    # Perturbed points
    point_1 = Point(intersection.x + epsilon, intersection.y)
    point_2 = Point(intersection.x - epsilon, intersection.y)
    point_3 = Point(intersection.x, intersection.y  + epsilon)
    point_4 = Point(intersection.x, intersection.y  - epsilon)
        
    for idx in [index]:
        duplicate_1[idx] = point_1
        duplicate_2[idx] = point_2
        duplicate_3[idx] = point_3
        duplicate_4[idx] = point_4    
        
    intersection_1, _ = find_intersection(duplicate_1)
    intersection_2, _ = find_intersection(duplicate_2)
    intersection_3, _ = find_intersection(duplicate_3)
    intersection_4, _ = find_intersection(duplicate_4)
    
    # print(intersection_1[0].vec)
    # print(point_1.vec)
    return approx_contains(intersection_1, point_1) and approx_contains(intersection_2, point_2) and approx_contains(intersection_3, point_3) and approx_contains(intersection_4, point_4)

def is_left(start, end, point):
    result = (end.x - start.x)*(point.y - start.y) - (end.y - start.y)*(point.x - start.x)
    return result > 0

def is_right(start, end, point):
    result = (end.x - start.x)*(point.y - start.y) - (end.y - start.y)*(point.x - start.x)
    return result < 0

def is_crossing_stable(seg_1, seg_2):
    intersection = seg_1[1]
    
    e1_start = seg_1[0]
    e1_end = seg_1[2]
    e2_start = seg_2[0]
    e2_end = seg_2[2]
    
    if is_left(e1_start, intersection, e2_start) and is_left(intersection, e1_end, e2_start) and is_left(e1_start, intersection, e2_end) and is_left(intersection, e1_end, e2_end):
        return False
    if not (is_left(e1_start, intersection, e2_start) or is_left(intersection, e1_end, e2_start) or is_left(e1_start, intersection, e2_end) or is_left(intersection, e1_end, e2_end)):
        return False
    
    if is_left(e2_start, intersection, e1_start) and is_left(intersection, e2_end, e1_start) and is_left(e2_start, intersection, e1_end) and is_left(intersection, e2_end, e1_end):
        return False
    if not (is_left(e2_start, intersection, e1_start) or is_left(intersection, e2_end, e1_start) or is_left(e2_start, intersection, e1_end) or is_left(intersection, e2_end, e1_end)):
        return False
    
    return True
    
# garbage code
def is_stable_alt(points, index):
    intersection = points[index]
    
    indices = [i for i, x in enumerate(points) if x == intersection]
    triple_list = []
    for i in indices:
        triple_list.append([points[i-1], intersection, points[i+1]])
    
    for seg_1, seg_2 in list(itertools.combinations(triple_list,2)):
        # if at least one crossing is stable, the entire point is stable
        print(seg_1)
        print(seg_2)
        if is_crossing_stable(seg_1, seg_2):
            return True
    
    return False
    
def side(points, pt):
    # Given a list of points and another single point, 
    # turn True if the point is inside of the closed polygonal curve formed by the point list;
    # otherwise return False.
    
    # Assumption: the first and last point in the list are the same(closed).
    
    odd = False
    
    i = 0
    j = len(points) - 2
    while i < len(points) - 2:
        i = i + 1
        if(((points[i].y > pt.y) != (points[j].y > pt.y)) and 
           (pt.x < ((points[j].x - points[i].x) * (pt.y - points[i].y) / (points[j].y - points[i].y)) + points[i].x)):
            
            odd = not odd
        j = i
    return odd

def count_v(intersect, pt):
    """Count the degree of intersection"""
    count = 0
    for v in intersect:
        if (pt == v):
            count = count + 1
    return count

def type_v(points):
    """Given a list of points, output a list of pairs with each vertex labelled 
    by its degree of intersection(vertex type)"""
    intersect = find_intersection(points)
    for pt in points:
        label_points = []
        if pt in intersect: #intersecting vertices
            count_pt = count_v(intersect, pt)
            label_points.append([count_pt, pt])
            return label_points
        else:
            label_points.append([0, pt])
            return label_points
        
def value_v(points):
    """Given a list of points, output a list of pairs with each vertex labelled 
    by its assigned value"""
    label_points = type_v(points)
    value_points = []
    if points[0] == points[-1]: #closed
            for pair in label_points:
                if pair[0] == 0:
                    value_points.append(pair)
                else:
                    value_points.append([-pair[0], pair[1]])
                   
def euler_curve(points):
    """Given a polygonal curve defined by a list of points, find its Euler Characteristic"""
    intersect = find_intersection(points)
    if intersect == []: #embedded curve
        if points[0] != points[-1]: #open
            return 1
        else: 
            return 0
    else: #Imbedded
        chi = 1
        for pt in points:
            if pt in intersect: #intersecting vertices
                count_pt = count_v(intersect, pt)
                chi = chi - count_pt
        return chi

#---------------------------------------------------------
#               Turning Number for Curves
#---------------------------------------------------------
 
def turning_number(points):
    ang_list = []
    for idx, pt in enumerate(points):
        if idx < len(points) - 2:
            ang_list.append([pt, points[idx + 1], points[idx + 2]])
    
    ang_list.append([points[-2], points[-1], points[0]])
    ang_list.append([points[-1], points[0], points[1]])
    
    number = 0
    for triple in ang_list:
        ang = angle(triple[0], triple[1], triple[2])
        if ang != math.pi: # isn't a striaght line
            vec_1 = triple[1].vec - triple[0].vec
            vec_2 = triple[2].vec - triple[1].vec
            if vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0] > 0:
                number += 0.25
            else:
                number -= 0.25
    return number
 
#---------------------------------------------------------
#               Euler Characteristic for Surfaces
#---------------------------------------------------------

    

# --------------------------------------------------------
#               Main Operation
# --------------------------------------------------------
def run(points, dimension, close):
    validate_input(points)
        
    # Apply appropriate scaling?
    # points = scale_input(points, 2)
    
    # If the curve is not closed, close it
    if close and points[0] != points[-1]:
        points.append(points[0])
        
    # print("This is a curve in dimension: ", dimension)
    # print("The list has length: ", len(points))
    # print("The list has points: ", points)
    # visualize(points, "Raw Input", True)
    # Refine the inputs first
    refined_points = refine_input(points)
    refined_points = avoid_grid(refined_points)
    refined_points = refine_input(refined_points)
    # visualize(refined_points, "Refined Input", True)
    
    # finds the intersections first
    # duplicate = deepcopy(refined_points)
        
    # _, index_list = find_intersection(refined_points)
    # for idx in index_list:
    #     if not is_stable(refined_points, idx):
    #     if not is_stable_alt(refined_points, idx):
    #         print("Intersection is not stable!")
    #         sys.exit(0)
    #     else:
    #         print("Intersection is stable!")
        
    # visualize(refined_points, "Refined Input")
    # solution = solve_alt(refined_points)
    # solution = solve_project(refined_points)
    # visualize(solution, "Grid Approximation", False)
    solution = solve_half_length(refined_points)
    solution = remove_adjacent(scale_input(solution, 2))
    # visualize(solution, "Title", False)
    return solution
    
    # solution = solve_halfLength(refined_points)


# The main body of the code:
if __name__ == "__main__":
    input_file = sys.argv[1]
    points, dimension = read_input(input_file)
    run(points, dimension, True)