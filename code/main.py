#!/bin/python3
import sys, itertools
from copy import copy, deepcopy
from point import *
# from projection import *
import matplotlib.pyplot as plt
from functools import reduce
import math

import sympy as sp

from util import int_between

min_range = 160/180*math.pi
max_range = 200/180*math.pi
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

def bad_vertices(points):
    """Given a list of points, find its bad vertices (angle too low) """
    # Angle of any three consecutive points are 160-200 degree
    ang_list = []
    for idx, pt in enumerate(points):
        if idx < len(points) - 2:
            ang_list.append([pt, points[idx + 1], points[idx + 2]])
    
    ang_list.append([points[-2], points[-1], points[0]])
    ang_list.append([points[-1], points[0], points[1]])
    
    faulty_ver_list = []
    for triple in ang_list:
        ang = angle(triple[0], triple[1], triple[2])
        if not (min_range <= ang and ang <= max_range):
            faulty_ver_list.append([triple[1].x, triple[1].y])
    
    return faulty_ver_list

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
    for vert in faulty_ver_list:
        print("Error: " + str(vert) + " has angle out of range.")
            # sys.exit(0)
            
def scale_input(points, n):
    """ Scales the points by a factor of n (integer) """
    result = list(map(lambda pt: Point(n * pt.x, n * pt.y), points))
    return result

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

def label(point):
    """ Given a point on the integer grid, labels whether it's on the vertical or horizontal part """
    is_on_x = (point.x - int(point.x) == 0)
    is_on_y = (point.y - int(point.y) == 0)
    return is_on_x, is_on_y

def label_index(point):
    is_on_x = (point.x - int(point.x) == 0)
    is_on_y = (point.y - int(point.y) == 0)
    
    index = 0 if is_on_x else 1
    return index
    

def has_grid_point(points):
    """ Checks if the points contain a point on the integer lattice """
    for pt in points:
        is_x_int = abs(int(pt.x) - pt.x) < 0.00001
        is_y_int = abs(int(pt.y) - pt.y) < 0.00001
        if is_x_int and is_y_int:
            return True
    return False

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

def vert_to_edges(points):
    """Given a sequence of vertices, convert them into a list of edges"""
    edges = []
    for idx, point in enumerate(points):
        if idx != len(points) - 1:
            edges.append([point, points[idx + 1]])
    edges.append([points[-1], points[0]])
    return edges

def vert_to_edges_open(points):
    edges = []
    for idx, point in enumerate(points):
        if idx != len(points) - 1:
            edges.append([point, points[idx + 1]])
    return edges

def edges_to_vert(edges):
    """Given a list of edges, convert them into a list of vertices"""
    vert = list(map(lambda ed: ed[0], edges))
    return vert

def local_smooth(points, current_index, labels):
    if labels == [0, 1, 0]:
        a = 1    
    elif labels == [1, 0, 1]:
        b = 1
    
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
    
    # result = scale_input(, 2)
    
def closest_grid_point(point):
    x_1 = math.floor(point.x)
    x_2 = math.ceil(point.x)
    y_1 = math.floor(point.y)
    y_2 = math.ceil(point.y)
    
    p1 = Point(x_1, y_1)
    p2 = Point(x_1, y_2)
    p3 = Point(x_2, y_1)
    p4 = Point(x_2, y_2)
    
    result = sorted([p1, p2, p3, p4], key=lambda pt: dist(pt, point))
    return result[0]  

def solve(points):
    """ Given a polygonal curve, constructs its grid approximation """
    grid_list = []
    label_list = list(map(lambda p: label_index(p), points))
    
    
    for idx, point in enumerate(points):
        # Find the closest grid point to the first point
        if idx == 0:
            initial_point = closest_grid_point(point)
            grid_list.append(initial_point)
        else:
            # Same Label, move straight?
            if label_list[idx - 1] == label_list[idx]:
                previous_point = grid_list[-1]
                if label_list[idx] == 0:
                    diff = point.y - previous_point.y
                    diff = diff/abs(diff) 
                    new_point = Point(previous_point.x, previous_point.y + diff)
                    grid_list.append(new_point)
                else:
                    diff = point.x - previous_point.x
                    diff = diff/abs(diff) 
                    new_point = Point(previous_point.x + diff, previous_point.y)
                    grid_list.append(new_point)
            # Different label, move corner?
            else:
                previous_point = grid_list[-1]
                diff_x = point.x - previous_point.x
                diff_x = diff_x/abs(diff_x)
    
                diff_y = point.y - previous_point.y
                diff_y = diff_y/abs(diff_y)
                
                new_point = Point(previous_point.x + diff_x, previous_point.y)
                new_point_2 = Point(new_point.x, new_point.y + diff_y)
                grid_list.append(new_point)
                grid_list.append(new_point_2)
                    
    
    # # Step 2: For all the points on the curve that intersect with the grids, 
    # # mark the point to be 0 if it meets
    # # the horizontal sides of the grid and 1 if it meets with the vertical sides.
    # grid_list = []
    
    # label_dict = {}
    # for pt in points:
    #     label_dict[pt] = label_index(pt)
    
    # adj_list = []
    # for idx, point in enumerate(points):
    #     if idx != len(points) - 1:
    #         adj_list.append([point, points[idx + 1]])
    # adj_list.append([points[-1], points[0]])
    
    print(grid_list)
    return grid_list

def solve_alt(points):
    """ Given a polygonal curve, constructs its grid approximation """
    grid_list = []
    label_list = list(map(lambda p: label_index(p), points))
    
    
    for idx, point in enumerate(points):
        # Find the closest grid point to the first point
        if idx == 0:
            grid_list.append(point)
        else:
            previous_point = grid_list[-1]
            next_point_1 = Point(previous_point.x, point.y)
            next_point_2 = Point(point.x, next_point_1.y)
            grid_list.append(next_point_1)
            grid_list.append(next_point_2)
    
    print(grid_list)
    return grid_list

def grid_points(point):
    x_1 = math.floor(point.x)
    x_2 = math.ceil(point.x)
    y_1 = math.floor(point.y)
    y_2 = math.ceil(point.y)
    
    p1 = Point(x_1, y_1)
    p2 = Point(x_1, y_2)
    p3 = Point(x_2, y_1)
    p4 = Point(x_2, y_2)
    
    return p1, p2, p3, p4

# Need to change later
def find_grid(pt, points):
    x_list = list(filter(lambda p: pt.x == p.x, points))
    y_list = list(filter(lambda p: pt.y == p.y, points))
    result = x_list + y_list
    return result[0]

def is_grid_point(pt):
    return isinstance(pt.x, int) and isinstance(pt.y, int)

def visualize_edges(grid_edge_list):
    for i, ed in enumerate(grid_edge_list):
        start = ed[0]
        end = ed[1]
        plt.plot([start.x, end.x], [start.y, end.y], 'k-')
        plt.annotate(i, [(start.x + end.x)/2, (start.y + end.y)/2])
    plt.show()

def solve_project(points):
    """ Given a polygonal curve, constructs its grid approximation """
    print("TESTTTT")
    grid_list = []
    grid_edge_list = []
    edge_list = vert_to_edges(points)
    label_list = list(map(lambda p: [label_index(p[0]), label_index(p[1])], edge_list))
    
    for idx, edge in enumerate(edge_list):
        current_label = label_list[idx]
        if current_label[0] == current_label[1]:
            if current_label[0] == 0:
                # Horizontal line:
                edge_1 = [Point(edge[0].x, math.ceil(edge[0].y)), Point(edge[1].x, math.ceil(edge[1].y))]
                edge_2 = [Point(edge[0].x, math.floor(edge[0].y)), Point(edge[1].x, math.floor(edge[1].y))]
                
                if not is_left(edge[0], edge[1], edge_1[0]):
                    grid_edge_list.append(edge_1)
                else:
                    grid_edge_list.append(edge_2)
            elif current_label[0] == 1:
                # Vertical line:
                edge_1 = [Point(math.ceil(edge[0].x), edge[0].y), Point(math.ceil(edge[1].x), edge[1].y)]
                edge_2 = [Point(math.floor(edge[0].x), edge[0].y), Point(math.floor(edge[1].x), edge[1].y)]
                
                if not is_left(edge[0], edge[1], edge_1[0]):
                    grid_edge_list.append(edge_1)
                else:
                    grid_edge_list.append(edge_2)
        else:
            start = edge[0]
            end = edge[1]
            midpoint = Point((start.x + end.x)/2, (start.y + end.y)/2)
            p1, p2, p3, p4 = grid_points(midpoint)
            
            right_points = list(filter(lambda p: not is_left(start, end, p), [p1, p2, p3, p4]))
            if len(right_points) == 1:
                edge[0] = edge[1]
                # only_right = right_points[0]
                # grid_edge_list.append([start, only_right])
                # grid_edge_list.append([only_right, end])
            elif len(right_points) == 3:
                grid_start = find_grid(start, right_points)
                grid_end = find_grid(end, right_points)
                right_points.remove(grid_start)
                right_points.remove(grid_end)
                
                corner = right_points[0]
                grid_edge_list.append([grid_start, corner])
                grid_edge_list.append([corner, grid_end])
                
    # Remove hedges
    for i, ed in enumerate(grid_edge_list):
        print(str(i) + ": " + str(ed))
    faulty_edge_list = []
    
    for idx, ed_1 in enumerate(grid_edge_list):
        if idx != len(grid_edge_list) - 1:
            second_idx = idx + 1
        else:
            second_idx = 0
        ed_2 = grid_edge_list[second_idx]
        inverse_ed_2 = [ed_2[1], ed_2[0]]
        if ed_1 == inverse_ed_2:
            faulty_edge_list.append(idx)
            faulty_edge_list.append(second_idx)

    print(faulty_edge_list)
    faulty_edge_list.sort(reverse=True)
    
    for i in faulty_edge_list:
        grid_edge_list.pop(i)

    visualize_edges(grid_edge_list)         
    grid_list = edges_to_vert(grid_edge_list)
    # if grid_list[0] != grid_list[-1]:
    #     grid_list.append(grid_list[0])
    
         
    # grid_list = list(filter(lambda p: is_grid_point(p), grid_list))
    # print(grid_list)

    # input = map(lambda ed: [[ed[0].x, ed[0].y], [ed[1].x, ed[1].y]], grid_edge_list)
    # # Plot of the Polygonal Curve
    # fig = plt.figure()
    # for i in range(0, len(input)):
    #     plt.plot(input[i][0], input[i][1])
        
    # # Integer Grid
    # ax = fig.gca()
    # xmin, xmax = ax.get_xlim() 
    # ymin, ymax = ax.get_ylim() 
    # ax.set_xticks(int_between(xmin, xmax))
    # ax.set_yticks(int_between(ymin, ymax))
    # plt.grid()
    # # Title
    # plt.title("Edge of Grid")
    # plt.show()
    
    
    return grid_list
    
def solve_direction(points):
    grid_list = []
    return grid_list
    

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
    
    print(point_1)
        
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
        print("dfaf")
        print(seg_1)
        print(seg_2)
        if is_crossing_stable(seg_1, seg_2):
            return True
    
    return False


def visualize(points, title, want_bad_vert):
    """ Given a list of points and a title, draws the curve traced out by it """
    input = map(lambda pt: [pt.x, pt.y], points)
    x_pts, y_pts = zip(*input) #create lists of x and y values
    
    # Plot of the Polygonal Curve
    fig = plt.figure()
    plt.plot(x_pts, y_pts)
    for i in range(0, len(x_pts)):
        is_on_x, is_on_y = label(points[i])
        color = 'r-o' if is_on_x else 'b-o'
        if is_on_x and is_on_y:
            color = 'g-o'
        plt.plot(x_pts[i], y_pts[i], color)
        plt.annotate(i, (x_pts[i], y_pts[i]))

    if want_bad_vert:
        ver_x, ver_y = zip(*bad_vertices(points))
        plt.scatter(ver_x, ver_y, c ="yellow",
                linewidths = 2,
                marker ="^",
                edgecolor ="red",
                s = 200)
    
    # Integer Grid
    ax = fig.gca()
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim() 
    ax.set_xticks(int_between(xmin, xmax))
    ax.set_yticks(int_between(ymin, ymax))
    plt.grid()
    # Title
    plt.title(title)

    plt.show()
    
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

def solve_half_length(points):
    grid_edge_list = []
    edge_list = vert_to_edges_open(points)

    for edge in edge_list:
        current_p1 = edge[0]
        current_p2 = edge[1]

        mid_x_1 = (math.floor(current_p1.x) + math.ceil(current_p1.x))/2
        mid_y_1 = (math.floor(current_p1.y) + math.ceil(current_p1.y))/2
        mid_x_2 = (math.floor(current_p2.x) + math.ceil(current_p2.x))/2
        mid_y_2 = (math.floor(current_p2.y) + math.ceil(current_p2.y))/2

        midpoint = Point((current_p1.x + current_p2.x)/2, (current_p1.y + current_p2.y)/2)
        c1, _, _, c1_op = grid_points(midpoint)
        center = Point((c1.x + c1_op.x)/2, (c1.y + c1_op.y)/2)

        grid_edge_list.append([Point(mid_x_1, mid_y_1), center])
        grid_edge_list.append([center, Point(mid_x_2, mid_y_2)])

    faulty_edge_list = []
    for idx, ed_1 in enumerate(grid_edge_list):
        if idx != len(grid_edge_list) - 1:
            second_idx = idx + 1
        else:
            second_idx = 0
        ed_2 = grid_edge_list[second_idx]
        inverse_ed_2 = [ed_2[1], ed_2[0]]
        if ed_1 == inverse_ed_2 or ed_1 == ed_2:
            faulty_edge_list.append(idx)
            faulty_edge_list.append(second_idx)

    print(faulty_edge_list)
    faulty_edge_list.sort(reverse=True)
    
    for i in faulty_edge_list:
        grid_edge_list.pop(i)

    for i, ed in enumerate(grid_edge_list):
        print(str(i) + ": " + str(ed))

    visualize_edges(grid_edge_list)

def run(points, dimension):
    validate_input(points)
        
    # Apply appropriate scaling?
    # points = scale_input(points, 2)
    
    # If the curve is not closed, close it
    if points[0] != points[-1]:
        points.append(points[0])
        
    print("This is a curve in dimension: ", dimension)
    print("The list has length: ", len(points))
    print("The list has points: ", points)
    visualize(points, "Raw Input", True)
    # Refine the inputs first
    refined_points = refine_input(points)
    refined_points = avoid_grid(refined_points)
    refined_points = refine_input(refined_points)
    visualize(refined_points, "Refined Input", True)
    
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
    # solution = solve_halfLength(refined_points)


# The main body of the code:
if __name__ == "__main__":
    input_file = sys.argv[1]
    points, dimension = read_input(input_file)
    run(points, dimension)
    """
    
    validate_input(points)
    
    # Apply appropriate scaling?
    points = scale_input(points, 5)
    
    # If the curve is not closed, close it
    if points[0] != points[-1]:
        points.append(points[0])

    print("This is a curve in dimension: ", dimension)
    print("The list has length: ", len(points))
    print("The list has points: ", points)
    visualize(points, "Raw Input")
    
    # Refine the inputs first
    refined_points = refine_input(points)
    refined_points = avoid_grid(refined_points)
    # refined_points = refine_input(refined_points)

    
    
    # finds the intersections first
    # duplicate = deepcopy(refined_points)
    
    # _, index_list = find_intersection(refined_points)
    # for idx in index_list:
    #     # if not is_stable(refined_points, idx):
    #     if not is_stable_alt(refined_points, idx):
    #         print("Intersection is not stable!")
    #         sys.exit(0)
    #     else:
    #         print("Intersection is stable!")
    
    visualize(refined_points, "Refined Input")
    # Hopefully this while loop terminates
    # Someone prove this?
    # while not smooth(refined_points) or has_grid_point(refined_points):
    #     refined_points = refine_input(refined_points)
    #     refined_points = avoid_grid(refined_points)
    #     print("-------------")
    # refined_points = refine_input(refined_points)
    # visualize(refined_points, "Smooth and No Grid Points")
    
    solution = solve_alt(refined_points)
    visualize(solution, "Grid Approximation")
    
    # solution = solve(refined_points)
    # visualize(refined_points + solution, "Grid Approximation")
    """