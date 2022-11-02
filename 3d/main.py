#!/bin/python3
import sys, itertools
from copy import copy, deepcopy
from point3 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
import math

epsilon = 0.01
delta = 0.00000000001
min_range = 160/180*math.pi
max_range = 200/180*math.pi
max_iter = 10

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
            z_bool = bool(int(tokens[5]))
            x_pos =  int(tokens[0]) if x_bool else float(tokens[0])
            y_pos =  int(tokens[2]) if y_bool else float(tokens[2])
            z_pos = int(tokens[4]) if z_bool else float(tokens[4])
            new_point = Point3(x_pos, y_pos, z_pos)
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
            faulty_ver_list.append(triple[1].vec.tolist())
    
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
        if not isinstance(pt.x, int) and not isinstance(pt.y, int) and not isinstance(pt.z, int):
            print("Error: " + str(pt) + " does not have integer coordinate")
            # sys.exit(0)
            
    # Angle of any three consecutive points are 160-200 degree
    faulty_ver_list = bad_vertices(points)
    for vert in faulty_ver_list:
        print("Error: " + str(vert) + " has angle out of range.")
            # sys.exit(0)
            
def scale_input(points, n):
    """ Scales the points by a factor of n (integer) """
    result = list(map(lambda pt: Point3(n * pt.x, n * pt.y, n*pt.z), points))
    return result

# --------------------------------------------------------
#               Refine Inputs
# --------------------------------------------------------

def int_between(x, y):
    if x < y:
        return range(math.ceil(x), math.floor(y) + 1)
    else:
        return range(math.ceil(y), math.floor(x) + 1)

# This needs some work to generalize
def index_segment(point_1, point_2):
    """ Given two end points, returns a list of points on the line connecting the two points
    that meets the integer lattice """
    # If they are the same point, this is degenerate line
    if point_1 == point_2:
        pt = point_1
        is_x_int = abs(round(pt.x) - pt.x) < delta
        is_y_int = abs(round(pt.y) - pt.y) < delta
        is_z_int = abs(round(pt.z) - pt.z) < delta
        if is_x_int or is_y_int or is_z_int:
            return [pt]
        else:
            return []
    
    x_start = point_1.x
    y_start = point_1.y
    z_start = point_1.z
    x_end = point_2.x
    y_end = point_2.y
    z_end = point_2.z
    
    x_int = int_between(x_start, x_end) # List of integers the x-value passes through
    y_int = int_between(y_start, y_end) # List of integers the y-value passes through
    z_int = int_between(z_start, z_end) # List of integers the z-value passes through
    
    x_int_cord = []
    y_int_cord = []
    z_int_cord = []
    
    diff = point_2.vec - point_1.vec
    
    if x_start != x_end:
        for x_cord in x_int:
            t = (x_cord - x_start)/diff[0]
            target = point_1.vec + t*diff
            target_point = Point3(x_cord, target[1], target[2])
            x_int_cord.append(target_point)
    if y_start != y_end:
        for y_cord in y_int:
            t = (y_cord - y_start)/diff[1]
            target = point_1.vec + t*diff
            target_point = Point3(target[0], y_cord, target[2])
            y_int_cord.append(target_point)
    if z_start != z_end:
        for z_cord in z_int:
            t = (z_cord - z_start)/diff[2]
            target = point_1.vec + t*diff
            target_point = Point3(target[0], target[1], z_cord)
            z_int_cord.append(target_point)

    # Only want unique results
    result = list(set(x_int_cord + y_int_cord + z_int_cord))
    # Sort the result in order of distance to the starting point
    result = sorted(result, key=lambda pt: dist(pt, point_1))
    
    return result

# Note that this does not affect a straight horizontal line?
def avoid_grid(points):
    result = []
    for pt in points:
        is_x_int = abs(round(pt.x) - pt.x) < delta
        is_y_int = abs(round(pt.y) - pt.y) < delta
        is_z_int = abs(round(pt.z) - pt.z) < delta
        if is_x_int and is_y_int and is_z_int:
            result.append(Point3(pt.x + epsilon, pt.y + epsilon, pt.z))
        elif is_x_int and is_y_int:
            result.append(Point3(pt.x + epsilon, pt.y, pt.z))
        elif is_y_int and is_z_int:
            result.append(Point3(pt.x, pt.y + epsilon, pt.z))
        elif is_z_int and is_x_int:
            result.append(Point3(pt.x, pt.y, pt.z + epsilon))
        else:
            result.append(pt)
    return result

def has_grid_point(points):
    for pt in points:
        is_x_int = abs(round(pt.x) - pt.x) < delta
        is_y_int = abs(round(pt.y) - pt.y) < delta
        is_z_int = abs(round(pt.z) - pt.z) < delta
        if is_x_int and is_y_int and is_z_int:
            return True
        elif is_x_int and is_y_int:
            return True
        elif is_y_int and is_z_int:
            return True
        elif is_z_int and is_x_int:
            return True

    return False

def vert_to_edges_open(points):
    edges = []
    for idx, point in enumerate(points):
        if idx != len(points) - 1:
            edges.append([point, points[idx + 1]])
    return edges

def clean_input(points):
    result = []
    for pt in points:
        is_x_int = abs(round(pt.x) - pt.x) < delta
        is_y_int = abs(round(pt.y) - pt.y) < delta
        is_z_int = abs(round(pt.z) - pt.z) < delta
        
        px, py, pz = pt.x, pt.y, pt.z
        
        if is_x_int:
            px = int(round(pt.x))
        if is_y_int:
            py = int(round(pt.y))
        if is_z_int:
            pz = int(round(pt.z))
        result.append(Point3(px, py, pz))

    return result
        

def refine_input(points):
    # Step 1: Start from one of the end point, index each line segment cut by the grid
    adj_list = vert_to_edges_open(points)
    adj_list.insert(0, [])
    refined_points = reduce(lambda prev, next: prev + index_segment(next[0], next[1]), adj_list)
    refined_points = [k for k, g in itertools.groupby(refined_points)]

    # Avoid Grid Points by Perturbing it
    # refined_points = avoid_grid(refined_points)
    refined_points = clean_input(refined_points)
    return refined_points


def label(point):
    """ Given a point on the integer grid, labels whether it's on the vertical or horizontal part """
    is_on_x = (point.x - round(point.x) == 0)
    is_on_y = (point.y - round(point.y) == 0)
    is_on_z = (point.z - round(point.z) == 0)
    return is_on_x, is_on_y, is_on_z

def visualize(points, title, want_bad_vert):
    """ Given a list of points and a title, draws the curve traced out by it """
    input = map(lambda pt: [pt.x, pt.y, pt.z], points)
    x_pts, y_pts, z_pts= zip(*input) #create lists of x and y values
        
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x_pts, y_pts, z_pts, label='Polygonal Curve')
    
    
    for i in range(0, len(points)):
        is_on_x, is_on_y, is_on_z = label(points[i])
        
        if is_on_x:
            color = '#FF0000'
        elif is_on_y:
            color = '#0000FF'
        else:
            color = '#00FF00'
        
        ax.scatter(x_pts[i], y_pts[i], z_pts[i], c=color)
        ax.text(x_pts[i], y_pts[i], z_pts[i], i)
    
    if want_bad_vert:
        ver_x, ver_y, ver_z = zip(*bad_vertices(points))
        ax.scatter(ver_x, ver_y, ver_z, c ="yellow",
                linewidths = 2,
                marker ="^",
                edgecolor ="red",
                s = 200)
    
    plt.title(title)
    ax.legend()
    plt.show()
    
def visualize_edges(grid_edge_list):
    ax = plt.figure().add_subplot(projection='3d')
    for i, ed in enumerate(grid_edge_list):
        start = ed[0]
        end = ed[1]
        ax.plot([start.x, end.x], [start.y, end.y], [start.z, end.z], 'k-')
        # plt.annotate(i, [(start.x + end.x)/2, (start.y + end.y)/2])
    plt.show()
    
def solve_half_length(points):
    grid_list = []
    grid_edge_list = []
    edge_list = vert_to_edges_open(points)

    for edge in edge_list:
        current_p1 = edge[0]
        current_p2 = edge[1]
        
        # Code breaks on integer lattice lol
        # print(current_p1)
        # print(current_p2)

        mid_x_1 = (math.floor(current_p1.x) + math.ceil(current_p1.x))/2
        mid_y_1 = (math.floor(current_p1.y) + math.ceil(current_p1.y))/2
        mid_z_1 = (math.floor(current_p1.z) + math.ceil(current_p1.z))/2
        
        mid_x_2 = (math.floor(current_p2.x) + math.ceil(current_p2.x))/2
        mid_y_2 = (math.floor(current_p2.y) + math.ceil(current_p2.y))/2
        mid_z_2 = (math.floor(current_p2.z) + math.ceil(current_p2.z))/2

        mp = midpoint_of(current_p1, current_p2)
        
        corner = Point3(math.floor(mp.x), math.floor(mp.y), math.floor(mp.z))
        corner_op = Point3(math.ceil(mp.x), math.ceil(mp.y), math.ceil(mp.z))
        center = midpoint_of(corner, corner_op)

        grid_edge_list.append([Point3(mid_x_1, mid_y_1, mid_z_1), center])
        grid_edge_list.append([center, Point3(mid_x_2, mid_y_2, mid_z_2)])

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

    # print(faulty_edge_list)
    faulty_edge_list.sort(reverse=True)
    
    for i in faulty_edge_list:
        grid_edge_list.pop(i)

    # for i, ed in enumerate(grid_edge_list):
    #     print(str(i) + ": " + str(ed))

    visualize_edges(grid_edge_list)
    grid_list = list(map(lambda ed: ed[0], grid_edge_list))
    grid_list.append(grid_edge_list[-1][1])
    
    return grid_list
    
# --------------------------------------------------------
#               Main Operation
# --------------------------------------------------------
def run(points, dimension, close):
    points = clean_input(points)
    validate_input(points)
    if close and points[0] != points[-1]:
        points.append(points[0])
        
    print("This is a curve in dimension: ", dimension)
    print("The list has length: ", len(points))
    print("The list has points: ", points)
    
    visualize(points, "Raw Input", False)
    # Refine the inputs first
    refined_points = refine_input(points)
    i = 0
    while(has_grid_point(refined_points) and i < max_iter):
        refined_points = avoid_grid(refined_points)
        refined_points = refine_input(refined_points)
        i = i + 1

    print("Has Grid Points: ", has_grid_point(refined_points))
    visualize(refined_points, "Refined Input", False)
    print("Refined Points: ", refined_points)
    solution = solve_half_length(refined_points)
    print(solution)
    # visualize(solution, "Grid Curve", False)

# The main body of the code:
if __name__ == "__main__":
    input_file = sys.argv[1]
    points, dimension = read_input(input_file)
    run(points, dimension, True)