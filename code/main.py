#!/bin/python3
from re import X
import sys, itertools
from copy import copy, deepcopy
from weakref import ref
from point import Point, dist, angle
import matplotlib.pyplot as plt
from functools import reduce
import math

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
    ang_list = []
    for idx, pt in enumerate(points):
        if idx < len(points) - 2:
            ang_list.append([pt, points[idx + 1], points[idx + 2]])
    
    ang_list.append([points[-2], points[-1], points[0]])
    ang_list.append([points[-1], points[0], points[1]])
    
    for triple in ang_list:
        ang = angle(triple[0], triple[1], triple[2])
        if not (min_range <= ang and ang <= max_range):
            print("Error: " + str(triple) + " has angle out of range - " + str(ang))
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
    
    return refined_points

def vert_to_edges(points):
    """Given a sequence of vertices, convert them into a list"""
    edges = []
    for idx, point in enumerate(points):
        if idx != len(points) - 1:
            edges.append([point, points[idx + 1]])
    edges.append([points[-1], points[0]]);
    return edges;

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
    # print(label_list)
    # print(triple_list)
    
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
    grid_list.append(Point(0, 1))
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
            next_point_1 = Point(previous_point.x, point.y);
            next_point_2 = Point(point.x, next_point_1.y);
            grid_list.append(next_point_1)
            grid_list.append(next_point_2)
    
    print(grid_list)
    return grid_list
        
    

def visualize(points, title):
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

# The main body of the code:
if __name__ == "__main__":
    input_file = sys.argv[1]
    points, dimension = read_input(input_file)
    validate_input(points)
    
    # Apply appropriate scaling?
    points = scale_input(points, 10)
    
    # If the curve is not closed, close it
    if points[0] != points[-1]:
        points.append(points[0])

    print("This is a curve in dimension: ", dimension)
    print("The list has length: ", len(points))
    print("The list has points: ", points)
    visualize(points, "Raw Input")
    
    # Refine the inputs first
    refined_points = refine_input(points)
    print(refined_points)
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