#!/bin/python3
from re import X
import sys, itertools
from copy import copy, deepcopy
from weakref import ref
from point import Point, dist
import matplotlib.pyplot as plt
from functools import reduce

from util import int_between

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
    """ Checks if the list of points matches the input specification """
    # Avoid Non-degenerate lines
    if len(points) < 2:
        print("Error: Need At Least 2 points to indicate a polygonal curve")
        sys.exit(0)
        
    # Every point needs to have one coordinate to be an integer
    for pt in points:
        if not isinstance(pt.x, int) and not isinstance(pt.y, int):
            print("Error: " + str(pt) + " does not have integer coordinate")
            sys.exit(0)
            
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
    
def solve(points):
    """ Given a polygonal curve, constructs its grid approximation """
    # Step 1: Start from one of the end point, index each line segment cut by the grid
    adj_list = []
    for idx, point in enumerate(points):
        if idx != len(points) - 1:
            adj_list.append([point, points[idx + 1]])
    adj_list.insert(0, [])
    refined_points = reduce(lambda prev, next: prev + index_segment(next[0], next[1]), adj_list)
    refined_points = [k for k, g in itertools.groupby(refined_points)]
    print(refined_points)
    visualize(refined_points, "Refined Input")
    
    # Step 2: For all the points on the curve that intersect with the grids, mark the point to be 0 if it meets
    # the horizontal sides of the grid and 1 if it meets with the vertical sides.

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
    print(index_segment(Point(0, 0), Point(1.5, 1)))
    points, dimension = read_input(input_file)
    validate_input(points)
    
    # Apply appropriate scaling?
    # points = scale_input(points, 2)
    
    # If the curve is not closed, close it
    if points[0] != points[-1]:
        points.append(points[0])

    print("This is a curve in dimension: ", dimension)
    print("The list has length: ", len(points))
    print("The list has points: ", points)
    
    visualize(points, "Raw Input")
    solve(points)