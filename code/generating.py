import sys, itertools
from copy import copy, deepcopy
from main import intersection_point, is_crossing_stable, run
from data.point import *
from util import *
# from projection import *
import matplotlib.pyplot as plt
import math

from util import int_between

"""Given a square range and the number of vertices, we generate a list of random validate vertices."""
# The lower and higher bound are both non-negative (?)

min_range = 160/180*math.pi
max_range = 200/180*math.pi

# Given two preceding vertices, check if the third vertex forms the angle out of range 160-200 degree.
def faulty_vertex(p1, p2, p3):
    ang = angle(p1, p2, p3)
    if not (min_range <= ang and ang <= max_range):
        return True # Bad Vertex
    else: return False
    
def adj_triple(points, i):
    triple = []
    if 1<= i and i <= len(points) - 2:
        triple.append(points[i-1])
        triple.append(points[i])
        triple.append(points[i+1])
    else: raise IndexError("Index out of range.")
    return triple



def current_intersection(points, p):
    """ Given a list of points, find out the intersecting edge with the new edge """
    edges = vert_to_edges_open(points)
    current_edge = [points[-1], p]
    points = list(map(lambda x: [x, False], points))
    #intersections = []
    
    for idx, ed in enumerate(edges):
        result = intersection_point(ed, current_edge) # Double Check: intersect at the vertex?
        if result != []: # Intersecting Edge: [points[idx], points[idx + 1]]
            if idx != len(edges) - 1:
                for pt in result:
                    cross_pnt = Point(float(pt.x), float(pt.y))
                    #intersections.append(cross_pnt)
        return [idx, cross_pnt]
    
def is_current_stable(points, p):
    idx = current_intersection(points, p)[0]
    cross_pnt = current_intersection(points, p)[1]
    seg_1 = [points[idx], cross_pnt, points[idx + 1]]
    seg_2 = [points[-1], cross_pnt, p]
    return is_crossing_stable(seg_1, seg_2)
        
def getpts(low, high, num_points):
    result = []
    for i in range(num_points):
        p = Point(random.uniform(low, high), random.uniform(low, high))
        while ((p.x - math.floor(p.x) < 0.001 and p.y - math.floor(p.y) < 0.001) # Avoid grid points
               or (i >= 2 and 
                   (faulty_vertex(result[i-2], result[i-1], p) or # Avoid bad vertices
                    not is_current_stable(result, p)))): # Avoid unstable-intersections
            p = Point(random.uniform(low, high), random.uniform(low, high))  
        result.append(p)
    return result
    
def getcurve(low, high, num_points):
    points = getpts(low, high, num_points)
    curve = vert_to_edges_open(points) 
    return curve
    
def generate(low, high, num_points, dimension):
    # # If the curve is not closed, close it
    # if points[0] != points[-1]:
    #     points.append(points[0])
    points = getpts(low, high, num_points)
    curve = getcurve(low, high, num_points)

    print("This is a curve in dimension: ", dimension)
    # print("The list has length: ", len(curve))
    print("The list has points: ", num_points)
    visualize(points, "Random Generated Points", True)
    
    
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

generate(-10, 10, 20, 2)
      
# f_x = lambda t: math.cos(t)
# f_y = lambda t: math.sin(2*t)
# theta = math.pi/4
# t_x = lambda t: f_x(t)*math.cos(theta) - f_y(t)*math.sin(theta)
# t_y = lambda t: f_x(t)*math.sin(theta) + f_y(t)*math.cos(theta)
# points = custom_cuve(t_x, t_y, 0, 2*math.pi, 100, 30)
# run(points, 2, False)