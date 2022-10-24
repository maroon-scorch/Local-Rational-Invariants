import sys, itertools
from copy import copy, deepcopy
from code.main import vert_to_edges_open, intersection_point, is_crossing_stable, visualize
from point import *
# from projection import *
import matplotlib.pyplot as plt
from functools import reduce
import math

import sympy as sp

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
    
def run(low, high, num_points, dimension):
    # # If the curve is not closed, close it
    # if points[0] != points[-1]:
    #     points.append(points[0])
    points = getpts(low, high, num_points)
    curve = getcurve(low, high, num_points)

    print("This is a curve in dimension: ", dimension)
    print("The list has length: ", len(curve))
    print("The list has points: ", num_points)
    visualize(points, "Random Generated Points", True)