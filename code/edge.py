import numpy as np
import sympy as sp
from point import *

# Class representing a a directed edge between 2 points
class Edge:
    def __init__(self, start, end):
        self.start = Point(start.x, start.y)
        self.end = Point(end.x, end.y)
        # self.start = np.array([start.x, start.y])
        # self.end = np.array([end.x, end.y])
        # The String Representation of the value of the Literal
        self.value = "(" + str(self.start) + " -> " + str(self.end) + ")"

    def __repr__(self):
        return "(" + str(self.start) + " -> " + str(self.end) + ")"

    def __eq__(self, other):
        if type(other) != Point:
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self):
      return hash((self.start, self.end))
    
    def length(self):
        return dist(self.start, self.end)
    
    
def is_on_edge(start, end, point):
    """ Checks if the point lies on said edge """
    epsilon = 0.000001
    return (dist(point, start) + dist(point, end) - dist(start, end) < epsilon)

# To do: Find a better algorithm for this
def intersection_point(edge_1, edge_2):
    """Given two edges, find its intersection points (hopefully our code won't return some parallel lines, I am not checking this tentatively) """
    
    p1, p2, p3, p4 = to_sp(edge_1.start), to_sp(edge_1.end), to_sp(edge_2.start), to_sp(edge_2.end)
    s1 = sp.Segment(p1, p2)
    s2 = sp.Segment(p3, p4)
    showIntersection = s1.intersection(s2)
    
    print(showIntersection)

# To do: Find a better algorithm for this
def intersect(edge_1, edge_2):
    """Given two edges, determine if they intersect or not"""
    
    p1, p2, p3, p4 = to_sp(edge_1.start), to_sp(edge_1.end), to_sp(edge_2.start), to_sp(edge_2.end)
    s1 = sp.Segment(p1, p2)
    s2 = sp.Segment(p3, p4)
    showIntersection = s1.intersection(s2)
    
    return (showIntersection != [])
    
    # diff_1 = edge_1.end - edge_1.start
    # diff_2 = edge_2.end - edge_2.start
    
    # a_1 = diff_1[1]
    # b_1 = -diff_1[0]
    # c_1 = a_1*edge_1.start.x + b_1*edge_1.start.y
    
    # a_2 = diff_2[1]
    # b_2 = -diff_2[0]
    # c_2 = a_2*edge_2.start.x + b_2*edge_2.start.y
    
    # Mat = np.matrix([[a_1, b_1], [a_2, b_2]])
    
    # if np.linalg.det(Mat) < 0.00000000001:
    #     return False
    # else:
        
    