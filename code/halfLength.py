import sys, itertools
from copy import copy, deepcopy
from main import *
from point import *
# from projection import *
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
import math

import sympy as sp

from util import int_between

def edge_type(point_1, point_2):
    vertical = (label_index(point_1) == 0 and label_index(point_2) == 0)
    horizontal = (label_index(point_1) == 1 and label_index(point_2) == 1)
    crooked1 = (label_index(point_1) == 1 and label_index(point_2) == 0) 
    crooked2 = (label_index(point_1) == 0 and label_index(point_2) == 1)
    return vertical, horizontal, crooked1, crooked2

def solve_halfLength(points):
    grid_edge_list = []
    edge_list = vert_to_edges(points)
    for edge in edge_list:
        current_p1 = edge[0]
        current_p2 = edge[1]
        if edge_type(current_p1, current_p2) == edge_type.vertical:
            x = (math.floor(current_p1.x) + math.ceil(current_p1.x))/2
            y = (current_p1.y + current_p2.y)/2
            c = Point(x,y)
            grid_edge_list.append([Point(x, current_p1.y), c])
            grid_edge_list.append([c, Point(x, current_p2.y)])
        elif edge_type(current_p1, current_p2) == edge_type.horizontal:
            x = (current_p1.x + current_p2.x)/2
            y = math.floor(current_p1.y) + math.ceil(current_p1.y)/2
            c = Point(x,y)
            grid_edge_list.append([Point(current_p1.x, y), c])
            grid_edge_list.append([c, Point(current_p2.x, y)])
        elif edge_type(current_p1, current_p2) == edge_type.crooked1:
            x = (math.floor(current_p2.x) + math.ceil(current_p2.x))/2
            y = (math.floor(current_p1.y) + math.ceil(current_p1.y))/2
            c = Point(x,y)
            grid_edge_list.append([Point(current_p1.x, y), c])
            grid_edge_list.append([c, Point(x, current_p2.y)])
        else:
            x = (math.floor(current_p1.x) + math.ceil(current_p1.x))/2
            y = (math.floor(current_p2.y) + math.ceil(current_p2.y))/2
            c = Point(x,y)
            grid_edge_list.append([Point(x, current_p1.y), c])
            grid_edge_list.append([c, Point(current_p2.x, y)])
            
    visualize_edges(grid_edge_list)
            
    
    