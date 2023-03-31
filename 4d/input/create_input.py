import sys, itertools
from copy import copy, deepcopy
# from projection import *
import matplotlib.pyplot as plt
import math
import numpy as np
    
def custom_cuve(curve_x, curve_y, start, stop, num_points, scale):
    # Given a parameterized function for a curve, produces its polygonal approximation
    samples = np.linspace(start, stop, num=num_points).tolist()
    
    points = []
    for t in samples:
        current_x = scale*curve_x(t)
        current_y = scale*curve_y(t)
        points.append([current_x, current_y])
        
    return points

if __name__ == "__main__":
    # points = generate_curve(-10, 10, 10, 2)
    # points = getpts(-10, 10, 3)
    # visualize(points, "Title", True)
      
    f_x = lambda t: math.cos(t)
    f_y = lambda t: math.sin(t)
    # theta = math.pi/4
    # t_x = lambda t: f_x(t)*math.cos(theta) - f_y(t)*math.sin(theta)
    # t_y = lambda t: f_x(t)*math.sin(theta) + f_y(t)*math.cos(theta)
    points = custom_cuve(f_x, f_y, 0, 4*math.pi, 100, 30)
    
    for i, p in enumerate(points):
        if i != len(points) - 1:
            print(str(p[0]) + " " + str(p[1]) + " " + str(points[i+1][0]) + " " + str(points[i+1][1]))
        else:
            print(str(p[0]) + " " + str(p[1]) + " " + str(points[0][0]) + " " + str(points[0][1]))