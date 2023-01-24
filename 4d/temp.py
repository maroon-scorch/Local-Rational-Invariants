import math
import numpy as np
from pointn import *

# vertex = []
# for i in range(30):
#     cur = i/30*(2*math.pi)
    
#     x = 1.5*math.cos(cur)
#     y = 1.5*math.sin(cur)
    
#     vertex.append((x, y))
    
# for i in range(len(vertex)):
#     if i != len(vertex) - 1:
#         current_v = vertex[i]
#         next_v = vertex[i+1]
#     else:
#         current_v = vertex[i]
#         next_v = vertex[0]
    
#     string = str(current_v[0]) + " " + str(current_v[1]) + " " + str(next_v[0]) + " " + str(next_v[1]) + "\n"
#     print(string)

def to_string(point3):
    output = str(point3.points[0]) + " " + str(point3.points[1]) + " " + str(point3.points[2])
    return output

def custom_cuve(curve_x, curve_y, curve_z, start, stop, num_points, scale):
    # Given a parameterized function for a curve, produces its polygonal approximation
    samples = np.linspace(start, stop, num=num_points).tolist()
    # print(samples)
    
    points = []
    for t in samples:
        current_x = scale*curve_x(t)
        current_y = scale*curve_y(t)
        current_z = scale*curve_z(t)
        # current_x = scale*curve_x(t)
        # current_y = scale*curve_y(t)
        # current_z = scale*curve_z(t)
        points.append(PointN([current_x, current_y, current_z]))
    return points

    
# Trefoil
f_x = lambda t: math.sin(t) + 2*math.sin(2*t)
f_y = lambda t: math.cos(t) - 2*math.cos(2*t)
f_z = lambda t: -math.sin(3*t)
points = custom_cuve(f_x, f_y, f_z, 0, 2*math.pi , 30, 5)

for i in range(len(points)):
    if i != len(points) - 1:
        current_v = points[i]
        next_v = points[i+1]
    else:
        current_v = points[i]
        next_v = points[0]
        
    string = to_string(current_v) + " " + to_string(next_v)
    print(string)