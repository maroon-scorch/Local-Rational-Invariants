from point3 import *
from main import run
import math

def custom_cuve(curve_x, curve_y, curve_z, start, stop, num_points, scale):
    # Given a parameterized function for a curve, produces its polygonal approximation
    samples = np.linspace(start, stop, num=num_points).tolist()
    print(samples)
    
    points = []
    for t in samples:
        current_x = round(scale*curve_x(t))
        current_y = round(scale*curve_y(t))
        current_z = round(scale*curve_z(t))
        # current_x = scale*curve_x(t)
        # current_y = scale*curve_y(t)
        # current_z = scale*curve_z(t)
        points.append(Point3(current_x, current_y, current_z))
        
    return points

# Broken Circle
f_x = lambda t: math.cos(t)
f_y = lambda t: 0.0001
f_z = lambda t: math.sin(t)
# points = custom_cuve(f_x, f_y, f_z, 0, 2*math.pi , 20, 30)
# Might have issue at close point?
points = custom_cuve(f_x, f_y, f_z, 1, 2*math.pi + 1 , 20, 30)

# Un-knot
theta = math.pi/4
f_x = lambda t: math.cos(t)
f_y = lambda t: 0.001
f_z = lambda t: math.sin(t)
t_x = lambda t: f_x(t)*math.cos(theta) - f_z(t)*math.sin(theta)
t_z = lambda t: f_x(t)*math.sin(theta) + f_z(t)*math.cos(theta)
# points = custom_cuve(t_x, f_y, t_z, 0, 2*math.pi , 20, 30)

# Trefoil
f_x = lambda t: math.sin(t) + 2*math.sin(2*t)
f_y = lambda t: math.cos(t) - 2*math.cos(2*t)
f_z = lambda t: -math.sin(3*t)
# points = custom_cuve(f_x, f_y, f_z, 0, 2*math.pi , 20, 30)

# Spiral
f_x = lambda t: math.cos(t)
f_y = lambda t: math.sin(t)
f_z = lambda t: t
# points = custom_cuve(f_x, f_y, f_z, 0, 8*math.pi , 20, 30)

# Twisted Cubic
f_x = lambda t: t
f_y = lambda t: t*t
f_z = lambda t: t*t*t
# points = custom_cuve(f_x, f_y, f_z, -3, 3 , 20, 30)

# theta = math.pi/4
# t_x = lambda t: f_x(t)*math.cos(theta) - f_y(t)*math.sin(theta)
# t_y = lambda t: f_x(t)*math.sin(theta) + f_y(t)*math.cos(theta)

# Avoid Catastrophy
# f_y = lambda t: 0.1
# points = custom_cuve(f_x, f_y, f_z, 0, 2*math.pi, 20, 30)
run(points, 3, False)