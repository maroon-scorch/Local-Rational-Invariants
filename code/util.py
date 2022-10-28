import math
from data.point import *
import matplotlib.pyplot as plt
# This files contains some common utilities function

# Parameters Given:
min_range = 160/180*math.pi
max_range = 200/180*math.pi

# Given two numbers x, y, find the integers between them inclusive
def int_between(x, y):
    if x < y:
        return range(math.ceil(x), math.floor(y) + 1)
    else:
        return range(math.ceil(y), math.floor(x) + 1)

# --------------------------------------------------------
#               Vertices and Edges
# --------------------------------------------------------
def vert_to_edges(points):
    """Given a sequence of vertices, convert them into a list of edges"""
    edges = []
    for idx, point in enumerate(points):
        if idx != len(points) - 1:
            edges.append([point, points[idx + 1]])
    edges.append([points[-1], points[0]])
    return edges

def vert_to_edges_open(points):
    edges = []
    for idx, point in enumerate(points):
        if idx != len(points) - 1:
            edges.append([point, points[idx + 1]])
    return edges

def edges_to_vert(edges):
    """Given a list of edges, convert them into a list of vertices"""
    vert = list(map(lambda ed: ed[0], edges))
    return vert

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
            print(ang)
            faulty_ver_list.append([triple[1].x, triple[1].y])
    
    return faulty_ver_list

# --------------------------------------------------------
#               Grid Points
# --------------------------------------------------------

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

def grid_points(point):
    x_1 = math.floor(point.x)
    x_2 = math.ceil(point.x)
    y_1 = math.floor(point.y)
    y_2 = math.ceil(point.y)
    
    p1 = Point(x_1, y_1)
    p2 = Point(x_1, y_2)
    p3 = Point(x_2, y_1)
    p4 = Point(x_2, y_2)
    
    return p1, p2, p3, p4

# Need to change later
def find_grid(pt, points):
    x_list = list(filter(lambda p: pt.x == p.x, points))
    y_list = list(filter(lambda p: pt.y == p.y, points))
    result = x_list + y_list
    return result[0]

def is_grid_point(pt):
    return isinstance(pt.x, int) and isinstance(pt.y, int)

def has_grid_point(points):
    """ Checks if the points contain a point on the integer lattice """
    for pt in points:
        is_x_int = abs(int(pt.x) - pt.x) < 0.00001
        is_y_int = abs(int(pt.y) - pt.y) < 0.00001
        if is_x_int and is_y_int:
            return True
    return False
# --------------------------------------------------------
#               Visualization Methods
# --------------------------------------------------------
def visualize(points, title, want_bad_vert):
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
        plt.annotate(i, (x_pts[i], y_pts[i]))

    if want_bad_vert:
        ver_x, ver_y = zip(*bad_vertices(points))
        plt.scatter(ver_x, ver_y, c ="yellow",
                linewidths = 2,
                marker ="^",
                edgecolor ="red",
                s = 200)
    
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
    
def visualize_edges(grid_edge_list):
    for i, ed in enumerate(grid_edge_list):
        start = ed[0]
        end = ed[1]
        plt.plot([start.x, end.x], [start.y, end.y], 'k-')
        # plt.annotate(i, [(start.x + end.x)/2, (start.y + end.y)/2])
    plt.show()