import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
# import pylab as pl
from point3 import *
import numpy as np
import math
import itertools
from surface import intersection_to_squares, square_to_voxel, Trig, Square

import timeit

def triangle_to_voxel(trig_list):
    ax = a3.Axes3D(plt.figure())
    for sq in trig_list:
        vtx = np.array([sq.p1.vec, sq.p2.vec, sq.p3.vec])
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_alpha(0.5)
        # tri.set_color(colors.rgb2hex(np.random.rand(3)))
        
        ax.scatter(sq.p1.x, sq.p1.y, sq.p1.z, c = '#FF0000')
        ax.scatter(sq.p2.x, sq.p2.y, sq.p2.z, c = '#FF0000')
        ax.scatter(sq.p3.x, sq.p3.y, sq.p3.z, c = '#FF0000')
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    plt.show()
    
def plot_points(point_list):
    ax = a3.Axes3D(plt.figure())
    for pt in point_list:
        ax.scatter(pt.x, pt.y, pt.z, c = '#FF0000')
    plt.show()

def custom_round(x):
    z = round(x)
    left = z - 0.5
    right = z + 0.5
    return left if abs(x - left) < abs(x - right) else right
    
    
# Given two numbers x, y, find the integers between them inclusive
def int_between(x, y):
    print(x, y)
    if x < y:
        return custom_round(x), custom_round(y)
    else:
        return custom_round(y), custom_round(x)

def visualize_surface(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, z)
    print(surf.get_figure())
    print(ax.collections[0])

    plt.show()
    
def lst_to_mid(start, end):
    current = start
    mid = []
    while current < end:
        mid.append(current)
        current += 1
    mid.append(current)
    
    return mid

def tetrahedron_volume(a, b, c, d):
    # Find the signed volume of the Tetrahedron with vertex a, b, c, d
    input_mat = np.array([
        [a.x, a.y, a.z, 1],
        [b.x, b.y, b.z, 1],
        [c.x, c.y, c.z, 1],
        [d.x, d.y, d.z, 1]
    ])
    det = np.linalg.det(input_mat)
    return 1/6*det

def triangle_area(a, b, c):
    # Find the un-signed area of the Triangle with vertex a, b, c
    v1 = b.vec - a.vec
    v2 = c.vec - a.vec
    return 1/2*abs(np.linalg.norm(np.cross(v1, v2)))

def is_point_on_triangle(trg, pt):
    # pt is coplanar with the tirangle
    a, b, c = trg.p1, trg.p2, trg.p3
    
    A = trg.area
    A1, A2, A3 = triangle_area(a, b, pt), triangle_area(a, c, pt), triangle_area(b, c, pt)
    
    return abs(A - (A1 + A2 + A3)) < 0.00001

def is_triangle_degenrate(trg):
    a, b, c = trg.p1, trg.p2, trg.p3
    
    return dist(a, b) < 0.0000000001 or dist(c, b) < 0.0000000001 or dist(a, c) < 0.0000000001

def segment_cross_triangle(trg, start, end):
    # Given a triangle and a line segment defined by start and end point;
    # finds the intersection of the line segment with the triangle (if exist)
    a, b, c = trg.p1, trg.p2, trg.p3
    vol_start = tetrahedron_volume(a, b, c, start)
    vol_end = tetrahedron_volume(a, b, c, end)
    
    if vol_start == 0 and vol_end == 0:
        print((trg, start, end))
        print("Shouldn't happen")
        return False, None
    elif vol_start == 0:
        # This is a grid point
        if is_point_on_triangle(trg, start):
            return True, start
        else:
            return False, None
    elif vol_end == 0:
        # This is a grid point
        if is_point_on_triangle(trg, end):
            return True, end
        else:
            return False, None
    elif np.sign(vol_end) == np.sign(vol_start):
        # This has no intersections
        return False, None
    else:
        # This has an intersection!
        diff = start.vec - end.vec
        normal_vec = trg.normal
        
        # Optimization since we know this is a grid line:
        temp = diff.tolist()
        if temp[1] == 0 and temp[2] == 0:
            x = (normal_vec[1]*(a.y - start.y) + normal_vec[2]*(a.z -start.z))/normal_vec[0] + a.x 
            output = Point3(x, start.y, start.z)
        elif temp[0] == 0 and temp[2] == 0:
            y = (normal_vec[0]*(a.x - start.x) + normal_vec[2]*(a.z - start.z))/normal_vec[1] + a.y 
            output = Point3(start.x, y, start.z)
        elif temp[0] == 0 and temp[1] == 0:
            z = (normal_vec[0]*(a.x - start.x) + normal_vec[1]*(a.y - start.y))/normal_vec[2] + a.z 
            output = Point3(start.x, start.y, z)
        else:
            print("Shouldn't happen - same start and end")
            return False, None
        
        if is_point_on_triangle(trg, output):
            return True, output
        else:
            return False, output
        
        # Solving for time t where line crosses plane
        # sum = normal_vec[0]*a.x + normal_vec[1]*a.y + normal_vec[2]*a.z
        # start_sum = normal_vec[0]*start.x + normal_vec[1]*start.y + normal_vec[2]*start.z
        # diff_sum = normal_vec[0]*diff[0] + normal_vec[1]*diff[1] + normal_vec[2]*diff[2]
        
        # if diff_sum == 0:
        #     # Something went wrong?
        #     return False, None
        # else:
        #     t = (sum - start_sum)/diff_sum
        #     intersection = start.vec + diff*t
        #     output = Point3(intersection[0], intersection[1], intersection[2])
        #     if is_point_on_triangle(trg, output):
        #         return True, output
        #     else:
        #         return False, output


def find_intersection(mesh_triangles, edges):
    point_list = []
    
    for line_start, line_end in edges:
        for trig in mesh_triangles:
            has_points, point = segment_cross_triangle(trig, line_start, line_end)
            if has_points:
                point_list.append(point)
    
    point_list = list(set(point_list))
    return point_list

def find_edges(center):
    line_list = []
    x, y, z = center.x, center.y, center.z
    x_min, x_max = int(math.floor(x)), int(math.ceil(x))
    y_min, y_max = int(math.floor(y)), int(math.ceil(y))
    z_min, z_max = int(math.floor(z)), int(math.ceil(z))
    
    axes = [
        [x_min, x_max],
        [y_min, y_max],
        [z_min, z_max]
    ]
    
    vertices = itertools.product(*axes)
    for start, end in itertools.combinations(vertices, 2):
        p_s = Point3(start[0], start[1], start[2])
        p_e = Point3(end[0], end[1], end[2])
        if abs(dist(p_s, p_e) - 1) < 0.0000001:
            line_list.append([p_s, p_e])
    return line_list

def generate_torus():
    R = 1.9
    r = 0.7
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, 2*np.pi, 25)
    # p: u, v
    f_x = lambda p: (R + r*math.cos(p[0]))*math.cos(p[1])
    f_y = lambda p: (R + r*math.cos(p[0]))*math.sin(p[1])
    f_z = lambda p: r*math.sin(p[0])
    px = []
    py = []
    for a in u:
        for b in v:
            px.append(a)
            py.append(b)
            
    indices = Triangulation(px, py).triangles
    
    mesh = []
    for i1, i2, i3 in indices:
        mesh.append([(px[i1], py[i1]), (px[i2], py[i2]), (px[i3], py[i3])])
    
    mesh_3d = list(map(lambda trig: [Point3(f_x(trig[0]), f_y(trig[0]), f_z(trig[0]))
                                     , Point3(f_x(trig[1]), f_y(trig[1]), f_z(trig[1])),
                                     Point3(f_x(trig[2]), f_y(trig[2]), f_z(trig[2]))], mesh))
    return mesh_3d

def generate_mesh():
    r = 1.9
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    # p: u, v
    f_x = lambda p: r*math.cos(p[0])*math.sin(p[1])
    f_y = lambda p: r*math.sin(p[0])*math.sin(p[1])
    f_z = lambda p: 3.9*math.cos(p[1])
    px = []
    py = []
    for a in u:
        for b in v:
            px.append(a)
            py.append(b)
            
    indices = Triangulation(px, py).triangles
    
    mesh = []
    for i1, i2, i3 in indices:
        mesh.append([(px[i1], py[i1]), (px[i2], py[i2]), (px[i3], py[i3])])
    
    mesh_3d = list(map(lambda trig: [Point3(f_x(trig[0]), f_y(trig[0]), f_z(trig[0]))
                                     , Point3(f_x(trig[1]), f_y(trig[1]), f_z(trig[1])),
                                     Point3(f_x(trig[2]), f_y(trig[2]), f_z(trig[2]))], mesh))
    # file2 = open(r"example.txt", "w+") 
    # for tri in mesh_3d:
    #     p1 = tri[0]
    #     p2 = tri[1]
    #     p3 = tri[2]
    #     string = str(p1.x) + " " + str(p1.y) + " " +  str(p1.z)  + " " +  str(p2.x) + " " + str(p2.y) + " " + str(p2.z) + " " + str(p3.x) + " " + str(p3.y) + " " + str(p3.z)
    #     file2.write(string + "\n")
    # file2.close()
    return mesh_3d

def potato_chip():
    a = 1
    b = 2
    u = np.linspace(-2, 2, 30)
    v = np.linspace(-2, 2, 30)
    
    # p: u, v
    f_x = lambda p: p[0]
    f_y = lambda p: p[1]
    f_z = lambda p: p[1]**2/a - p[0]**2/b + 0.1
    px = []
    py = []
    for a in u:
        for b in v:
            px.append(a)
            py.append(b)
            
    indices = Triangulation(px, py).triangles
    
    mesh = []
    for i1, i2, i3 in indices:
        mesh.append([(px[i1], py[i1]), (px[i2], py[i2]), (px[i3], py[i3])])
    
    mesh_3d = list(map(lambda trig: [Point3(f_x(trig[0]), f_y(trig[0]), f_z(trig[0]))
                                     , Point3(f_x(trig[1]), f_y(trig[1]), f_z(trig[1])),
                                     Point3(f_x(trig[2]), f_y(trig[2]), f_z(trig[2]))], mesh))
    return mesh_3d

def debug_plot(trig_list, point_list, center):
    ax = a3.Axes3D(plt.figure())
    for sq in trig_list:
        vtx = np.array([sq.p1.vec, sq.p2.vec, sq.p3.vec])
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_alpha(0.5)
        # tri.set_color(colors.rgb2hex(np.random.rand(3)))
        
        # ax.scatter(sq.p1.x, sq.p1.y, sq.p1.z, c = '#FF0000')
        # ax.scatter(sq.p2.x, sq.p2.y, sq.p2.z, c = '#FF0000')
        # ax.scatter(sq.p3.x, sq.p3.y, sq.p3.z, c = '#FF0000')
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
        
    for pt in point_list:
        ax.scatter(pt.x, pt.y, pt.z, c = '#00FF00')
    for pt in center:
        ax.scatter(pt.x, pt.y, pt.z, c = '#FF0000')
        grid_edge_list = find_edges(pt)
        for i, ed in enumerate(grid_edge_list):
            start = ed[0]
            end = ed[1]
            ax.plot([start.x, end.x], [start.y, end.y], [start.z, end.z], 'k-')    
    plt.show()

def solve(mesh_triangles):
    x_list = list(map(lambda trigs: [trigs.p1.x, trigs.p2.x, trigs.p3.x], mesh_triangles))
    x_list = np.array(x_list).flatten()
    x_start, x_end = int_between(np.min(x_list), np.max(x_list))
    x_cen = lst_to_mid(x_start, x_end)
    
    y_list = list(map(lambda trigs: [trigs.p1.y, trigs.p2.y, trigs.p3.y], mesh_triangles))
    y_list = np.array(y_list).flatten()
    y_start, y_end = int_between(np.min(y_list), np.max(y_list))
    y_cen = lst_to_mid(y_start, y_end)
    
    z_list = list(map(lambda trigs: [trigs.p1.z, trigs.p2.z, trigs.p3.z], mesh_triangles))
    z_list = np.array(z_list).flatten()
    z_start, z_end = int_between(np.min(z_list), np.max(z_list))
    z_cen = lst_to_mid(z_start, z_end)
    
    print(x_cen, y_cen, z_cen)
    
    center = []
    for a in x_cen:
        for b in y_cen:
            for c in z_cen:
                center.append(Point3(a, b, c))
    print("Number of Voxels: ", len(center))
    start = timeit.default_timer()
    print("---------------------- Good Luck ---------------------")
    # plot_points(center)
    sq_list = []
    pts = []
    edges_list = []
    # Pre-processing
    for c in center:
        edges = find_edges(c)
        edges_list.append(edges)
    
    for i in range(0, len(edges_list)):
        intersections = find_intersection(mesh_triangles, edges_list[i])
        pts += intersections
        squares = intersection_to_squares(intersections, center[i])
        sq_list += squares
        
        # try:
        #     intersections = find_intersection(mesh_triangles, c)
        #     pts += intersections
        #     squares = intersection_to_squares(intersections, c)
        #     sq_list += squares
        # except Exception as e:
        #     print(e)
    print("---------------------Finished Finding Squares----------")
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return sq_list

if __name__ == "__main__":
    # Sphere
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = 2.5 * np.outer(np.cos(u), np.sin(v))
    # y = 2.5 * np.outer(np.sin(u), np.sin(v))
    # z = 2.5 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Torus

    mesh_3d = generate_mesh()
    # mesh_3d = generate_torus()
    # mesh_3d = potato_chip()
    m_triangles = list(map(lambda trig: Trig(trig[0], trig[1], trig[2]), mesh_3d))
    mesh_triangles = []
    
    for trg in m_triangles:
        if not is_triangle_degenrate(trg):
            mesh_triangles.append(trg)
    
    # file2 = open(r"example.txt", "w+") 
    # for tri in mesh_triangles:
    #     p1 = tri.p1
    #     p2 = tri.p2
    #     p3 = tri.p3
    #     string = str(p1.x) + " " + str(p1.y) + " " +  str(p1.z)  + " " +  str(p2.x) + " " + str(p2.y) + " " + str(p2.z) + " " + str(p3.x) + " " + str(p3.y) + " " + str(p3.z)
    #     file2.write(string + "\n")
    # file2.close()
    
    print("Number of Triangles: ", len(mesh_triangles))
    # triangle_to_voxel(mesh_triangles)
    
    sq_list = solve(mesh_triangles)
    # debug_plot(mesh_triangles, pts, center)
    square_to_voxel(sq_list)

    
    # y_int = int_between(np.min(y), np.max(y))
    # z_int = int_between(np.min(z), np.max(z))

    # x_cen = lst_to_mid(x_int)
    # y_cen = lst_to_mid(y_int)
    # z_cen = lst_to_mid(z_int)
    
    # triangle_to_voxel(mesh_triangles)
    # plt.figure()
    # plt.gca().set_aspect('equal')
    # plt.triplot(triang, 'bo-', lw=1)
    
    # x = np.outer(R + r*np.cos(u), np.cos(v))
    # y = np.outer(R + r*np.cos(u), np.sin(v))
    # z = np.outer(r*np.sin(u), np.ones(np.size(v)))
    
    # x, y, z = x.flatten(), y.flatten(), z.flatten()

    # visualize_surface(x, y, z)
    
    # u, v = symbols('u v')
    # test = plot3d_parametric_surface(
    #     (R + r*cos(u))*cos(v), (R + r*cos(u))*sin(v), r*sin(u),
    # (u, 0, 2 * math.pi), (v, 0, 2*math.pi))
    # print(test)
    # x_int = int_between(np.min(x), np.max(x))
    # y_int = int_between(np.min(y), np.max(y))
    # z_int = int_between(np.min(z), np.max(z))

    # x_cen = lst_to_mid(x_int)
    # y_cen = lst_to_mid(y_int)
    # z_cen = lst_to_mid(z_int)

    # # print(np.max(z))

    # center = []
    # for a in x_int:
    #     for b in y_int:
    #         for c in z_int:
    #             center.append(Point3(a, b, c))

    # plot_points(center)
    # sq_list = []
    # for c in center:
    #     intersections = find_intersection(x, y, z, c)
    #     squares = intersection_to_squares(intersections)
    #     sq_list += squares
        
    # square_to_voxel(sq_list)