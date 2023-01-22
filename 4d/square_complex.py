from pointn import *
import sys, itertools, timeit, time
from itertools import repeat
import matplotlib.pyplot as plt
import numpy as np
epsilon = 0
#epsilon = 0.0001

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors

def square_to_voxel(square_list):
    """ Visualization script that shows the list of squares to a cubical
    surface """
    ax = a3.Axes3D(plt.figure())
    for p1, p2, p3, p4 in square_list:
        vtx = np.array([p1.vec, p2.vec, p3.vec, p4.vec])
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_alpha(0.5)
        ax.scatter(p1.points[0], p1.points[1], p1.points[2], c = '#FF0000')
        ax.scatter(p2.points[0], p2.points[1], p2.points[2], c = '#FF0000')
        ax.scatter(p3.points[0], p3.points[1], p3.points[2], c = '#FF0000')
        ax.scatter(p4.points[0], p4.points[1], p4.points[2], c = '#FF0000')
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    plt.show()
    
def triangle_to_voxel(trig_list):
    """ Visualization script that shows the list of triangles """
    ax = a3.Axes3D(plt.figure())
    for p1, p2, p3 in trig_list:
        vtx = np.array([p1.vec, p2.vec, p3.vec])
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_alpha(0.5)
        # tri.set_color(colors.rgb2hex(np.random.rand(3)))
        
        ax.scatter(p1.points[0], p1.points[1], p1.points[2], c = '#FF0000')
        ax.scatter(p2.points[0], p2.points[1], p2.points[2], c = '#FF0000')
        ax.scatter(p3.points[0], p3.points[1], p3.points[2], c = '#FF0000')
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    plt.show()

def find_edges(center):
    line_list = []
    x, y, z = center[0], center[1], center[2]
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
        p_s = [start[0], start[1], start[2]]
        p_e = [end[0], end[1], end[2]]
        if abs(np.linalg.norm(np.array(p_s) - np.array(p_e)) - 1) < 0.0000001:
            line_list.append([p_s, p_e])
    return line_list

def triangle_grid_to_voxel(trig_list, center, intersections):
    ax = a3.Axes3D(plt.figure())
    for p1, p2, p3 in trig_list:
        vtx = np.array([p1.vec, p2.vec, p3.vec])
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_alpha(0.5)
        # tri.set_color(colors.rgb2hex(np.random.rand(3)))
        
        ax.scatter(p1.points[0], p1.points[1], p1.points[2], c = '#FF0000')
        ax.scatter(p2.points[0], p2.points[1], p2.points[2], c = '#FF0000')
        ax.scatter(p3.points[0], p3.points[1], p3.points[2], c = '#FF0000')
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
        
    grid_edge_list = find_edges(center)
    
    for i, ed in enumerate(grid_edge_list):
        start = ed[0]
        end = ed[1]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k-')
        
    for p in intersections:
        ax.scatter(p.points[0], p.points[1], p.points[2], c='#FF0000')
    
    plt.show()

def read_input(inputFile):
    """ Read and parse the input file, returning the list of triangles """
    trig_list = []
    with open(inputFile, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            p1 = PointN([float(tokens[0]), float(tokens[1]), float(tokens[2])])
            p2 = PointN([float(tokens[3]), float(tokens[4]), float(tokens[5])])
            p3 = PointN([float(tokens[6]), float(tokens[7]), float(tokens[8])])
            tri = [p1, p2, p3]
            trig_list.append(tri)
    return trig_list
    # trig_list = []
    # with open(inputFile, "r") as f:
    #     for line in f.readlines():
    #         tokens = line.strip().split()
    #         p1 = PointN([float(tokens[0]), float(tokens[1])])
    #         p2 = PointN([float(tokens[2]), float(tokens[3])])
    #         tri = [p1, p2]
    #         trig_list.append(tri)
    # return trig_list

def read_input_vertex(inputFile):
    trig_list = []
    with open(inputFile, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            p1 = PointN([float(tokens[0]), float(tokens[1])])
            trig_list.append(p1)

    edges = []
    for i in range(len(trig_list)):
        edges.append((trig_list[i], trig_list[(i+1) % len(trig_list)]))
    return edges


def visualize_edges(grid_edge_list):
    for i, ed in enumerate(grid_edge_list):
        start = ed[0]
        end = ed[1]
        plt.plot([start.points[0], end.points[0]], [start.points[1], end.points[1]], 'k-')
    plt.show()

# =======================================
# The functions above are irrelevant to the main algorithm
# =======================================

def find_center(p):
    # Given a point p, assuming in general position, finds the center
    # of the unit voxel it is contained in.

    # Note if the point has some integer coordinates, this means it is
    # contained in some k-subcomplex of the general unit voxel, in that
    # case the code returns the center of that k-simplex.

    point_list = p.points

    max_list = list(map(lambda p: math.ceil(p), point_list))
    min_list = list(map(lambda p: math.floor(p), point_list))

    max_pt = PointN(max_list)
    min_pt = PointN(min_list)

    return midpoint_of(max_pt, min_pt)


def custom_round(x):
    """ Rounds the float to the nearest 0.5-value on the real line
    (ie. the possible outputs are 1.5, -0.5, 0.5, 1.5, etc."""
    z = round(x)
    left = z - 0.5
    right = z + 0.5
    return left if abs(x - left) < abs(x - right) else right

def is_int(x):
    return abs(x - round(x)) < 0.00001

# Given two numbers x, y, find the 0.5-values between them inclusive
def int_between(x, y):
    
    # print(x, y)
    if x == y:
        if is_int(x):
            return x - 0.5, y + 0.5
        
        new_x = custom_round(x)
        
        if new_x < x:
            return new_x, new_x + 1
        
        if new_x > x:
            return new_x - 1, new_x
        
        if new_x == x:
            return new_x - 1, new_x + 1
        
    
    if x < y:
        if is_int(x):
            new_x = x - 0.5
        else:
            new_x = custom_round(x)
            
        if is_int(y):
            new_y = y + 0.5
        else:
            new_y = custom_round(y)
        
        return new_x, new_y
    else:
        if is_int(x):
            new_x = x + 0.5
        else:
            new_x = custom_round(x)
            
        if is_int(y):
            new_y = y - 0.5
        else:
            new_y = custom_round(y)
        
        return new_y, new_x

def lst_to_mid(start, end):
    # Given two 0.5-values, fills the middle up.
    # For example, if start = -0.5, end = 2.5, the result is
    # -0.5, 0.5, 1.5, 2.5
    current = start
    mid = []
    while current < end:
        mid.append(current)
        current += 1
    mid.append(current)
    
    return mid

def find_subcomplex(center, indices, moves):
    """ Given a center contained in a unit cube, finds its k-subfaces. Each subface is represented
    by a vector pointing to its center, it turns out this is sufficient """

    # The parameters indices and moves are just here for computational ease
    # Conceptually, to obtain all the centers we want, we would perform a sequence of moves in
    # each orthogonal direction by +0.5 or -0.5
    # The indices contains a list of index where we would perform the moves
    # The moves indicate whether to +0.5 or -0.5 

    cp = PointN(list(center))
    # print(cp)
    output = []
    for index in indices:
        for m in moves:
            current = cp.vec.tolist()
            # print(current)
            for j in range(0, len(index)):
                current_pos = list(index)[j]
                current[current_pos] = m[j](current[current_pos])
            output.append(tuple(current))
    return output
        
# https://stackoverflow.com/questions/49852455/how-to-find-the-null-space-of-a-matrix-in-python-using-numpy
def nullspace(A, atol=1e-13, rtol=0):
    # Finds the null space of the given matrix
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    # ns = vh[nnz:].conj().T
    ns = vh[nnz:].conj()
    return ns

def is_consistent(A, b):
    """ Checks if Ax = b is consistent or not.
    Note that a linear system Ax = b has solution if and only if
    (A|b) has the same rank as rank of A """
    B = np.reshape(b, (1, -1))
    matrix = np.concatenate((A, B.T), axis=1)
    return np.linalg.matrix_rank(matrix) == np.linalg.matrix_rank(A)

def find_intersection(complex, tri, n):
    # Given the list of subfaces of a certain center and a simplex
    # Computes the intersection points between them.

    # The idea is that, the triangle is determined by its
    # a basis of its orthogonal vectors (ie.
    # the basis of the kernel of the matrix formed by the edges of the triangles)
    # Each subface is determined by equations of the form
    # x_i = n, that restrict certain coordinates to integers

    # The idea is to combine the equations and just solve the system!

    point_list = []
    linear_list = []
    # Computing the equations representing each subface
    for c in complex:
        matrix_append = []
        output = []
        for i in range(0, len(c)):
            if type(c[i]) == type(0):
                entry = [0 for i in range(n)]
                entry[i] = 1
                # This is the equation
                # x_i = c[i]
                matrix_append.append(entry)
                output.append(c[i])
        
        linear_list.append((matrix_append, output))

    # This part is computing the equations representing the triangle
    trig_vectors = []
    output_vectors = []
    for i in range(1, len(tri)):
        current_vec = (tri[i].vec - tri[0].vec).tolist()
        trig_vectors.append(current_vec)
    # print("Complex: ", trig_vectors)
    kernel = nullspace(np.array(trig_vectors))
    # print("Kernel: ", kernel)
    output_vectors = np.dot(kernel, tri[0].vec.T)

    # We are intersecting each subface with the given triangle:
    for j in range(0, len(linear_list)):
        # Iterating through the data of complex

        # We are making the coefficient matrix and the output matrix
        whole_matrix = np.array(linear_list[j][0] + kernel.tolist())
        output = np.array(linear_list[j][1] + output_vectors.tolist())

        # We should check if this system is consistent. If it is inconsistent, there's no intersection.
        if is_consistent(whole_matrix, output):
            value = np.linalg.matrix_rank(whole_matrix)
            
            # Finding the intersection, this should be unique.
            intersection = np.linalg.solve(whole_matrix, output)
            intersection = to_point(intersection)

            if value < n:
                # Reaching here means that we have a CONSISTENT system
                # Ax = b where A does not ahve full rank, so we have
                # infinitely many solutions. Assuming general position, we should never reach here.
                print(value)
                print(whole_matrix)
                print(output)
                print(intersection)
                print("This actually has infinitely many solutions...")
                continue

            # print("Intersection: ", intersection)
            # The intersection point should be contained in the simplex AND is contained in the current subface.
            if is_point_in_simplex(intersection, tri) and max_dist(intersection, PointN(complex[j])) <= 0.5 + epsilon:
            # if max_dist(intersection, PointN(complex[j])) <= 0.5 + epsilon:
                point_list.append(intersection)

    return point_list

def find_centers_on_trig(triangle, n):
    """ Given a simplex in R^n, find the bound on the triangle
    (this is the smallest box with integer dimensions on the grid that contains the triangle)
    then we enumerate all the unit voxels contained in this box and returns them. """
    # The idea is that - if the area of the triangle is really small, there won't be many 
    # points in the output.
    # print(triangle)
    centers = []
    
    for i in range(0, n):
        x_list = [p.points[i] for p in triangle]
        x_list = np.array(x_list).flatten()
        x_min, x_max = np.min(x_list), np.max(x_list)
        x_start, x_end = int_between(x_min, x_max)
        x_cen = lst_to_mid(x_start, x_end)
        centers.append(x_cen)

    return centers

def intersection_to_squares(center, intersections, n, k):
    # Given a center and a list of intersections its subfaces has
    # converts the intersections to squares.
    squares = []
    # print(len(intersections))

    # For every point in the intersection, we construct the appropriate
    # cube between the center and that point.
    for p in intersections:
        c = find_center(p)
        diff = c.vec - center.vec

        # print("p: ", p)
        # print("c: ", c)
        # print("Center: ", center)

        idx = np.nonzero(diff)[0].tolist()
        
        # This is exactly the construction of a Boolean Lattice
        # We inductively make copy of the previous lattice shifted by a vector
        sq = [center]
        for i in idx:
            current_vec = [0 for i in range(n)]
            current_vec[i] = diff[i]
            new_pts = []
            for s in sq:
                new_pts.append(translate(s, current_vec))
            sq = sq + new_pts
        
        squares.append(sq)

    return squares

# Trig list is really a list of n-dimensional triangles
def solve_debug(trig_list, n, k):
    # n is the dimension of points
    # k is how many points each triangle has
    # Assume n >= k - 1 and k != 0
    assert n >= k - 1 and k != 0

    # This is the setup for indexes and moves to find the subfaces of a unit cube
    # We use this part in find_subcomplex
    indices = list(itertools.combinations(range(0, n), (k-1)))
    up = lambda x: math.ceil(x)
    down = lambda x: math.floor(x)
    moves = list(itertools.product([up, down], repeat=(k-1)))

    # This will become a dictionary whose:
    # key is the center of the unit cube
    # the output is the set of intersection points its subfaces make with the triangle
    centers_record = {}

    tri_list = []
    i_list = []
    for tri in trig_list:
        cen = find_centers_on_trig(tri, n)
        
        # centers = itertools.product(*cen)
        c = (1.5, 0.5, -2.5)
        cpx = find_subcomplex(c, indices, moves)
        # Computes the intersection
        pts = find_intersection(cpx, tri, n)
        
        if pts != []:
            print(pts)
            print(tri)
            tri_list.append(tri)
            i_list += pts
            print("----------------")
        # Put it in the dictionary
        key = PointN(c)
        if key in centers_record:
            centers_record[key].update(pts)
        else:
            centers_record[key] = set(pts)
        
        
        # For each center of the unit voxels
        # for c in centers:
        #     # Finds its subfaces
        #     cpx = find_subcomplex(c, indices, moves)
        #     # Computes the intersection
        #     pts = find_intersection(cpx, tri, n)
            
        #     # Put it in the dictionary
        #     key = PointN(c)
        #     if key in centers_record:
        #         centers_record[key].update(pts)
        #     else:
        #         centers_record[key] = set(pts)
    triangle_grid_to_voxel(tri_list, [1.5, 0.5, -2.5], i_list)
    cr = list(centers_record.keys())
    # print(cr)
    # print(cr)
    
    square_list = []
    # Converts each intersection to the squares.
    for c in cr:
        squares = intersection_to_squares(c, centers_record[c], n, k)
        square_list += squares
        # print(squares)
    
    # local = []
    # ky = PointN([1.5, 0.5, -2.5])
    # for sq in square_list:
    #     if ky in sq:
    #         local.append(sq)
            
    # for sq in local:
    #     temp = sq[2]
    #     sq[2] = sq[3]
    #     sq[3] = temp
    
    # square_to_voxel(local)

    return square_list

# Trig list is really a list of n-dimensional triangles
def solve_old(trig_list, n, k):
    # n is the dimension of points
    # k is how many points each triangle has
    # Assume n >= k - 1 and k != 0
    assert n >= k - 1 and k != 0

    # This is the setup for indexes and moves to find the subfaces of a unit cube
    # We use this part in find_subcomplex
    indices = list(itertools.combinations(range(0, n), (k-1)))
    up = lambda x: math.ceil(x)
    down = lambda x: math.floor(x)
    moves = list(itertools.product([up, down], repeat=(k-1)))

    # This will become a dictionary whose:
    # key is the center of the unit cube
    # the output is the set of intersection points its subfaces make with the triangle
    centers_record = {}

    center_set = set()
    for tri in trig_list:
        cen = find_centers_on_trig(tri, n)
        centers = itertools.product(*cen)
        center_set.update(centers)
    
    for tri in trig_list:
        # For each center of the unit voxels
        for c in center_set:
            # Finds its subfaces
            cpx = find_subcomplex(c, indices, moves)
            # Computes the intersection
            pts = find_intersection(cpx, tri, n)
            
            # Put it in the dictionary
            key = PointN(c)
            if key in centers_record:
                centers_record[key].update(pts)
            else:
                centers_record[key] = set(pts)
    
    cr = list(centers_record.keys())
    square_list = []
    # Converts each intersection to the squares.
    for c in cr:
        squares = intersection_to_squares(c, centers_record[c], n, k)
        square_list += squares
        # print(squares)

    return square_list
# -----------------------------------------------------------------------------
# Old Algorithm in R^3
# -----------------------------------------------------------------------------
def find_edges_3d(center):
    line_list = []
    x, y, z = center[0], center[1], center[2]
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
        p_s = [start[0], start[1], start[2]]
        p_e = [end[0], end[1], end[2]]
        if abs(np.linalg.norm(np.array(p_s) - np.array(p_e)) - 1) < 0.0000001:
            line_list.append([PointN(p_s), PointN(p_e)])
    return line_list

def tetrahedron_volume(a, b, c, d):
    # Find the signed volume of the Tetrahedron with vertex a, b, c, d
    input_mat = np.array([
        [a.points[0], a.points[1], a.points[2], 1],
        [b.points[0], b.points[1], b.points[2], 1],
        [c.points[0], c.points[1], c.points[2], 1],
        [d.points[0], d.points[1], d.points[2], 1]
    ])
    det = np.linalg.det(input_mat)
    return 1/6*det

def find_normal_vec(trg):
    p1, p2, p3 = trg[0], trg[1], trg[2]
    plane_vec_1 = p2.vec - p1.vec
    plane_vec_2 = p3.vec - p1.vec
    return np.cross(plane_vec_1, plane_vec_2)

def triangle_area(a, b, c):
    # Find the un-signed area of the Triangle with vertex a, b, c
    v1 = b.vec - a.vec
    v2 = c.vec - a.vec
    return 1/2*abs(np.linalg.norm(np.cross(v1, v2)))

def is_point_on_triangle(trg, pt):
    # pt is coplanar with the tirangle
    a, b, c = trg[0], trg[1], trg[2]
    
    A = triangle_area(a, b, c)
    A1, A2, A3 = triangle_area(a, b, pt), triangle_area(a, c, pt), triangle_area(b, c, pt)
    
    return abs(A - (A1 + A2 + A3)) < 0.00001

def segment_cross_triangle(trg, start, end):
    # Given a triangle and a line segment defined by start and end point;
    # finds the intersection of the line segment with the triangle (if exist)
    a, b, c = trg[0], trg[1], trg[2]
    vol_start = tetrahedron_volume(a, b, c, start)
    vol_end = tetrahedron_volume(a, b, c, end)
    
    if vol_start == 0 and vol_end == 0:
        print((trg, start, end))
        print("Shouldn't happen")
        return False, None
    elif vol_start == 0:
        # This is a grid point
        if is_point_in_simplex(start, trg):
        # if is_point_on_triangle(trg, start):
            return True, start
        else:
            return False, None
    elif vol_end == 0:
        # This is a grid point
        if is_point_in_simplex(end, trg):
        # if is_point_on_triangle(trg, end):
            return True, end
        else:
            return False, None
    elif np.sign(vol_end) == np.sign(vol_start):
        # This has no intersections
        return False, None
    else:
        # This has an intersection!
        diff = start.vec - end.vec
        normal_vec = find_normal_vec(trg)
        
        # Optimization since we know this is a grid line:
        temp = diff.tolist()
        if temp[1] == 0 and temp[2] == 0:
            x = (normal_vec[1]*(a.points[1] - start.points[1]) + normal_vec[2]*(a.points[2] -start.points[2]))/normal_vec[0] + a.points[0] 
            output = PointN([x, start.points[1], start.points[2]])
        elif temp[0] == 0 and temp[2] == 0:
            y = (normal_vec[0]*(a.points[0] - start.points[0]) + normal_vec[2]*(a.points[2] - start.points[2]))/normal_vec[1] + a.points[1]
            output = PointN([start.points[0], y, start.points[2]])
        elif temp[0] == 0 and temp[1] == 0:
            z = (normal_vec[0]*(a.points[0] - start.points[0]) + normal_vec[1]*(a.points[1] - start.points[1]))/normal_vec[2] + a.points[2]
            output = PointN([start.points[0], start.points[1], z])
        else:
            print("Shouldn't happen - same start and end")
            return False, None
        
        s1 = is_point_on_triangle(trg, output)
        s2 = is_point_in_simplex(output, trg)
        
        if s1 != s2:
            print("-------------------------")
            print(output)
            print(trg)
            print(s1)
            print(s2)
            print(barycentric_coordinate(output, trg))
        
        if is_point_in_simplex(output, trg):
            return True, output
        else:
            return False, output

def find_intersection_3d(trig, edges):
    point_list = []
    
    for line_start, line_end in edges:
        has_points, point = segment_cross_triangle(trig, line_start, line_end)
        if has_points:
            point_list.append(point)
    
    point_list = list(set(point_list))
    return point_list

def solve_3d(trig_list, n, k):
    # n is the dimension of points
    # k is how many points each triangle has
    # Assume n >= k - 1 and k != 0
    assert n >= k - 1 and k != 0

    # This is the setup for indexes and moves to find the subfaces of a unit cube
    # We use this part in find_subcomplex
    indices = list(itertools.combinations(range(0, n), (k-1)))
    up = lambda x: math.ceil(x)
    down = lambda x: math.floor(x)
    moves = list(itertools.product([up, down], repeat=(k-1)))

    # This will become a dictionary whose:
    # key is the center of the unit cube
    # the output is the set of intersection points its subfaces make with the triangle
    centers_record = {}

    center_set = set()
    # for tri in trig_list:
    #     cen = find_centers_on_trig(tri, n)
    #     centers = itertools.product(*cen)
    #     center_set.update(centers)
    
    for tri in trig_list:
        cen = find_centers_on_trig(tri, n)
        center_set = itertools.product(*cen)
        # For each center of the unit voxels
        for c in center_set:
            # Finds its subfaces
            edges = find_edges_3d(c)
            # Computes the intersection
            pts = find_intersection_3d(tri, edges)
            
            # Finds its subfaces
            # cpx = find_subcomplex(c, indices, moves)
            # # Computes the intersection
            # pts = find_intersection(cpx, tri, n)
            
            # Put it in the dictionary
            key = PointN(c)
            if key in centers_record:
                centers_record[key].update(pts)
            else:
                centers_record[key] = set(pts)
    
    cr = list(centers_record.keys())
    square_list = []
    # Converts each intersection to the squares.
    for c in cr:
        squares = intersection_to_squares(c, centers_record[c], n, k)
        square_list += squares
        # print(squares)

    return square_list

# Trig list is really a list of n-dimensional triangles
def solve(trig_list, n, k):
    # n is the dimension of points
    # k is how many points each triangle has
    # Assume n >= k - 1 and k != 0
    assert n >= k - 1 and k != 0

    # This is the setup for indexes and moves to find the subfaces of a unit cube
    # We use this part in find_subcomplex
    indices = list(itertools.combinations(range(0, n), (k-1)))
    up = lambda x: math.ceil(x)
    down = lambda x: math.floor(x)
    moves = list(itertools.product([up, down], repeat=(k-1)))

    # This will become a dictionary whose:
    # key is the center of the unit cube
    # the output is the set of intersection points its subfaces make with the triangle
    centers_record = {}


    for tri in trig_list:
        cen = find_centers_on_trig(tri, n)
        centers = itertools.product(*cen)

        # For each center of the unit voxels
        for c in centers:
            # Finds its subfaces
            cpx = find_subcomplex(c, indices, moves)
            # Computes the intersection
            pts = find_intersection(cpx, tri, n)
            
            # Put it in the dictionary
            key = PointN(c)
            if key in centers_record:
                centers_record[key].update(pts)
            else:
                centers_record[key] = set(pts)
    
    cr = list(centers_record.keys())
    square_list = []
    # Converts each intersection to the squares.
    for c in cr:
        squares = intersection_to_squares(c, centers_record[c], n, k)
        square_list += squares
        # print(squares)

    return square_list 

def square_to_string(sq):
    output = ""
    for p in sq:
        for item in p.points:
            output += str(item) + " "
    return output 

if __name__ == "__main__":
    input_file = sys.argv[1]
    triangles = read_input(input_file)
    # triangles = read_input_vertex(input_file)
    print("Number of Triangles: ", len(triangles))
    # triangle_to_voxel(triangles)
    
    triangle_grid_to_voxel(triangles, [1.5, 0.5, -2.5], [])
    
    input = []
    # for p1, p2, p3 in triangles:
    #     q1 = translate(p1, [0.01, 0.008, 0.006])
    #     q2 = translate(p2, [0.001, 0.1, 0.035])
    #     q3 = translate(p3, [0.04, 0.002, 0.01])
    #     input.append([q1, q2, q3])

    # print(input)
    # visualize_edges(input)

    start = timeit.default_timer()
    print("Start")
    square = solve_3d(triangles, 3, 3)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    for sq in square:
        temp = sq[2]
        sq[2] = sq[3]
        sq[3] = temp

    # file2 = open(r"square.txt", "w+") 
    # for sq in square:
    #     file2.write(square_to_string(sq) + "\n")
    # file2.close()

    # print(square)
    square_to_voxel(square)
