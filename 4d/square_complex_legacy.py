from legacy_pointn import *
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

def read_input(inputFile):
    """ Read and parse the input file, returning the list of triangles """
    trig_list = []
    
    with open(inputFile, "r") as f:
        first_line = f.readline().split()
        n = int(first_line[0])
        k = int(first_line[1])
        
        for line in f.readlines():
            tokens = line.strip().split()
            assert len(tokens) % n == 0
            tokens = [float(elt) for elt in tokens]
            
            chunks = [tokens[i:i + n] for i in range(0, len(tokens), n)]
            print(chunks)
            tri = []
            for c in chunks:
                current_point = PointN(c)
                tri.append(current_point)
            assert len(tri) == k
            trig_list.append(tri)
    return trig_list, n, k

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

def find_center(p):
    # Given a point p, assuming in general position, finds the center
    # of the unit voxel it is contained in
    point_list = p.points

    max_list = list(map(lambda p: math.ceil(p), point_list))
    min_list = list(map(lambda p: math.floor(p), point_list))

    max_pt = PointN(max_list)
    min_pt = PointN(min_list)

    return midpoint_of(max_pt, min_pt)


def custom_round(x):
    z = round(x)
    left = z - 0.5
    right = z + 0.5
    return left if abs(x - left) < abs(x - right) else right
    
# Given two numbers x, y, find the integers between them inclusive
def int_between(x, y):
    # print(x, y)
    if x < y:
        return custom_round(x), custom_round(y)
    else:
        return custom_round(y), custom_round(x)

def lst_to_mid(start, end):
    current = start
    mid = []
    while current < end:
        mid.append(current)
        current += 1
    mid.append(current)
    
    return mid

def find_subcomplex(center, indices, moves):
    """ Given a center contained in a unit cube and number a, finds the n-a dimensional
    sub-faces of the cube. """
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
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    # ns = vh[nnz:].conj().T
    ns = vh[nnz:].conj()
    return ns

def is_consistent(A, b):
    """ A linear system Ax = b has solution if and only if
    (A|b) has the same rank as rank of A """
    B = np.reshape(b, (1, -1))
    matrix = np.concatenate((A, B.T), axis=1)
    return np.linalg.matrix_rank(matrix) == np.linalg.matrix_rank(A)

def find_intersection(complex, tri, n):
    point_list = []
    linear_list = []
    for c in complex:
        matrix_append = []
        output = []
        for i in range(0, len(c)):
            if type(c[i]) == type(0):
                entry = [0 for i in range(n)]
                entry[i] = 1
                matrix_append.append(entry)
                output.append(c[i])
        
        linear_list.append((matrix_append, output))

    trig_vectors = []
    output_vectors = []
    for i in range(1, len(tri)):
        current_vec = (tri[i].vec - tri[0].vec).tolist()
        trig_vectors.append(current_vec)
    # print("Complex: ", trig_vectors)
    kernel = nullspace(np.array(trig_vectors))
    # print("Kernel: ", kernel)
    output_vectors = np.dot(kernel, tri[0].vec.T)

    for j in range(0, len(linear_list)):
        # Iterating through the data of complex
        whole_matrix = np.array(linear_list[j][0] + kernel.tolist())
        output = np.array(linear_list[j][1] + output_vectors.tolist())

        if is_consistent(whole_matrix, output):
            value = np.linalg.matrix_rank(whole_matrix)

            intersection = np.linalg.solve(whole_matrix, output)
            intersection = to_point(intersection)

            if value < n:
                print(value)
                print(whole_matrix)
                print(output)
                print(intersection)
                print("This actually has infinitely many solutions...")
                continue

            # print("Intersection: ", intersection)
            # if max_dist(to_point(intersection), PointN(list(complex[j]))) <= 1 + epsilon:
            if is_point_in_simplex(intersection, tri) and max_dist(intersection, PointN(complex[j])) <= 0.5 + epsilon:
                point_list.append(intersection)

    return point_list

def find_centers_on_trig(triangle, n):
    """ Given a simplex in R^n, find the bound on the triangle """
    # print(triangle)
    centers = []
    for i in range(0, n):
        x_list = [p.points[i] for p in triangle]
        x_list = np.array(x_list).flatten()
        x_start, x_end = int_between(np.min(x_list), np.max(x_list))
        x_cen = lst_to_mid(x_start, x_end)
        centers.append(x_cen)

    return centers

def intersection_to_squares(center, intersections, n, k):
    squares = []
    # print(len(intersections))
    for p in intersections:
        c = find_center(p)
        diff = c.vec - center.vec

        # print("p: ", p)
        # print("c: ", c)
        # print("Center: ", center)

        idx = np.nonzero(diff)[0].tolist()
        # This is exactly the construction of a Boolean Lattice
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
def solve(trig_list, n, k):
    # n is the dimension of points
    # k is how many points each triangle has
    # Assume n >= k - 1 and k != 0
    assert n >= k - 1 and k != 0
    indices = list(itertools.combinations(range(0, n), (k-1)))
    up = lambda x: math.ceil(x)
    down = lambda x: math.floor(x)
    moves = list(itertools.product([up, down], repeat=(k-1)))

    centers_record = {}

    for tri in trig_list:
        cen = find_centers_on_trig(tri, n)
        centers = itertools.product(*cen)

        for c in centers:
            cpx = find_subcomplex(c, indices, moves)
            pts = find_intersection(cpx, tri, n)
            
            key = PointN(c)
            if key in centers_record:
                centers_record[key].update(pts)
            else:
                centers_record[key] = set(pts)
    
    cr = list(centers_record.keys())
    square_list = []
    for c in cr:
        squares = intersection_to_squares(c, centers_record[c], n, k)
        square_list += squares
        # print(squares)

    return square_list     

def visualize_edges(grid_edge_list):
    for i, ed in enumerate(grid_edge_list):
        start = ed[0]
        end = ed[1]
        plt.plot([start.points[0], end.points[0]], [start.points[1], end.points[1]], 'k-')
    plt.show()

if __name__ == "__main__":
    input_file = sys.argv[1]
    triangles, n, k = read_input(input_file)
    print(n, k)
    print("Number of Triangles: ", len(triangles))
    
    if (n, k) == (2, 2):
        visualize_edges(triangles)
    if (n, k) == (3, 3):
        triangle_to_voxel(triangles)
        # triangle_grid_to_voxel(triangles, [1.5, 0.5, -2.5], [])
    
    test = []
    for i in range(len(triangles)):
        p1 = triangles[i][0].points[0] + 0.1
        p2 = triangles[i][0].points[1] + 0.005
        
        q1 =  triangles[i][1].points[0] + 0.1
        q2 = triangles[i][1].points[1] + 0.005
        
        test.append([PointN([p1, p2]), PointN([q1, q2])])
        
    for t in test:
        print(t)
    
    # input = []
    # for p1, p2, p3 in triangles:
    #     q1 = translate(p1, [0.01, 0.008, 0.006])
    #     q2 = translate(p2, [0.001, 0.1, 0.035])
    #     q3 = translate(p3, [0.04, 0.002, 0.01])
    #     input.append([q1, q2, q3])

    # print(input)
    # visualize_edges(input)

    start = timeit.default_timer()
    print("Start")
    square = solve(triangles, n, k)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    if (n, k) == (2, 2):
        visualize_edges(square)
    
    
    if (n, k) == (3, 3):
        for sq in square:
            temp = sq[2]
            sq[2] = sq[3]
            sq[3] = temp

        square_to_voxel(square)