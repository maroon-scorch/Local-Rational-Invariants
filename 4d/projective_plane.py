# Produces a projective plane via a mapping from S^2 to R^4 satisfying the antipodal property
# We will triangulate the domain, so the image of the triangulation would be a triangulation of S^2

# Note, our S^2 will have radius greater than 1, due to technicalities of having to produce unit surfaces

# Run - python projective_plane.py pointn.py

from pointn import *
import sys, itertools, timeit, time
import scipy.linalg as la
from threading import Thread
from multiprocessing.pool import Pool
from itertools import repeat
epsilon = 0.0001

class intersectionFinder(Thread):
    def __init__(self, start, end, n, complex_list, kernel_list, output_list):  
        Thread.__init__(self)
        self.s = start
        self.e = end
        self.n = n
        self.complex_list = complex_list
        self.kernel_list = kernel_list
        self.output_list = output_list
        self.value = None

    def find(self, complex, n, kernel, output_vectors):
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

        for i in range(self.s, self.e + 1):
            # Iterating through the data of triangles
            for j in range(0, len(linear_list)):
                # Iterating through the data of complex
                whole_matrix = np.array(linear_list[j][0] + kernel[i].tolist())
                output = np.array(linear_list[j][1] + output_vectors[i].tolist())

                if is_consistent(whole_matrix, output):
                    intersection = np.linalg.solve(whole_matrix, output)
                    # print("Intersection: ", intersection)
                    if max_dist(to_point(intersection), PointN(list(complex[j]))) <= 1 + epsilon:
                        point_list.append(intersection.tolist())
        return point_list
        
    def run(self):
        pts = []
        for i in range(0, len(self.complex_list)):
            intersections = self.find(self.complex_list[i], self.n, self.kernel_list, self.output_list)
            pts += intersections

        self.value = pts

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

def euler_characteristic(trig_list):
    vertices = set()
    edges = set()

    for p1, p2, p3 in trig_list:
        vertices.add(p1)
        vertices.add(p2)
        vertices.add(p3)
        edges.add((p1, p2))
        edges.add((p2, p3))
        edges.add((p3, p1))

    # Scuffed way to remove duplicate from edges
    edges = list(edges)
    n = len(edges)
    # print(n)
    pop_index = set()
    for i in range(n):
        for j in range(i + 1, n):
            if edges[j][0] == edges[i][1] and edges[j][1] == edges[i][0]:
                pop_index.add(j)
    pop_index = list(pop_index)
    pop_index.sort(reverse=True)

    for p in pop_index:
        edges.pop(p)
    
    print(len(vertices))
    print(len(edges))

    output = len(vertices) - len(edges) + len(trig_list)
    return output


def triangle_area(a, b, c):
    # Find the un-signed area of the Triangle with vertex a, b, c
    v1 = b.vec - a.vec
    v2 = c.vec - a.vec
    return 1/2*abs(np.linalg.norm(np.cross(v1, v2)))

def center(p):
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

def find_rref(A, b):
    #print(A)
    #print(b)
    B = np.reshape(b, (1, -1))
    matrix = np.concatenate((A, B.T), axis=1)
    #print(matrix)
    (_, rref) = la.qr(matrix)
    
    return rref

def is_consistent(A, b):
    """ A linear system Ax = b has solution if and only if
    (A|b) has the same rank as rank of A """
    B = np.reshape(b, (1, -1))
    matrix = np.concatenate((A, B.T), axis=1)
    return np.linalg.matrix_rank(matrix) == np.linalg.matrix_rank(A)

def find_intersection_alt(complex, n, kernel, output_vectors):
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

    for i in range(0, len(kernel)):
        # Iterating through the data of triangles
        for j in range(0, len(linear_list)):
            # Iterating through the data of complex
            whole_matrix = np.array(linear_list[j][0] + kernel[i].tolist())
            output = np.array(linear_list[j][1] + output_vectors[i].tolist())

            if is_consistent(whole_matrix, output):
                intersection = np.linalg.solve(whole_matrix, output)
                # print("Intersection: ", intersection)
                if max_dist(to_point(intersection), PointN(list(complex[j]))) <= 1 + epsilon:
                    point_list.append(intersection.tolist())
    
    return point_list

def find_intersection(trig_list, complex, n):
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

    # The matrix defining the complex should be the basis vectors 
    # of the kernel, we can optimize this later

    # print(linear_list)
    for tri in trig_list:
        trig_vectors = []
        output_vectors = []
        for i in range(1, len(tri)):
            current_vec = (tri[i].vec - tri[0].vec).tolist()
            trig_vectors.append(current_vec)

        # print("Complex: ", trig_vectors)
        kernel = nullspace(np.array(trig_vectors))
        # print("Kernel: ", kernel)
        output_vectors = np.dot(kernel, tri[0].vec.T)
        # print("Output: ", output_vectors)
        for j in range(0, len(linear_list)):
            whole_matrix = np.array(linear_list[j][0] + kernel.tolist())
            output = np.array(linear_list[j][1] + output_vectors.tolist())
            # print("Whole Matrix: ", whole_matrix)
            # print("Output: ", output)
            
            rref = find_rref(whole_matrix, output)
            
            # print(rref)
            # last_column = rref[-1]
            # print(last_column)

            if is_consistent(whole_matrix, output):
                intersection = np.linalg.solve(whole_matrix, output)
                # print("Intersection: ", intersection)

                if max_dist(to_point(intersection), PointN(list(complex[j]))) <= 1 + epsilon:
                    point_list.append(intersection.tolist())
    
    return point_list
            # if np.linalg.norm(last_column[:-1]) < 0.0000001 and np.abs(last_column[-1]) > 0.0000001:
            #     # This equation has no solution
            #     print("No intersection")
            #     continue
            # else:
            #     intersection = np.linalg.solve(whole_matrix, output)
            #     print("Intersection: ", intersection)
            # print(whole_matrix)

# Trig list is really a list of n-dimensional triangles
def solve(trig_list, n, k):
    # n is the dimension of points
    # k is how many points each triangle has
    # Assume n >= k - 1 and k != 0
    assert n >= k - 1 and k != 0

   
    if k == 1:
        output = []
        for p in trig_list:
            output.append(center(p))
        return output
    else:
        cen = []
        for i in range(0, n):
            x_list = [[p.points[i] for p in tri] for tri in trig_list]
            x_list = np.array(x_list).flatten()
            x_start, x_end = int_between(np.min(x_list), np.max(x_list))
            x_cen = lst_to_mid(x_start, x_end)
            cen.append(x_cen)

        centers = itertools.product(*cen)
        complex_list = []
        if n == k-1:
            complex_list = centers
        else:
            # Attempt to optimize by doing the computation once
            indices = list(itertools.combinations(range(0, n), (k-1)))

            # print(list(indices))
            # The shifts 
            up = lambda x: math.ceil(x)
            down = lambda x: math.floor(x)
            moves = list(itertools.product([up, down], repeat=(k-1)))

            for c in centers:
                # print("Center ", c)
                c_complex = find_subcomplex(c, indices, moves)
                complex_list.append(c_complex)

        # print(complex_list)
        # Finding intersections:
        print("--------------------- Finding Intersections ---------")
        # print(len(complex_list))

        kernel_list = []
        output_list = []
        for tri in trig_list:
            trig_vectors = []
            output_vectors = []
            for i in range(1, len(tri)):
                current_vec = (tri[i].vec - tri[0].vec).tolist()
                trig_vectors.append(current_vec)
            # print("Complex: ", trig_vectors)
            kernel = nullspace(np.array(trig_vectors))
            kernel_list.append(kernel)
            # print("Kernel: ", kernel)
            output_vectors = np.dot(kernel, tri[0].vec.T)
            output_list.append(output_vectors)

        sq_list = []
        pts = []
        # ---------------- Multiprocessing
        N = 10
        with Pool(N) as pool:
            result = pool.starmap(find_intersection_alt, zip(complex_list, repeat(n), repeat(kernel_list), repeat(output_list)))


        # ---------------- Multithreading
        # N = 10
        # cords = np.array_split(range(len(trig_list)), N)
        # thread_list = []
        # # Getting to here took about 0.06 seconds 
        # for i in range(len(cords)):
        #     # create a new thread
        #     start = cords[i][0]
        #     end = cords[i][-1]
        #     thread = intersectionFinder(start, end, n, complex_list, kernel_list, output_list)
        #     # start the thread
        #     thread.start()
        #     thread_list.append(thread)
        
        # # wait for the thread to finish
        # for t in thread_list:
        #     t.join()
        #     data = t.value


        # for i in range(0, len(complex_list)):
        #     # intersections = find_intersection(trig_list, complex_list[i], n)
        #     intersections = find_intersection_alt(complex_list[i], n, kernel_list, output_list)
        #     pts += intersections
        #     # try:
        #     #     intersections = find_intersection(trig_list, complex_list[i], n)
        #     #     pts += intersections
        #     #     # squares = intersection_to_squares(intersections, center[i])
        #     #     # sq_list += squares
        #     # except Exception as e:
        #     #     print(e)
        #     #     print("___________________________________________________")
        print("---------------------Finished Finding Squares----------")


        
        

if __name__ == "__main__":
    input_file = sys.argv[1]
    triangles = read_input(input_file)
    print("Number of Triangles: ", len(triangles))

    start = timeit.default_timer()
    print("Start")
    solve(triangles, 3, 3)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # x = lambda p: p[0]*p[1]
    # y = lambda p: p[0]*p[2]
    # z = lambda p: p[1]**2 - p[2]**2
    # w = lambda p: 2*p[1]*p[2]

    # projective_plane = []
    # roman_surface = []
    # for p1, p2, p3 in triangles:
    #     if triangle_area(p1, p2, p3) < 0.01:
    #         print("BRUH")
    #     q1 = p1.points
    #     q2 = p2.points
    #     q3 = p3.points

    #     n1 = PointN([x(q1), y(q1), z(q1), w(q1)])
    #     n2 = PointN([x(q2), y(q2), z(q2), w(q2)])
    #     n3 = PointN([x(q3), y(q3), z(q3), w(q3)])

    #     projective_plane.append([n1, n2, n3])

    #     t1 = PointN([x(q1), y(q1), w(q1)])
    #     t2 = PointN([x(q2), y(q2), w(q2)])
    #     t3 = PointN([x(q3), y(q3), w(q3)])

    #     roman_surface.append([t1, t2, t3])

    # # for t in roman_surface:
    # #     print(t)

    # print("Euler Characteristic: ", euler_characteristic(triangles))
    # print("Euler Characteristic: ", euler_characteristic(projective_plane))

    # file2 = open(r"rp2.txt", "w+") 
    # for p1, p2, p3 in projective_plane:
    #     string = "{:.5f}".format(p1.points[0]) + " " + "{:.5f}".format(p1.points[1]) + " " +  "{:.5f}".format(p1.points[2])  + " " +  "{:.5f}".format(p1.points[3])
    #     string += "{:.5f}".format(p2.points[0]) + " " + "{:.5f}".format(p2.points[1]) + " " +  "{:.5f}".format(p2.points[2])  + " " +  "{:.5f}".format(p2.points[3])
    #     string += "{:.5f}".format(p3.points[0]) + " " + "{:.5f}".format(p3.points[1]) + " " +  "{:.5f}".format(p3.points[2])  + " " +  "{:.5f}".format(p3.points[3])
    #     file2.write(string + "\n")
    # file2.close()