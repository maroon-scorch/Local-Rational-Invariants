# Produces a projective plane via a mapping from S^2 to R^4 satisfying the antipodal property
# We will triangulate the domain, so the image of the triangulation would be a triangulation of S^2

# Note, our S^2 will have radius greater than 1, due to technicalities of having to produce unit surfaces
from pointn import *
import sys

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

if __name__ == "__main__":
    input_file = sys.argv[1]
    triangles = read_input(input_file)
    print("Number of Triangles: ", len(triangles))

    x = lambda p: p[0]*p[1]
    y = lambda p: p[0]*p[2]
    z = lambda p: p[1]**2 - p[2]**2
    w = lambda p: 2*p[1]*p[2]

    projective_plane = []
    roman_surface = []
    for p1, p2, p3 in triangles:
        if triangle_area(p1, p2, p3) < 0.01:
            print("BRUH")
        q1 = p1.points
        q2 = p2.points
        q3 = p3.points

        n1 = PointN([x(q1), y(q1), z(q1), w(q1)])
        n2 = PointN([x(q2), y(q2), z(q2), w(q2)])
        n3 = PointN([x(q3), y(q3), z(q3), w(q3)])

        projective_plane.append([n1, n2, n3])

        t1 = PointN([x(q1), y(q1), w(q1)])
        t2 = PointN([x(q2), y(q2), w(q2)])
        t3 = PointN([x(q3), y(q3), w(q3)])

        roman_surface.append([t1, t2, t3])

    # for t in roman_surface:
    #     print(t)

    print("Euler Characteristic: ", euler_characteristic(triangles))
    print("Euler Characteristic: ", euler_characteristic(projective_plane))