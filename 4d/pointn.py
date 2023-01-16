import numpy as np
from sympy import Matrix
import math

# Class representing an n-dimensional point
class PointN:
    def __init__(self, points):
        self.dim = len(points)
        assert self.dim > 0
        # The list of coordinates of the point
        self.points = points
        # Numpy Array representing the same point
        self.vec = np.array(points)

    def __repr__(self):
        return str(self.vec)

    def __eq__(self, other):
        if type(other) != PointN:
            return False
        return self.points == other.points

    def __hash__(self):
      return hash((tuple(self.points)))
  
    def to_string(self):
        return str(self.points)

def to_point(array):
    """ Converts a given 1d numpy array to a point """
    return PointN(array.tolist())

def scale_point(pt, factor):
    """ Scales a point by the given factor """
    return to_point(factor*pt.points)

def unit_vector(vector):
    """ Scales the vector to a unit vector """
    return vector / np.linalg.norm(vector)

def dist(p1, p2):
    """ Given 2 points, find the L-2 distance between them """
    return np.linalg.norm(p2.vec - p1.vec)

def max_dist(p1, p2):
    """ Returns the L-infinity distance between the two points """
    return np.max(np.absolute(p2.vec - p1.vec))

def midpoint_of(p1, p2):
    """ Finds the midpoint of the given two points """
    return to_point((p1.vec + p2.vec)/2)

def translate(pt, vec):
    """ Translates the point by a constant vector """
    assert len(pt.points) == len(vec)
    return to_point(pt.vec + np.array(vec))

def barycentric_coordinate(pt, simplex, tot_min=1e-9):
    """ Simplex (non-degenerate) is represented as a list of n-points, representing its vertices.
    The point is assumed to lie on the vector space spanned by the simplex. """
    assert len(simplex) > 1
    first_point = simplex[0]
    matrix = list(map(lambda v: v.vec - first_point.vec, simplex[1:]))
    matrix = np.stack(matrix, axis=0).T
    answer = pt.vec - first_point.vec
    # print("A: ", matrix)
    # print("b: ", answer)

    # Need to check
    # If we have a k-simplex in R^n, k < n, then the matrix here is n \times k
    # This is actually an overdetermined system, so we find k linearly independent
    # rows of the matrix to solve the system
    # k = len(simplex) - 1
    # output = np.matmul(np.linalg.inv(matrix[0:k][0:k]), answer[0:k])
    # last_cord = 1 - np.sum(output)
    # output = np.append(output, last_cord)
    # print("Cropped Output: ", output)
    
    # Alternatively, we could use the least square solution
    output, residual, _, _ = np.linalg.lstsq(matrix, answer, rcond=None)
    assert residual <= tot_min
    last_cord = 1 - np.sum(output)
    output = np.append(output, last_cord)
    # print("Least Square Output: ", output)
    
    return output

def is_point_in_simplex(pt, simplex):
    """ Checks if a given point is contained in the given simplex """
    adjusted_pt = barycentric_coordinate(pt, simplex)
    return np.min(adjusted_pt) >= 0 and np.sum(adjusted_pt) <= 1

# p = PointN([0.4,0.5])
# simplex = [PointN([0, 1]), PointN([1, 2]), PointN([1, 0])]
# output = barycentric_coordinate(p, simplex)
# print(is_point_in_simplex(p, simplex))
# min_tol = 1e-9

# A = [[1, 1], [2, 2], [2, 0]]
# A = Matrix(A)
# print(A.rref())
# _, r = np.linalg.qr(A)
# print(r)

# indep = np.where(np.abs(r.diagonal()) >  min_tol)[0]
# print(A[:, indep])
# print("Independent columns are: {}".format(indep))