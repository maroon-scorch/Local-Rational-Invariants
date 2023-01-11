import numpy as np
import math

# Class representing an n-dimensional point
class PointN:
    def __init__(self, points):
        self.dim = len(points)
        assert self.dim > 0
        self.points = points
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
    return PointN(array.tolist())

def scale_point(pt, factor):
    return to_point(factor*pt.points)

def unit_vector(vector):
    """ Scales the vector to a unit vector """
    return vector / np.linalg.norm(vector)

def dist(p1, p2):
    """ Given 2 points, find the distance between them """
    return np.linalg.norm(p2.vec - p1.vec)

def max_dist(p1, p2):
    """ Returns the L-infinity norm"""
    return np.max(np.absolute(p2.vec - p1.vec))

def midpoint_of(p1, p2):
    return to_point((p1.vec + p2.vec)/2)

def translate(pt, vec):
    assert len(pt) == len(vec)
    return to_point(pt.vec + np.array(vec))
    
    