import numpy as np
import random
import math

# Class representing an n-dimensional point
class NPoint:
    def __init__(self, point):
        # Array of numbers representing the point
        self.point = point
        # Numpy Array
        self.vec = np.array(self.point)
        # The String Representation of the value of the Literal
        self.value = str(tuple(self.point))

    def __repr__(self):
        return str(tuple(self.point))

    def __eq__(self, other):
        if type(other) != NPoint:
            return False
        return self.point == other.point
    
    def __hash__(self):
      return hash(self.point)

def unit_vector(vector):
    """ Scales the vector to a unit vector """
    return vector / np.linalg.norm(vector)

def angle(p1, p2, p3):
    """ Given 3 points, find the angle at p2 formed by the 3 points """
    vec1 = unit_vector(p1.vec - p2.vec)
    vec2 = unit_vector(p3.vec - p2.vec)
    angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
    
    return angle
    
    
    