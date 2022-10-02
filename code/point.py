import numpy as np
import random


# Class representing a 2-dimensional point (should switch to a list later)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # Probably how we are gonna roll with this:
        self.vec = np.array([x, y])
        # The String Representation of the value of the Literal
        self.value = "(" + str(self.x) + ", " + str(self.y) + ")"

    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def __eq__(self, other):
        if type(other) != Point:
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
      return hash((self.x, self.y))

def unit_vector(vector):
    """ Scales the vector to a unit vector """
    return vector / np.linalg.norm(vector)

def dist(p1, p2):
    """ Given 2 points, find the distance between them """
    return np.linalg.norm(p2.vec - p1.vec)

def angle(p1, p2, p3):
    """ Given 3 points, find the angle at p2 formed by the 3 points """
    vec1 = unit_vector(p1.vec - p2.vec)
    vec2 = unit_vector(p3.vec - p2.vec)
    angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
    
    return angle
    
    
    