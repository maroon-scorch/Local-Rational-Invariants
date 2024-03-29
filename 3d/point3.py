import numpy as np
import math

# Class representing a 2-dimensional point (should switch to a list later)
class Point3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        # Probably how we are gonna roll with this:
        self.vec = np.array([x, y, z])
        # The String Representation of the value of the Literal
        self.value = str(self.vec)

    def __repr__(self):
        return str(self.vec)

    def __eq__(self, other):
        if type(other) != Point3:
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
      return hash((self.x, self.y, self.z))
  
    def to_string(self):
        return str(self.x) + " " + str(self.y) + " " + str(self.z)
        # return str(int(2*self.x)) + " " + str(int(2*self.y)) + " " + str(int(2*self.z))
  
    # def setCoordinate(self, new_x, new_y):
    #     self.x = new_x
    #     self.y = new_y
    #     self.vec = np.array([new_x, new_y])
    #     self.value = "(" + str(self.x) + ", " + str(self.y) + ")"

def to_point(lst):
    assert len(lst) == 3
    return Point3(lst[0], lst[1], lst[2])

def to_list(pt):
    return [pt.x, pt.y, pt.z]

def scale_point3(pt, factor):
    return Point3(pt.x*factor, pt.y*factor, pt.z*factor)

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

def approx_contains(lst, point):
    """ Given a list of points and a given point, check if the point is approximately in the list """
    for pt in lst:
        if dist(pt, point) < 0.000001:
            return True
        
    return False

def round_p3(p):
    return Point3(int(round(p.x)), int(round(p.y)), int(round(p.z)))

def midpoint_of(p1, p2):
    return Point3((p1.x + p2.x)/2, (p1.y + p2.y)/2, (p1.z + p2.z)/2)
    
    