from code.main import label_index
import math
import numpy as np
import sympy as sp
from point import *
from edge import *
from main import *

class projection:
    def edge_type(self, point_1, point_2):
        self.vertical = (label_index(point_1) == 0 and label_index(point_2) == 0)
        self.horizontal = (label_index(point_1) == 1 and label_index(point_2) == 1)
        self.crooked = ((label_index(point_1) == 1 and label_index(point_2) == 0)
                   or (label_index(point_1) == 0 and label_index(point_2) == 1))
    
    pt = Point(point_1.x + randrange(0,1) * (point_2.x - point_1.x))
        
    def map_grid_h(self, point_1, point_2, points):
        if side(points, Point(point_1.x, math.ceil(point_1.y))) == True:
            