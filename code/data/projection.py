from main import label_index
import math
import numpy as np
import sympy as sp
from point import *
from edge import *
from main import *

# Assumption: We always map the polygonal curve to the RIGHT side.
#             The edge has the direction from point_1 poinitng to point_2.
class projection:
    def __init__(self, point_1, point_2):
        self.start = point_1
        self.end = point_2
    
    def edge_type(self, point_1, point_2):
        self.vertical = (label_index(point_1) == 0 and label_index(point_2) == 0)
        self.horizontal = (label_index(point_1) == 1 and label_index(point_2) == 1)
        self.crooked = ((label_index(point_1) == 1 and label_index(point_2) == 0) 
                         or (label_index(point_1) == 0 and label_index(point_2) == 1))
        return self.vertical, self.horizontal, self.crooked
    
    def edge_label(self, point_1, point_2):
        self.vertical = (label_index(point_1) == 0 and label_index(point_2) == 0)
        self.horizontal = (label_index(point_1) == 1 and label_index(point_2) == 1)
        self.crooked = ((label_index(point_1) == 1 and label_index(point_2) == 0) 
                         or (label_index(point_1) == 0 and label_index(point_2) == 1))
        if self.vertical:
            index = 1
        elif self.horizontal:
            index = -1
        else: #self.crooked
            index = 0    
    
    # # Map for horizontal edge    
    # def map_grid_h(self, point_1, point_2, p):
    #     epsilon = 0.000001
    
    #     if is_on_edge(point_1, point_2, p):    
    #         if is_right(point_1, point_2, Point(point_1.x, math.ceil(point_1.y))):
    #             map = lambda p: Point(p.x, math.ceil(p.y))
    #         else:
    #             map = lambda p: Point(p.x, math.floor(p.y))
    #         return p
        
    # """Given an edge that points from point_1 to point_2, we define the horizontal, vertical,
    # and the radial projecting map"""    
    
    
    # # Map for vertical edge 
    # def map_grid_v(self, point_1, point_2, p):
    #     epsilon = 0.000001
    
    #     if is_on_edge(point_1, point_2, p):    
    #         if is_right(point_1, point_2, Point(math.ceil(point_1.x), point_1.y)):
    #             map = lambda p: Point(math.ceil(p.x), p.y)
    #         else:
    #             map = lambda p: Point(math.floor(p.x), p.y)
    #         return p
    
     
    # # Map for crooked edge  
    # def map_grid_r(self, point_1, point_2, p):
    #     epsilon = 0.000001
    
    #     if is_on_edge(point_1, point_2, p):
    #     #    f_ind = label_index(point_1)
    #     #    s_ind = label_index(point_2)
    #         if (label_index(point_1) == 1):
    #             p0 = Point(point_2.x, point_1.y)
    #         else:
    #             p0 = Point(point_1.x, point_2.y)
                
    #         if (is_left(point_1, point_2, p0)):
    #             map = lambda p: Point( #radial project to an image larger than 2
                    
    #             )
    #         else:
    #             map = lambda p: Point(#radial project to an image smaller than 2)
                
        
        
        
        
        
               
    # #   if is_right(point_1, point_2, Point(math.ceil(point_1.x), point_1.y)):
    # #      map = lambda p: Point(math.ceil(p.x), p.y)
    # #   else:
    # #       map = lambda p: Point(math.floor(p.x), p.y)
    # #   return p 