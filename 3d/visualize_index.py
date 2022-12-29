import numpy as np
from point3 import *
from surface import Square, square_to_voxel
from euler import vert_link
from rigid_motion import rotate_squares_psi, integerify_square
import math

path_dict = {
    1: Point3(1, 0, 0),
    2: Point3(-1, 0, 0),
    3: Point3(0, 1, 0),
    4: Point3(0, -1, 0),
    5: Point3(0, 0, 1),
    6: Point3(0, 0, -1)
}

def index_to_squares(order_list):
    sq_list = []
    center = Point3(0, 0, 0)
    
    for i in range(len(order_list)):
        if i != len(order_list) - 1:
            current = path_dict[order_list[i]]
            next = path_dict[order_list[i + 1]]
            new_sq = Square(center, current, to_point((current.vec + next.vec).tolist()), next)
            sq_list.append(new_sq)
    
    return sq_list

if __name__ == "__main__":
    order_list = [
        [1,3,6,4,1],
    ]
    
    for i in range(len(order_list)):
        print("Vertex ", i)
        sq_list = index_to_squares(order_list[i])
        # sq_list = integerify_square(rotate_squares_psi(sq_list, 2*math.pi/2))

        print(vert_link(Point3(0, 0, 0), sq_list))
        square_to_voxel(sq_list)
    
    # order_list = [
    #     [1,4,6,2,3,5,1],
    #     [2,3,5,4,6,2],
    #     [1,4,6,3,5,1]
    # ]
    
    # order_list = [
    #     [1,3,5,2,4,6,1],
    #     [2,3,6,2],
    #     [1,4,5,1]
    # ]
    
    # order_list = [
    #     "14261",
    #     "14251",
    #     "13641",
    #     "25362",
    #     "235462",
    #     "145361",
    #     "146351",
    #     "2452",
    #     "136451",
    #     "2462",
    #     "1362541",
    #     "25462",
    #     "132541",
    #     "1463251",
    #     "132641"
    # ]
    

    
    # order_list.sort(key = lambda x: len(x))
    # print(len(order_list))
    
    # new_list = []
    # for elt in order_list:
    #     input = []
    #     for c in elt:
    #         input.append(int(c))
    #     new_list.append(input)
    
    # print(new_list)
    
    # for i in range(len(new_list)):
    #     print("Vertex ", i)
    #     sq_list = index_to_squares(new_list[i])
    #     # sq_list = integerify_square(rotate_squares_theta(sq_list, 3*math.pi/2))

    #     print(vert_link(Point3(0, 0, 0), sq_list))
    #     square_to_voxel(sq_list)