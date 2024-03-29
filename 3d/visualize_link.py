import numpy as np
from surface import visualize_edges_lst, Square
from point3 import *

path_dict = {
    1: [1, 0, 0],
    2: [-1, 0, 0],
    3: [0, 1, 0],
    4: [0, -1, 0],
    5: [0, 0, 1],
    6: [0, 0, -1]
}

def link_to_edges(order_list):
    start = np.asarray([0, 0, 0])
    
    prev = start
    path = [start]
    for order in order_list:
        next = prev + np.asarray(path_dict[order])
        path.append(next)
        prev = next
    
    edges = []
    for idx, v in enumerate(path):
        if idx != len(path) - 1:
            edges.append([v, path[idx + 1]])
    print(edges)
    
    return edges

if __name__ == "__main__":
    # x_524623316415
    order_list = [5,2,3,6,1,4]
    opposite = [6, 1, 4, 5, 2, 3]
    edges = link_to_edges(order_list)
    visualize_edges_lst(edges)
    
    edges = link_to_edges(opposite)
    visualize_edges_lst(edges)