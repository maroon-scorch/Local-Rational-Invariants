import numpy as np
from surface import Trig

# Input: a list of pairs of numbers
# Output: a list of vertices forming an oriented cycle
def make_cyclic(edges):
    cyclic = [edges[0][0], edges[0][1]]
    edges = edges[1:]
    while edges != []:
        x, y = edges[0]
        if x in cyclic:
            cyclic.append(y)
        elif y in cyclic:
            cyclic.append(y)
        else: edges = edges[1:].append(edges[0])
    return cyclic

def ori_edges(cycle):
    oriented_edges = []
    for idx in range(-1, len(cycle)):
        oriented_edges.append([cycle[idx], cycle[idx+1]])
    return oriented_edges
        
# Input: a list of lists of three points ##### Tranform Trig into a list of three points
def link_cycle(trig_list):
    edges = []
    point = trig_list[0][0]
    for trig in trig_list:
        if point in trig:
            edges.append([ver != point for ver in trig])
            new_trig_list = trig_list.remove(trig)
    vertices = make_cyclic(edges)
    oriented_link = []
    oriented_link = list(map(lambda edge : oriented_link.append([point, edge[0], edge[1]])), ori_edges(vertices))
    return oriented_link, new_trig_list

def is_edge_contain(tri, oriented_tri_list):
    for ori_tri in oriented_tri_list:
        intersect_edge = list(set(ori_tri) & set(tri))
        if len(intersect_edge) == 2:
            for idx, point in enumerate(ori_tri):
                if point not in intersect_edge:
                    return idx, ori_tri
        else: return []

def oriented_tri(tri_list):
    oriented_tri_list, new_trig_list = link_cycle(tri_list)
    while new_trig_list != []:
        for tri in new_trig_list and is_edge_contain(tri, oriented_tri_list) != []:
            idx, ori_tri = is_edge_contain(tri, oriented_tri_list)
            for p in tri and p not in ori_tri:
                ori_tri[idx] = p
            new_ori_tri = ori_tri.reverse
            oriented_tri_list = oriented_tri_list.append(new_ori_tri)
            new_trig_list = new_trig_list.remove(tri)
    oriented_triangles = list(map(lambda trig : Trig(trig[0], trig[1], trig[2]), oriented_tri_list))
    return oriented_triangles
            