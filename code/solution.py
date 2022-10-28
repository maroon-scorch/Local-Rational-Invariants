import sys, itertools
from copy import copy, deepcopy

from matplotlib.pyplot import grid
from data.point import *
# from projection import *
from functools import reduce
import math
import sympy as sp
from util import *

def solve(points):
    """ Given a polygonal curve, constructs its grid approximation """
    grid_list = []
    label_list = list(map(lambda p: label_index(p), points))
    
    
    for idx, point in enumerate(points):
        # Find the closest grid point to the first point
        if idx == 0:
            initial_point = closest_grid_point(point)
            grid_list.append(initial_point)
        else:
            # Same Label, move straight?
            if label_list[idx - 1] == label_list[idx]:
                previous_point = grid_list[-1]
                if label_list[idx] == 0:
                    diff = point.y - previous_point.y
                    diff = diff/abs(diff) 
                    new_point = Point(previous_point.x, previous_point.y + diff)
                    grid_list.append(new_point)
                else:
                    diff = point.x - previous_point.x
                    diff = diff/abs(diff) 
                    new_point = Point(previous_point.x + diff, previous_point.y)
                    grid_list.append(new_point)
            # Different label, move corner?
            else:
                previous_point = grid_list[-1]
                diff_x = point.x - previous_point.x
                diff_x = diff_x/abs(diff_x)
    
                diff_y = point.y - previous_point.y
                diff_y = diff_y/abs(diff_y)
                
                new_point = Point(previous_point.x + diff_x, previous_point.y)
                new_point_2 = Point(new_point.x, new_point.y + diff_y)
                grid_list.append(new_point)
                grid_list.append(new_point_2)
                    
    
    # # Step 2: For all the points on the curve that intersect with the grids, 
    # # mark the point to be 0 if it meets
    # # the horizontal sides of the grid and 1 if it meets with the vertical sides.
    # grid_list = []
    
    # label_dict = {}
    # for pt in points:
    #     label_dict[pt] = label_index(pt)
    
    # adj_list = []
    # for idx, point in enumerate(points):
    #     if idx != len(points) - 1:
    #         adj_list.append([point, points[idx + 1]])
    # adj_list.append([points[-1], points[0]])
    
    print(grid_list)
    return grid_list

def solve_alt(points):
    """ Given a polygonal curve, constructs its grid approximation """
    grid_list = []
    label_list = list(map(lambda p: label_index(p), points))
    
    
    for idx, point in enumerate(points):
        # Find the closest grid point to the first point
        if idx == 0:
            grid_list.append(point)
        else:
            previous_point = grid_list[-1]
            next_point_1 = Point(previous_point.x, point.y)
            next_point_2 = Point(point.x, next_point_1.y)
            grid_list.append(next_point_1)
            grid_list.append(next_point_2)
    
    print(grid_list)
    return grid_list

def solve_project(points):
    """ Given a polygonal curve, constructs its grid approximation """
    print("TESTTTT")
    grid_list = []
    grid_edge_list = []
    edge_list = vert_to_edges(points)
    label_list = list(map(lambda p: [label_index(p[0]), label_index(p[1])], edge_list))
    
    for idx, edge in enumerate(edge_list):
        current_label = label_list[idx]
        if current_label[0] == current_label[1]:
            if current_label[0] == 0:
                # Horizontal line:
                edge_1 = [Point(edge[0].x, math.ceil(edge[0].y)), Point(edge[1].x, math.ceil(edge[1].y))]
                edge_2 = [Point(edge[0].x, math.floor(edge[0].y)), Point(edge[1].x, math.floor(edge[1].y))]
                
                if not is_left(edge[0], edge[1], edge_1[0]):
                    grid_edge_list.append(edge_1)
                else:
                    grid_edge_list.append(edge_2)
            elif current_label[0] == 1:
                # Vertical line:
                edge_1 = [Point(math.ceil(edge[0].x), edge[0].y), Point(math.ceil(edge[1].x), edge[1].y)]
                edge_2 = [Point(math.floor(edge[0].x), edge[0].y), Point(math.floor(edge[1].x), edge[1].y)]
                
                if not is_left(edge[0], edge[1], edge_1[0]):
                    grid_edge_list.append(edge_1)
                else:
                    grid_edge_list.append(edge_2)
        else:
            start = edge[0]
            end = edge[1]
            midpoint = Point((start.x + end.x)/2, (start.y + end.y)/2)
            p1, p2, p3, p4 = grid_points(midpoint)
            
            right_points = list(filter(lambda p: not is_left(start, end, p), [p1, p2, p3, p4]))
            if len(right_points) == 1:
                edge[0] = edge[1]
                # only_right = right_points[0]
                # grid_edge_list.append([start, only_right])
                # grid_edge_list.append([only_right, end])
            elif len(right_points) == 3:
                grid_start = find_grid(start, right_points)
                grid_end = find_grid(end, right_points)
                right_points.remove(grid_start)
                right_points.remove(grid_end)
                
                corner = right_points[0]
                grid_edge_list.append([grid_start, corner])
                grid_edge_list.append([corner, grid_end])
                
    # Remove hedges
    for i, ed in enumerate(grid_edge_list):
        print(str(i) + ": " + str(ed))
    faulty_edge_list = []
    
    for idx, ed_1 in enumerate(grid_edge_list):
        if idx != len(grid_edge_list) - 1:
            second_idx = idx + 1
        else:
            second_idx = 0
        ed_2 = grid_edge_list[second_idx]
        inverse_ed_2 = [ed_2[1], ed_2[0]]
        if ed_1 == inverse_ed_2:
            faulty_edge_list.append(idx)
            faulty_edge_list.append(second_idx)

    print(faulty_edge_list)
    faulty_edge_list.sort(reverse=True)
    
    for i in faulty_edge_list:
        grid_edge_list.pop(i)

    visualize_edges(grid_edge_list)         
    grid_list = edges_to_vert(grid_edge_list)
    # if grid_list[0] != grid_list[-1]:
    #     grid_list.append(grid_list[0])
    
         
    # grid_list = list(filter(lambda p: is_grid_point(p), grid_list))
    # print(grid_list)

    # input = map(lambda ed: [[ed[0].x, ed[0].y], [ed[1].x, ed[1].y]], grid_edge_list)
    # # Plot of the Polygonal Curve
    # fig = plt.figure()
    # for i in range(0, len(input)):
    #     plt.plot(input[i][0], input[i][1])
        
    # # Integer Grid
    # ax = fig.gca()
    # xmin, xmax = ax.get_xlim() 
    # ymin, ymax = ax.get_ylim() 
    # ax.set_xticks(int_between(xmin, xmax))
    # ax.set_yticks(int_between(ymin, ymax))
    # plt.grid()
    # # Title
    # plt.title("Edge of Grid")
    # plt.show()
    
    
    return grid_list

def solve_half_length(points):
    grid_edge_list = []
    edge_list = vert_to_edges_open(points)

    for edge in edge_list:
        current_p1 = edge[0]
        current_p2 = edge[1]

        mid_x_1 = (math.floor(current_p1.x) + math.ceil(current_p1.x))/2
        mid_y_1 = (math.floor(current_p1.y) + math.ceil(current_p1.y))/2
        mid_x_2 = (math.floor(current_p2.x) + math.ceil(current_p2.x))/2
        mid_y_2 = (math.floor(current_p2.y) + math.ceil(current_p2.y))/2

        midpoint = Point((current_p1.x + current_p2.x)/2, (current_p1.y + current_p2.y)/2)
        c1, _, _, c1_op = grid_points(midpoint)
        center = Point((c1.x + c1_op.x)/2, (c1.y + c1_op.y)/2)

        grid_edge_list.append([Point(mid_x_1, mid_y_1), center])
        grid_edge_list.append([center, Point(mid_x_2, mid_y_2)])

    faulty_edge_list = []
    for idx, ed_1 in enumerate(grid_edge_list):
        if idx != len(grid_edge_list) - 1:
            second_idx = idx + 1
        else:
            second_idx = 0
        ed_2 = grid_edge_list[second_idx]
        inverse_ed_2 = [ed_2[1], ed_2[0]]
        if ed_1 == inverse_ed_2 or ed_1 == ed_2:
            faulty_edge_list.append(idx)
            faulty_edge_list.append(second_idx)

    # print(faulty_edge_list)
    faulty_edge_list.sort(reverse=True)
    
    for i in faulty_edge_list:
        grid_edge_list.pop(i)

    # for i, ed in enumerate(grid_edge_list):
    #     print(str(i) + ": " + str(ed))

    visualize_edges(grid_edge_list)
    grid_list = edges_to_vert(grid_edge_list)
    
    return grid_list