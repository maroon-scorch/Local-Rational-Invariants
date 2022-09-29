#!/bin/python3
import sys
from copy import copy, deepcopy
import random
from point import Point
import matplotlib.pyplot as plt

# Run program using 'python main.py [directory to file]'

def read_input(inputFile):
    """ Read and parse the input file, returning the list of points and its length """
    poly_points = []
    with open(inputFile, "r") as f:
        length = int(f.readline())
        for line in f.readlines():
            tokens = line.strip().split()
            x_bool = bool(int(tokens[2]))
            y_bool = bool(int(tokens[3]))
            x_pos =  int(tokens[0]) if x_bool else float(tokens[0])
            y_pos =  int(tokens[1]) if y_bool else float(tokens[1])
            new_point = Point(x_pos, y_pos)
            poly_points.append(new_point)
    
    return poly_points, length

def visualize(points):
    """ Given a list of points, draws the curve traced out by it """
    input = map(lambda pt: [pt.x, pt.y], points)
    x_pts, y_pts = zip(*input) #create lists of x and y values
    
    plt.figure()
    plt.plot(x_pts, y_pts)
    plt.show()

def printOutput(result):
    result = ""
    print(result)
    
# The main body of the code:
if __name__ == "__main__":
    input_file = sys.argv[1]
    poly_points, length = read_input(input_file)

    print("The list has length: ", length)
    print("The list has points: ", poly_points)
    visualize(poly_points)