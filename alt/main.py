#!/bin/python3
import sys
from copy import copy, deepcopy
from npoint import NPoint

# Run program using 'python main.py [directory to file]'

def read_input(inputFile):
    """ Read and parse the input file, returning the list of points and its dimension """
    poly_points = []
    with open(inputFile, "r") as f:
        dimension = int(f.readline())
        for line in f.readlines():
            tokens = line.strip().split()
            
            bool_point = map(lambda b: bool(int(b)), tokens[1::2])
            new_point = map(lambda b, pt: int(pt) if b else float(pt), bool_point, tokens[0::2])
            new_point = NPoint(list(new_point))
            
            poly_points.append(new_point)
    
    return poly_points, dimension
    
# The main body of the code:
if __name__ == "__main__":
    input_file = sys.argv[1]
    poly_points, dimension = read_input(input_file)

    print("This is a curve in dimension: ", dimension)
    print("The list has length: ", len(poly_points))
    print("The list has points: ", poly_points)