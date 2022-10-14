# Setup
import sys
import pytest, math

sys.path.insert(1, '../code')

# Import
from point import *
from edge import *

def test_intersection():
    assert intersect(Edge(Point(0, 0), Point(0, 2)), Edge(Point(1, 1), Point(-1, 1)))
    assert intersect(Edge(Point(0, 0), Point(1/2, 1/2)), Edge(Point(0, 1), Point(1, 0)))
    assert not intersect(Edge(Point(0, 0), Point(0, 2)), Edge(Point(1, 1), Point(2, 1)))
    
    assert intersect(Edge(Point(0, 0), Point(0, 1)), Edge(Point(0, 1/2), Point(0, 1)))