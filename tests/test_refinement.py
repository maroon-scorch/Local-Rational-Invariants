# Setup
import sys
sys.path.insert(1, '../code')
import pytest, math

# Import
from point import Point
from main import index_segment

def test_index_segment():
    assert index_segment(Point(0, 0), Point(0, 0)) == [Point(0, 0)]
    assert index_segment(Point(0, 0), Point(2, 2)) == [Point(0, 0), Point(1, 1), Point(2, 2)]
    assert index_segment(Point(0, 0), Point(2, 0)) == [Point(0, 0), Point(1, 0), Point(2, 0)]
    assert index_segment(Point(1, 1), Point(1, -1)) == [Point(1, 1), Point(1, 0), Point(1, -1)]
    
    assert index_segment(Point(0, 0), Point(1, 4)) == [Point(0, 0), Point(0.25, 1), Point(0.5, 2), Point(0.75, 3), Point(1, 4)]