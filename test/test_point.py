# Setup
import sys
sys.path.insert(1, '../code')
import pytest, math

# Import
from point import Point, dist, angle

def test_dist():
    assert dist(Point(1, 3), Point(1, 3)) == 0
    assert dist(Point(0, 0), Point(0, 1)) == 1
    assert dist(Point(0, 0), Point(1, 1)) == pytest.approx(1.41421356237)
    
def test_angle():
    assert angle(Point(0, 0), Point(1, 0), Point(1, 1)) == math.pi/2
    assert angle(Point(0, 0), Point(1, 0), Point(2, 0)) == math.pi