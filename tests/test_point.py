# Setup
import sys
import pytest, math

sys.path.insert(1, '../code')

# Import
from point import Point, dist, angle

def test_dist():
    assert dist(Point(1, 3), Point(1, 3)) == 0
    assert dist(Point(0, 0), Point(0, 1)) == 1
    assert dist(Point(0, 0), Point(1, 1)) == pytest.approx(1.41421356237)
    
def test_angle():
    assert angle(Point(0, 0), Point(1, 0), Point(1, 1)) == math.pi/2
    assert angle(Point(0, 0), Point(1, 0), Point(2, 0)) == math.pi
    assert angle(Point(0, 0), Point(1, 0), Point(0, 0)) == 0
    
    assert angle(Point(0, 0), Point(1, 0), Point(2, -1)) == math.pi*3/4
    assert angle(Point(0, 0), Point(1, 0), Point(2, 1)) == math.pi*3/4
    
    assert angle(Point(0, 0), Point(1, 0), Point(0, 1)) == pytest.approx(math.pi/4)
    
    assert angle(Point(0, 0), Point(2, 1), Point(3, 1)) == pytest.approx(2.67794504459)