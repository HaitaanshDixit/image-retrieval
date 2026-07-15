import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.color_utils import nearest_named_color


def test_nearest_named_color_primary_colors():
    assert nearest_named_color((250, 0, 0)) == "red"
    assert nearest_named_color((0, 0, 250)) == "blue"
    assert nearest_named_color((250, 250, 250)) == "white"
    assert nearest_named_color((5, 5, 5)) == "black"
