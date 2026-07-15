import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.query_parser import parse_query


def test_attribute_specific():
    q = parse_query("A person in a bright yellow raincoat.")
    assert "yellow" in q.colors
    assert "raincoat" in q.categories
    assert ("yellow", "raincoat") in q.color_category_pairs


def test_contextual_place():
    q = parse_query("Professional business attire inside a modern office.")
    assert q.scene_bucket == "office"
    assert q.style == "formal"


def test_complex_semantic():
    q = parse_query("Someone wearing a blue shirt sitting on a park bench.")
    assert "blue" in q.colors
    assert "shirt" in q.categories
    assert q.scene_bucket == "park"


def test_style_inference():
    q = parse_query("Casual weekend outfit for a city walk.")
    assert q.style == "casual"
    assert q.scene_bucket == "urban_street"


def test_compositional_pairs():
    q = parse_query("A red tie and a white shirt in a formal setting.")
    assert ("red", "tie") in q.color_category_pairs
    assert ("white", "shirt") in q.color_category_pairs
    assert q.scene_bucket == "office"  # "formal setting" maps to office bucket
    assert q.style == "formal"
