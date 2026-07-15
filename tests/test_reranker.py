import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.query_parser import parse_query
from retriever.reranker import rerank

WEIGHTS = {
    "clip_similarity": 0.55,
    "color_match": 0.15,
    "category_match": 0.20,
    "scene_match": 0.10,
}


def _write_metadata(metadata_dir, image_id, garments, scene_bucket="office", scene_conf=0.8):
    record = {
        "image_id": image_id,
        "image_path": f"/fake/{image_id}.jpg",
        "scene": {"bucket": scene_bucket, "confidence": scene_conf},
        "garments": garments,
    }
    with open(os.path.join(metadata_dir, f"{image_id}.json"), "w") as f:
        json.dump(record, f)


def test_compositional_reranking_prefers_correct_color_assignment():
    """
    The classic CLIP failure mode from the assignment brief:
    'red tie and white shirt' vs an image where the colors are swapped.
    Both images get the SAME clip_similarity here on purpose, to prove the
    reranker - not CLIP - is what separates them.
    """
    with tempfile.TemporaryDirectory() as metadata_dir:
        # Correct image: red tie + white shirt
        _write_metadata(metadata_dir, "correct", garments=[
            {"category": "tie", "color": "red", "color_alt": "red"},
            {"category": "shirt", "color": "white", "color_alt": "white"},
        ])
        # Swapped image: white tie + red shirt (same categories/colors present,
        # just attached to the wrong garment - this is what a single
        # whole-image CLIP embedding cannot distinguish)
        _write_metadata(metadata_dir, "swapped", garments=[
            {"category": "tie", "color": "white", "color_alt": "white"},
            {"category": "shirt", "color": "red", "color_alt": "red"},
        ])

        query = parse_query("A red tie and a white shirt in a formal setting.")
        candidates = [("correct", 0.5), ("swapped", 0.5)]  # identical CLIP score

        results = rerank(query, candidates, metadata_dir, WEIGHTS, top_k=2)

        assert results[0].image_id == "correct"
        assert results[0].final_score > results[1].final_score


def test_scene_match_boosts_score():
    with tempfile.TemporaryDirectory() as metadata_dir:
        _write_metadata(metadata_dir, "in_office", garments=[], scene_bucket="office", scene_conf=0.9)
        _write_metadata(metadata_dir, "in_park", garments=[], scene_bucket="park", scene_conf=0.9)

        query = parse_query("Professional business attire inside a modern office.")
        candidates = [("in_office", 0.4), ("in_park", 0.4)]

        results = rerank(query, candidates, metadata_dir, WEIGHTS, top_k=2)
        assert results[0].image_id == "in_office"
