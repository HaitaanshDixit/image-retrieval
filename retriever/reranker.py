import json
import os
from dataclasses import dataclass
from typing import List, Dict

from retriever.query_parser import ParsedQuery


@dataclass
class RankedResult:
    image_id: str
    image_path: str
    final_score: float
    clip_similarity: float
    color_match: float
    category_match: float
    scene_match: float


def _load_metadata(metadata_dir: str, image_id: str) -> Dict:
    path = os.path.join(metadata_dir, f"{image_id}.json")
    if not os.path.exists(path):
        return {"garments": [], "scene": {"bucket": "other", "confidence": 0.0}}
    with open(path) as f:
        return json.load(f)


def _color_category_score(query: ParsedQuery, garments: List[Dict]) -> float:
    """Score in [0, 1]: how well do the query's (color, category) pairs
    (or, if none were paired, its loose color/category mentions) match
    the garments actually detected in this image."""
    if not garments:
        return 0.0

    if query.color_category_pairs:
        pair_scores = []
        for color, category in query.color_category_pairs:
            best = 0.0
            for g in garments:
                cat_match = 1.0 if g["category"] == category else 0.0
                color_match = 1.0 if g["color"] == color else (
                    0.4 if g.get("color_alt") == color else 0.0
                )
                combined = 0.5 * cat_match + 0.5 * color_match
                best = max(best, combined)
            pair_scores.append(best)
        return sum(pair_scores) / len(pair_scores)

    scores = []
    if query.colors:
        colors_present = {g["color"] for g in garments} | {g.get("color_alt") for g in garments}
        hits = sum(1 for c in query.colors if c in colors_present)
        scores.append(hits / len(query.colors))
    if query.categories:
        categories_present = {g["category"] for g in garments}
        hits = sum(1 for c in query.categories if c in categories_present)
        scores.append(hits / len(query.categories))
    return sum(scores) / len(scores) if scores else 0.0


def _scene_score(query: ParsedQuery, scene_meta: Dict) -> float:
    if not query.scene_bucket:
        return 0.0
    if scene_meta.get("bucket") == query.scene_bucket:
        return float(scene_meta.get("confidence", 0.5)) or 0.5
    return 0.0


def rerank(query: ParsedQuery, candidates: List[tuple], metadata_dir: str,
           weights: Dict, top_k: int) -> List[RankedResult]:
    """`candidates` is [(image_id, clip_similarity), ...] from clip_search.py."""
    results = []
    for image_id, clip_sim in candidates:
        meta = _load_metadata(metadata_dir, image_id)
        garments = meta.get("garments", [])

        combined_attr_score = _color_category_score(query, garments)
        color_component = combined_attr_score
        category_component = combined_attr_score
        scene_component = _scene_score(query, meta.get("scene", {}))

        final = (
            weights["clip_similarity"] * clip_sim
            + weights["color_match"] * color_component
            + weights["category_match"] * category_component
            + weights["scene_match"] * scene_component
        )

        results.append(RankedResult(
            image_id=image_id,
            image_path=meta.get("image_path", ""),
            final_score=round(final, 4),
            clip_similarity=round(clip_sim, 4),
            color_match=round(color_component, 4),
            category_match=round(category_component, 4),
            scene_match=round(scene_component, 4),
        ))

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[:top_k]
