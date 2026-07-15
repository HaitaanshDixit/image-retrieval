import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from utils.fashion_vocab import (
    COLOR_VOCAB, GARMENT_CATEGORIES, STYLE_CATEGORY_MAP, SCENE_KEYWORDS,
    canonical_color,
)

_PAIRING_WINDOW = 4


@dataclass
class ParsedQuery:
    raw_text: str
    colors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    scene_bucket: Optional[str] = None
    style: Optional[str] = None  # "formal" | "casual" | None
    # Compositional (color, category) pairs, e.g. [("red", "tie"), ("white", "shirt")]
    color_category_pairs: List[Tuple[str, str]] = field(default_factory=list)


def parse_query(text: str) -> ParsedQuery:
    lowered = text.lower()
    colors_found = []
    for color in sorted(COLOR_VOCAB, key=len, reverse=True):
        if color in lowered:
            canon = canonical_color(color)
            if canon not in colors_found:
                colors_found.append(canon)

    # --- Garment categories mentioned by name ---
    categories_found = [c for c in GARMENT_CATEGORIES if c in lowered]

    words = re.findall(r"[a-z]+", lowered)
    color_category_pairs = []
    for cat in GARMENT_CATEGORIES:
        cat_tokens = cat.split()
        for i in range(len(words) - len(cat_tokens) + 1):
            if words[i:i + len(cat_tokens)] != cat_tokens:
                continue
            window_start = max(0, i - _PAIRING_WINDOW)
            window_words = words[window_start:i]
            window_text = " ".join(window_words)
            nearest_color = None
            for color in sorted(COLOR_VOCAB, key=len, reverse=True):
                if color in window_text:
                    nearest_color = canonical_color(color)
                    break
            if nearest_color:
                color_category_pairs.append((nearest_color, cat))

    scene_bucket = None
    best_hits = 0
    for bucket, keywords in SCENE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits > best_hits:
            best_hits, scene_bucket = hits, bucket

    style = None
    if any(w in lowered for w in ["formal", "business", "professional", "office attire"]):
        style = "formal"
    elif any(w in lowered for w in ["casual", "weekend", "relaxed", "everyday"]):
        style = "casual"
    if style is None and categories_found:
        for style_name, members in STYLE_CATEGORY_MAP.items():
            if style_name in ("formal", "casual") and any(c in members for c in categories_found):
                style = style_name
                break

    return ParsedQuery(
        raw_text=text,
        colors=colors_found,
        categories=categories_found,
        scene_bucket=scene_bucket,
        style=style,
        color_category_pairs=color_category_pairs,
    )
