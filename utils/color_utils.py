from typing import Tuple
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from utils.fashion_vocab import COLOR_VOCAB, canonical_color

# Reference RGB anchors for each basic color name. Approximate on purpose -
# we only need "nearest named color", not colorimetric precision.
_COLOR_ANCHORS = {
    "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
    "brown": (101, 67, 33), "beige": (222, 202, 172), "tan": (210, 180, 140),
    "navy": (0, 0, 128), "blue": (0, 102, 204), "red": (204, 0, 0),
    "maroon": (128, 0, 0), "pink": (255, 153, 204), "orange": (255, 140, 0),
    "yellow": (255, 221, 51), "green": (34, 139, 34), "olive": (107, 142, 35),
    "purple": (128, 0, 128), "lavender": (200, 162, 200), "teal": (0, 128, 128),
    "gold": (212, 175, 55), "silver": (192, 192, 192), "cream": (255, 253, 208),
}


def dominant_color_rgb(crop: Image.Image, k: int = 3) -> Tuple[int, int, int]:
    """Return the most dominant RGB color in a garment crop via k-means."""
    small = crop.convert("RGB").resize((50, 50))
    pixels = np.asarray(small).reshape(-1, 3)

    km = KMeans(n_clusters=min(k, len(pixels)), n_init=4, random_state=0)
    labels = km.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant = km.cluster_centers_[np.argmax(counts)]
    return tuple(int(c) for c in dominant)


def nearest_named_color(rgb: Tuple[int, int, int]) -> str:
    """Snap an RGB triple to the nearest basic color name (Euclidean in RGB)."""
    best_name, best_dist = "gray", float("inf")
    for name, anchor in _COLOR_ANCHORS.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, anchor))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return canonical_color(best_name)
