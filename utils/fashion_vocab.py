GARMENT_CATEGORIES = [
    "shirt", "blouse", "t-shirt", "sweater", "cardigan", "hoodie",
    "jacket", "blazer", "coat", "raincoat", "vest",
    "pants", "jeans", "shorts", "skirt", "dress", "jumpsuit",
    "suit", "tie", "scarf", "hat", "glasses", "belt",
    "shoe", "bag", "glove",
]

STYLE_CATEGORY_MAP = {
    "formal": {"shirt", "blouse", "blazer", "suit", "tie", "dress", "coat"},
    "casual": {"t-shirt", "hoodie", "jeans", "shorts", "sweater", "cardigan"},
    "outerwear": {"jacket", "coat", "raincoat", "vest", "blazer"},
}

COLOR_VOCAB = [
    "black", "white", "gray", "grey", "brown", "beige", "tan", "navy",
    "blue", "light blue", "sky blue", "red", "maroon", "pink", "hot pink",
    "orange", "yellow", "bright yellow", "green", "olive", "khaki",
    "purple", "lavender", "teal", "turquoise", "gold", "silver", "cream",
]

COLOR_CANONICAL = {
    "grey": "gray", "sky blue": "blue", "light blue": "blue",
    "hot pink": "pink", "bright yellow": "yellow", "khaki": "tan",
    "turquoise": "teal",
}


def canonical_color(color: str) -> str:
    color = color.lower().strip()
    return COLOR_CANONICAL.get(color, color)


SCENE_BUCKETS = ["office", "urban_street", "park", "home", "other"]
SCENE_KEYWORDS = {
    "office": ["office", "workplace", "corporate", "cubicle", "boardroom",
               "conference room", "business setting", "formal setting"],
    "urban_street": ["street", "city", "sidewalk", "downtown", "urban",
                      "city walk", "crosswalk", "avenue"],
    "park": ["park", "bench", "garden", "outdoors", "picnic", "trail"],
    "home": ["home", "living room", "bedroom", "house", "apartment"],
}
