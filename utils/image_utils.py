import os
from typing import List
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_images(directory: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def safe_load_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load image at {path}: {e}") from e


def image_id_from_path(path: str) -> str:
    """Stable id for an image = filename without extension."""
    return os.path.splitext(os.path.basename(path))[0]
