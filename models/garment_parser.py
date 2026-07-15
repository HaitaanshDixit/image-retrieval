from typing import List, Dict
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from utils.fashion_vocab import GARMENT_CATEGORIES, COLOR_VOCAB, canonical_color
from utils.color_utils import dominant_color_rgb, nearest_named_color
from models.clip_model import ClipEncoder


class GarmentParser:
    def __init__(self, cfg: dict, clip_encoder: ClipEncoder, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        det_cfg = cfg["models"]["garment_detector"]
        self.score_threshold = det_cfg["box_score_threshold"]
        self.max_boxes = det_cfg["max_boxes_per_image"]

        self.processor = OwlViTProcessor.from_pretrained(det_cfg["hf_model_id"])
        self.detector = OwlViTForObjectDetection.from_pretrained(
            det_cfg["hf_model_id"]
        ).to(self.device).eval()

        self.clip = clip_encoder  # reuse the same CLIP instance as the indexer

    # ---------------------------------------------------------------
    @torch.no_grad()
    def _detect_boxes(self, image: Image.Image) -> List[Dict]:
        prompts = [f"a {c}" for c in GARMENT_CATEGORIES]
        inputs = self.processor(text=[prompts], images=image, return_tensors="pt").to(self.device)
        outputs = self.detector(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)

        # transformers renamed this method in newer versions; support both.
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            results = self.processor.post_process_grounded_object_detection(
                outputs, threshold=self.score_threshold, target_sizes=target_sizes,
                text_labels=[prompts],
            )[0]
        else:
            results = self.processor.post_process_object_detection(
                outputs, threshold=self.score_threshold, target_sizes=target_sizes
            )[0]

        boxes = []
        # Newer API returns "text_labels" (strings like "a shirt") directly;
        # older API returns "labels" (integer indices into `prompts`).
        if "text_labels" in results:
            for score, text_label, box in zip(results["scores"], results["text_labels"], results["boxes"]):
                category = text_label[2:] if text_label.startswith("a ") else text_label
                boxes.append({
                    "category": category,
                    "category_conf": float(score),
                    "bbox": [round(v) for v in box.tolist()],
                })
        else:
            for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
                boxes.append({
                    "category": GARMENT_CATEGORIES[label_idx],
                    "category_conf": float(score),
                    "bbox": [round(v) for v in box.tolist()],
                })

        # Keep only the most confident boxes to bound cost per image.
        boxes.sort(key=lambda b: b["category_conf"], reverse=True)
        return boxes[: self.max_boxes]

    def _classify_color(self, crop: Image.Image) -> Dict:
        clip_probs = self.clip.zero_shot_classify(
            crop, COLOR_VOCAB, template="a {} colored piece of clothing"
        )
        best_i = max(range(len(clip_probs)), key=lambda i: clip_probs[i])
        clip_color = canonical_color(COLOR_VOCAB[best_i])
        clip_conf = clip_probs[best_i]

        kmeans_rgb = dominant_color_rgb(crop)
        kmeans_color = nearest_named_color(kmeans_rgb)

        agree = clip_color == kmeans_color
        return {
            "color": clip_color,
            "color_conf": min(1.0, clip_conf * (1.15 if agree else 1.0)),
            "color_alt": kmeans_color,
        }

    # ---------------------------------------------------------------
    def parse(self, image: Image.Image) -> List[Dict]:
        """Return a list of {category, category_conf, color, color_conf,
        color_alt, bbox} dicts, one per detected garment."""
        boxes = self._detect_boxes(image)
        garments = []
        for box in boxes:
            x0, y0, x1, y1 = box["bbox"]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image.width, x1), min(image.height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = image.crop((x0, y0, x1, y1))
            color_info = self._classify_color(crop)
            garments.append({**box, **color_info})
        return garments
