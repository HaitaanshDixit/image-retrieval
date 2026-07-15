"""
Places365 scene classifier (ResNet-50 backbone), used to answer the
"where" half of every query (office / urban street / park / home).

We use the official pretrained Places365 checkpoint (free, no training
required) and then collapse its 365 fine-grained categories down to the
handful of coarse buckets this assignment actually needs, via the mapping
in configs/config.yaml (`scene_buckets`). This keeps scene reasoning
zero-shot with respect to OUR bucket definitions - if the assignment's
"future work" ask (adding cities/weather) comes up, you only need to add
new buckets to that mapping, not retrain anything.
"""
import os
from typing import Tuple, Dict
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
import urllib.request
from PIL import Image

from utils.config import get_logger

logger = get_logger("scene_model")

_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SceneClassifier:
    def __init__(self, cfg: dict, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg["models"]["places365"]
        self.bucket_map = cfg["scene_buckets"]

        self._ensure_weights_downloaded()
        self.categories = self._load_categories()
        self.model = self._load_model()

    # ---------------------------------------------------------------
    def _ensure_weights_downloaded(self):
        weights_path = self.cfg["local_weights_path"]
        cats_path = self.cfg["local_categories_path"]
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)

        if not os.path.exists(weights_path):
            logger.info("Downloading Places365 ResNet50 weights (one-time, ~100MB)...")
            urllib.request.urlretrieve(self.cfg["weights_url"], weights_path)
        if not os.path.exists(cats_path):
            logger.info("Downloading Places365 category list...")
            urllib.request.urlretrieve(self.cfg["categories_url"], cats_path)

    def _load_categories(self):
        cats_path = self.cfg["local_categories_path"]
        categories = []
        with open(cats_path) as f:
            for line in f:
                # format: "/o/office/cubicle 137"
                name = line.strip().split(" ")[0]
                name = name.split("/")[-1] if "/" in name else name
                categories.append(name)
        return categories

    def _load_model(self):
        model = tv_models.resnet50(num_classes=365)
        checkpoint = torch.load(self.cfg["local_weights_path"],
                                 map_location=self.device)
        state_dict = {k.replace("module.", ""): v
                      for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    # ---------------------------------------------------------------
    def _raw_topk(self, image: Image.Image, k: int = 5):
        x = _PREPROCESS(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1).squeeze(0)
        top_probs, top_idx = probs.topk(k)
        return [(self.categories[i], float(p)) for p, i in zip(top_probs, top_idx)]

    def classify_bucket(self, image: Image.Image, k: int = 5) -> Tuple[str, float, Dict]:
        raw = self._raw_topk(image, k=k)
        bucket_scores = {b: 0.0 for b in self.bucket_map}
        for raw_cat, prob in raw:
            for bucket, members in self.bucket_map.items():
                if any(m in raw_cat for m in members):
                    bucket_scores[bucket] += prob

        best_bucket = max(bucket_scores, key=bucket_scores.get)
        best_score = bucket_scores[best_bucket]
        if best_score < 0.05:
            return "other", best_score, dict(raw)
        return best_bucket, best_score, dict(raw)
