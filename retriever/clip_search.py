import os
import json
from typing import List, Tuple
import numpy as np
import faiss

from models.clip_model import ClipEncoder


class ClipSearcher:
    def __init__(self, cfg: dict, clip_encoder: ClipEncoder):
        self.clip = clip_encoder
        faiss_dir = cfg["paths"]["faiss_dir"]
        self.index = faiss.read_index(os.path.join(faiss_dir, "index.faiss"))
        with open(os.path.join(faiss_dir, "id_map.json")) as f:
            self.id_map = json.load(f)  # row i -> image_id

    def search(self, query_text: str, pool_size: int) -> List[Tuple[str, float]]:
        query_vec = self.clip.encode_text(query_text).cpu().numpy().astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, pool_size)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.id_map[idx], float(score)))
        return results
