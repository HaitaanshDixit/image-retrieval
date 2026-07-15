import os
import json
import argparse
import numpy as np
import faiss

from utils.config import load_config, get_logger

logger = get_logger("build_faiss_index")


def build_index(cfg: dict, index_type: str = None):
    index_type = index_type or cfg["faiss"]["index_type"]
    emb_dir = cfg["paths"]["embeddings_dir"]

    embeddings = np.load(os.path.join(emb_dir, "clip_embeddings.npy")).astype("float32")
    with open(os.path.join(emb_dir, "image_ids.json")) as f:
        image_ids = json.load(f)

    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)  # embeddings are already unit-norm from CLIP, but be safe

    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    elif index_type == "IndexIVFFlat":
        nlist = cfg["faiss"]["nlist"]
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")

    os.makedirs(cfg["paths"]["faiss_dir"], exist_ok=True)
    faiss.write_index(index, os.path.join(cfg["paths"]["faiss_dir"], "index.faiss"))
    with open(os.path.join(cfg["paths"]["faiss_dir"], "id_map.json"), "w") as f:
        json.dump(image_ids, f, indent=2)

    logger.info(f"Built {index_type} FAISS index with {index.ntotal} vectors (dim={dim}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-type", type=str, default=None,
                         help="Override configs/config.yaml faiss.index_type")
    args = parser.parse_args()

    config = load_config()
    build_index(config, index_type=args.index_type)
