import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from utils.config import load_config, get_logger
from utils.image_utils import list_images, safe_load_image, image_id_from_path
from models.clip_model import ClipEncoder
from models.scene_model import SceneClassifier
from models.garment_parser import GarmentParser

logger = get_logger("extract_features")


def _load_existing_embeddings(cfg: dict):
    emb_path = os.path.join(cfg["paths"]["embeddings_dir"], "clip_embeddings.npy")
    ids_path = os.path.join(cfg["paths"]["embeddings_dir"], "image_ids.json")
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        embeddings = np.load(emb_path)
        with open(ids_path) as f:
            ids = json.load(f)
        return list(embeddings), ids
    return [], []


def _save_checkpoint(cfg: dict, all_embeddings, all_ids):
    embeddings_matrix = np.stack(all_embeddings).astype("float32")
    np.save(os.path.join(cfg["paths"]["embeddings_dir"], "clip_embeddings.npy"), embeddings_matrix)
    with open(os.path.join(cfg["paths"]["embeddings_dir"], "image_ids.json"), "w") as f:
        json.dump(all_ids, f, indent=2)


def run(cfg: dict, limit: int = None, checkpoint_every: int = 25):
    image_paths = list_images(cfg["paths"]["raw_images_dir"])
    if not image_paths:
        raise RuntimeError(
            f"No images found in {cfg['paths']['raw_images_dir']}. "
            "Unzip your dataset there first (see data/raw/README.md)."
        )
    if limit:
        image_paths = image_paths[:limit]
    logger.info(f"Found {len(image_paths)} images to index.")

    os.makedirs(cfg["paths"]["embeddings_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["metadata_dir"], exist_ok=True)

    # skip images already indexed in a previous run
    all_embeddings, all_ids = _load_existing_embeddings(cfg)
    already_done = set(all_ids)
    if already_done:
        logger.info(f"Found {len(already_done)} already-indexed images from a "
                     f"previous run - these will be skipped.")
    remaining = [p for p in image_paths
                 if image_id_from_path(p) not in already_done]
    if not remaining:
        logger.info("Nothing left to index - all images already done. "
                     "Delete data/processed/embeddings/ to force a full rerun.")
        return
    logger.info(f"{len(remaining)} images left to process.")

    device = cfg["models"]["clip"]["device"]
    clip = ClipEncoder(
        backbone=cfg["models"]["clip"]["backbone"],
        pretrained=cfg["models"]["clip"]["pretrained"],
        device=device,
    )
    scene_model = SceneClassifier(cfg, device=device)
    garment_parser = GarmentParser(cfg, clip_encoder=clip, device=device)

    for i, path in enumerate(tqdm(remaining, desc="Indexing"), start=1):
        image_id = image_id_from_path(path)
        try:
            image = safe_load_image(path)
        except RuntimeError as e:
            logger.warning(str(e))
            continue

        embedding = clip.encode_image(image).cpu().numpy()[0]

        scene_bucket, scene_conf, scene_raw = scene_model.classify_bucket(image)

        garments = garment_parser.parse(image)

        metadata = {
            "image_id": image_id,
            "image_path": path,
            "scene": {"bucket": scene_bucket, "confidence": round(scene_conf, 4)},
            "garments": garments,
        }
        with open(os.path.join(cfg["paths"]["metadata_dir"], f"{image_id}.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        all_embeddings.append(embedding)
        all_ids.append(image_id)

        # --- Checkpoint: save progress periodically so a crash/interrupt
        if i % checkpoint_every == 0:
            _save_checkpoint(cfg, all_embeddings, all_ids)
            logger.info(f"Checkpoint saved at {len(all_ids)} total images indexed.")

    _save_checkpoint(cfg, all_embeddings, all_ids)
    logger.info(f"Saved {len(all_ids)} total embeddings and metadata records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of images (useful for a quick test run).")
    args = parser.parse_args()

    config = load_config()
    run(config, limit=args.limit)