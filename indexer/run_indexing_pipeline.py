"""
PART A - INDEXER: single entry point.

Runs feature extraction (CLIP + Places365 + garment parsing) followed by
FAISS index construction. This is the only script a user needs to run to
go from a folder of raw images to a searchable index.

Usage:
    python -m indexer.run_indexing_pipeline
    python -m indexer.run_indexing_pipeline --limit 50   # quick smoke test
"""
import argparse
from utils.config import load_config, get_logger
from indexer.extract_features import run as extract_features
from indexer.build_faiss_index import build_index

logger = get_logger("run_indexing_pipeline")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of images (quick test run).")
    args = parser.parse_args()

    config = load_config()

    logger.info("=== Step 1/2: feature extraction ===")
    extract_features(config, limit=args.limit)

    logger.info("=== Step 2/2: FAISS index build ===")
    build_index(config)

    logger.info("Indexing complete. You can now run: python -m retriever.search --query \"...\"")
