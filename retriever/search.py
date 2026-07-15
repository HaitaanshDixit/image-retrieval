import argparse
import json

from utils.config import load_config, get_logger
from models.clip_model import ClipEncoder
from retriever.query_parser import parse_query
from retriever.clip_search import ClipSearcher
from retriever.reranker import rerank

logger = get_logger("search")


def search(query_text: str, cfg: dict, clip: ClipEncoder, top_k: int = None):
    top_k = top_k or cfg["retrieval"]["top_k_default"]
    pool_size = cfg["retrieval"]["recall_pool_size"]

    parsed = parse_query(query_text)
    logger.info(f"Parsed query -> colors={parsed.colors} categories={parsed.categories} "
                f"pairs={parsed.color_category_pairs} scene={parsed.scene_bucket} style={parsed.style}")

    searcher = ClipSearcher(cfg, clip)
    candidates = searcher.search(query_text, pool_size=pool_size)
    logger.info(f"CLIP recall stage returned {len(candidates)} candidates.")

    results = rerank(
        parsed, candidates,
        metadata_dir=cfg["paths"]["metadata_dir"],
        weights=cfg["retrieval"]["weights"],
        top_k=top_k,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Natural language search query")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")
    args = parser.parse_args()

    config = load_config()
    clip_encoder = ClipEncoder(
        backbone=config["models"]["clip"]["backbone"],
        pretrained=config["models"]["clip"]["pretrained"],
        device=config["models"]["clip"]["device"],
    )

    ranked = search(args.query, config, clip_encoder, top_k=args.top_k)

    if args.json:
        print(json.dumps([r.__dict__ for r in ranked], indent=2))
    else:
        print(f"\nTop {len(ranked)} results for: \"{args.query}\"\n")
        print(f"{'rank':<5}{'image_id':<25}{'final':<8}{'clip':<8}{'color':<8}{'cat':<8}{'scene':<8}")
        for i, r in enumerate(ranked, 1):
            print(f"{i:<5}{r.image_id:<25}{r.final_score:<8}{r.clip_similarity:<8}"
                  f"{r.color_match:<8}{r.category_match:<8}{r.scene_match:<8}")
