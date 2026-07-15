# Fashion Image Retrieval

Natural-language search over a fashion image dataset that reasons about
**garment attributes** (category, color) and **scene/place** (office,
street, park, home) jointly and not just whole-image CLIP similarity.

This README covers setup and running the code.

## Architecture at a glance

```
                        ┌───────────────────────────┐
   Part A: INDEXER      │  raw images                │
   (indexer/)           └──────────────┬──────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                          ▼
     OpenCLIP ViT-L/14           Places365 ResNet50        OWL-ViT + CLIP zero-shot
     whole-image embedding        scene bucket              per-garment category+color
              │                         │                          │
              ▼                         └────────────┬─────────────┘
      FAISS index (data/processed/faiss)              ▼
                                          per-image metadata JSON
                                          (data/processed/metadata)

                        ┌───────────────────────────┐
   Part B: RETRIEVER    │  natural language query    │
   (retriever/)         └──────────────┬──────────────┘
                                        │
                      ┌─────────────────┼─────────────────┐
                      ▼                                    ▼
        query_parser.py (rule-based,           clip_search.py
        colors/categories/scene/               CLIP text encoder -> FAISS
        color-category pairs)                  -> top ~100 candidates
                      │                                    │
                      └──────────────┬─────────────────────┘
                                      ▼
                         reranker.py: blend CLIP similarity
                         with structured attribute/scene match
                                      ▼
                                 top-k results



## Repo layout

```
image-retrieval/
├── configs/config.yaml       # all paths, model names, weights - single source of truth
├── indexer/                  # Part A: feature extraction + FAISS index build
├── retriever/                # Part B: query parsing, recall, reranking
├── models/                   # shared model wrappers (CLIP, Places365, garment parser)
├── utils/                    # vocab, color utils, image/io/config helpers
├── tests/                    # pytest unit tests for query parser + reranker logic
├── notebooks/demo.ipynb      # end-to-end demo
└── data/                     # raw/ (you provide) + processed/ (generated)
```

Indexer and retriever share **no logic**, only the vocabulary in
`utils/fashion_vocab.py` and the model wrappers in `models/` - each can
be read, tested, and modified independently, which is also why they're
laid out seperately.
