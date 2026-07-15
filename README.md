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
```

## Setup (conda + VS Code)

```bash
conda create -n fashion-retrieval python=3.10 -y
conda activate fashion-retrieval
pip install -r requirements.txt
```

every model used (OpenCLIP, Places365, OWL-ViT) is free and auto-downloads its pretrained weights on
first run (Places365 weights are cached to `models/weights/`, everything
else is cached by `torch`/`huggingface_hub` in their default cache dirs).
If you're on a machine with no GPU, everything falls back to CPU
automatically (slower, but works, see `configs/config.yaml`, set
`models.clip.device: "cpu"` explicitly if autodetection ever misbehaves).

## 1. Add your dataset

Unzip your images into `data/raw/images/` (see `data/raw/README.md`).

## 2. Build the index (Part A)

```bash
# quick smoke test on the first 25 images
python -m indexer.run_indexing_pipeline --limit 25

# full run
python -m indexer.run_indexing_pipeline
```

This writes:
- `data/processed/embeddings/clip_embeddings.npy` + `image_ids.json`
- `data/processed/metadata/<image_id>.json` (scene + garments per image)
- `data/processed/faiss/index.faiss` + `id_map.json`

## 3. Query (Part B)

```bash
python -m retriever.search --query "A person in a bright yellow raincoat." --top_k 5
python -m retriever.search --query "A red tie and a white shirt in a formal setting." --top_k 5 --json
```

Or open `notebooks/demo.ipynb` to run all 5 evaluation queries from the
assignment brief at once and preview the top image.

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