# Where your dataset goes

This folder is intentionally empty in the repo (see `.gitignore`).

1. Unzip your image dataset so that individual image files
   (`.jpg`/`.jpeg`/`.png`) sit directly inside `data/raw/images/`
   (subfolders are fine too - `indexer/extract_features.py` walks the
   directory recursively).

2. If you have Fashionpedia-style annotations (COCO-format JSON), drop
   them in `data/raw/annotations/`. They are **not required** to run the
   pipeline - the current indexer parses garment attributes zero-shot at
   index time (see `models/garment_parser.py`) - but if you want to
   swap in ground-truth Fashionpedia categories/attributes instead of
   the zero-shot parser for a subset of images, this is where they'd be
   read from.

3. From the repo root, run:
   ```bash
   python -m indexer.run_indexing_pipeline
   ```

That's it - embeddings, metadata, and the FAISS index will be written to
`data/processed/`.
