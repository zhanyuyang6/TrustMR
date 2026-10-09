"""TrustMR experiments: select fixed image and text queries from the COCO validation set."""

import sys
from pathlib import Path

import h5py
import numpy as np

SEED = 42
N_IMAGE_QUERIES = 1000
N_TEXT_QUERIES = 1000
EMB_PATH = Path("data/embeddings/coco_imagebind.h5")
OUT_DIR = Path("data/queries")
METADATA_PATH = OUT_DIR / "query_set_metadata.txt"


def main():
    if not EMB_PATH.exists():
        print(f"[ERROR] Embedding file not found: {EMB_PATH}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {EMB_PATH} ...")
    with h5py.File(EMB_PATH, "r") as f:
        image_emb = f["image_embeddings"][:]
        text_emb = f["text_embeddings"][:]
        splits = f["splits"][:].astype(str)
        image_ids = f["image_ids"][:] if "image_ids" in f else None

    val_mask = splits == "val"
    val_indices = np.where(val_mask)[0]
    n_val = len(val_indices)
    print(f"Val split size: {n_val}")

    if n_val < max(N_IMAGE_QUERIES, N_TEXT_QUERIES):
        print(f"[ERROR] Val split has only {n_val} entries, "
              f"need {max(N_IMAGE_QUERIES, N_TEXT_QUERIES)}.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(SEED)
    img_query_idx = rng.choice(n_val, size=N_IMAGE_QUERIES, replace=False)

    # Text queries use a separate seed to avoid duplicating the image-query sample.
    rng_text = np.random.default_rng(SEED + 1)
    txt_query_idx = rng_text.choice(n_val, size=N_TEXT_QUERIES, replace=False)

    img_global_idx = val_indices[img_query_idx]
    txt_global_idx = val_indices[txt_query_idx]

    image_queries = image_emb[img_global_idx].astype(np.float32)
    text_queries = text_emb[txt_global_idx].astype(np.float32)

    image_queries /= np.linalg.norm(image_queries, axis=1, keepdims=True) + 1e-12
    text_queries /= np.linalg.norm(text_queries, axis=1, keepdims=True) + 1e-12

    img_out = OUT_DIR / "image_queries.npy"
    txt_out = OUT_DIR / "text_queries.npy"
    np.save(img_out, image_queries)
    np.save(txt_out, text_queries)
    print(f"Saved {image_queries.shape} -> {img_out}")
    print(f"Saved {text_queries.shape} -> {txt_out}")

    with open(METADATA_PATH, "w") as f:
        f.write("TrustMR Fixed Query Set Metadata\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Source file:      {EMB_PATH}\n")
        f.write(f"Source split:     val\n")
        f.write(f"Val split size:   {n_val}\n\n")
        f.write(f"Seed (image):     {SEED}\n")
        f.write(f"Seed (text):      {SEED + 1}\n")
        f.write(f"N image queries:  {N_IMAGE_QUERIES}\n")
        f.write(f"N text queries:   {N_TEXT_QUERIES}\n")
        f.write(f"Embedding dim:    {image_queries.shape[1]}\n")
        f.write(f"Normalization:    L2\n\n")
        f.write(f"Image query HDF5 indices (val-local, first 20): "
                f"{img_query_idx[:20].tolist()}\n")
        f.write(f"Text  query HDF5 indices (val-local, first 20): "
                f"{txt_query_idx[:20].tolist()}\n\n")
        f.write(f"Image query global HDF5 indices (first 20): "
                f"{img_global_idx[:20].tolist()}\n")
        f.write(f"Text  query global HDF5 indices (first 20): "
                f"{txt_global_idx[:20].tolist()}\n")
        if image_ids is not None:
            f.write(f"\nImage COCO IDs for image queries (first 20): "
                    f"{image_ids[img_global_idx[:20]].tolist()}\n")
            f.write(f"Image COCO IDs for text queries  (first 20): "
                    f"{image_ids[txt_global_idx[:20]].tolist()}\n")

    print(f"Metadata saved to {METADATA_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
