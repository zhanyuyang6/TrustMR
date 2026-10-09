"""TrustMR experiments: prepare fixed corpus vectors, queries, and exact neighbors."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.protocol_v6 import unit


def exact_topk(points, queries, k=20):
    ids, values, quantized = [], [], []
    rids = np.arange(len(points))
    for query in queries:
        scores = np.einsum("ij,j->i", points, query, dtype=np.float64, optimize=False)
        ranks = np.rint(scores * 1e8).astype(np.int64)
        selected = np.lexsort((rids, -ranks))[:k]
        ids.append(selected)
        values.append(scores[selected])
        quantized.append(ranks[selected])
    return np.asarray(ids, dtype="<i8"), np.asarray(values), np.asarray(quantized)


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_embeddings(h5, rows, modality, split):
    indices = np.asarray([int(row["hdf5_index"]) for row in rows])
    image_ids = h5["image_ids"][:]
    splits = h5["splits"][:].astype(str)
    if not np.all(splits[indices] == split):
        raise ValueError("HDF5 split differs from the published ID map")
    if any(int(image_ids[i]) != int(row["coco_image_id"]) for i, row in zip(indices, rows)):
        raise ValueError("HDF5 image IDs differ from the published ID map")
    return np.asarray([h5[f"{modality}_embeddings"][int(i)] for i in indices], dtype="<f4")


def prepare(h5_path):
    metadata = ROOT / "evaluation_metadata"
    out = ROOT / "results/evaluation_manifest_v1"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh results directory; refusing to replace {out}")
    corpus_rows = read_rows(metadata / "corpus_ids.csv")
    query_rows = read_rows(metadata / "query_ids.csv")
    if [int(row["rid"]) for row in corpus_rows] != list(range(len(corpus_rows))):
        raise ValueError("Corpus RID order is not contiguous")
    with h5py.File(h5_path, "r") as h5:
        points = np.asarray([unit(v) for v in read_embeddings(h5, corpus_rows, "image", "train")])
        queries = {}
        for modality in ("image", "text"):
            rows = [row for row in query_rows if row["modality"] == modality]
            if [int(row["query_index"]) for row in rows] != list(range(len(rows))):
                raise ValueError("Query indices are not contiguous")
            raw = read_embeddings(h5, rows, modality, "val")
            # Match query extraction normalization before applying unit().
            raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12
            queries[modality] = np.asarray([unit(v) for v in raw])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("corpus_ids.csv", "query_ids.csv"):
        shutil.copyfile(metadata / name, out / name)
    np.save(out / "corpus_vectors.npy", points)
    for modality, vectors in queries.items():
        np.save(out / f"{modality}_queries.npy", vectors)
        for suffix, values in zip(("ids", "scores", "quantized_scores"), exact_topk(points, vectors)):
            np.save(out / f"{modality}_gt_{suffix}.npy", values)
        print(f"Prepared {modality}: {len(vectors)} queries", flush=True)
    manifest = json.loads((metadata / "gt_manifest.json").read_text())
    artifacts = {}
    for name, expected in manifest["artifacts"].items():
        actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
        artifacts[name] = {"sha256": actual, "matches_archived": actual == expected["sha256"]}
    (out / "input_verification.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    matched = all(row["matches_archived"] for row in artifacts.values())
    print("Archived inputs: " + ("exact match" if matched else "different; this is a new embedding run"))

    # Native HCNNG consumes fvecs files; benchmark verification uses the frozen npy.
    native = ROOT / "results/vgann_screen_v2"
    native.mkdir(parents=True, exist_ok=True)
    for n in (1000, 5000, 20000, 118287):
        if len(points) < n:
            continue
        records = np.empty((n, points.shape[1] + 1), dtype="<i4")
        records[:, 0] = points.shape[1]
        records[:, 1:] = points[:n].view("<i4")
        records.tofile(native / f"corpus_{n}.fvecs")
    for name in ("vgann_final_screen_v1", "vsq_original_v1", "vsq_vector_v1"):
        (ROOT / "results" / name).mkdir(exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=ROOT / "data/embeddings/coco_imagebind.h5")
    prepare(parser.parse_args().h5)
