"""TrustMR experiments: full COCO validation queries with frozen corpus and ranking rules."""
import csv
import json
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from src.protocol_v6 import unit

GT_STATE = None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def expanded_indices(old_indices, validation_indices):
    old = list(map(int, old_indices))
    validation = list(map(int, validation_indices))
    if len(set(old)) != len(old) or not set(old).issubset(validation):
        raise ValueError("Archived queries do not form a unique validation subset")
    seen = set(old)
    return old + [i for i in validation if i not in seen]


def normalize_queries(raw):
    values = np.asarray(raw, dtype="<f4").copy()
    values /= np.linalg.norm(values, axis=1, keepdims=True) + 1e-12
    return np.asarray([unit(v) for v in values])


def exact_neighbors(points, query, k=20):
    scores = np.einsum("ij,j->i", points, query, dtype=np.float64, optimize=False)
    quantized = np.rint(scores * 1e8).astype(np.int64)
    ids = np.lexsort((np.arange(len(points)), -quantized))[:k]
    return ids, scores[ids], quantized[ids]


def worker_threads():
    threadpool_limits(limits=1)


def gt_one(task):
    modality, index = task
    points, queries = GT_STATE
    return modality, index, exact_neighbors(points, queries[modality][index])


def prepare_inputs(archive, output, workers):
    global GT_STATE
    frozen = archive / "results/evaluation_manifest_v1"
    destination = output / "inputs"
    complete = destination / "complete.json"
    if complete.exists():
        manifest = json.loads(complete.read_text())
        if manifest["archive"] != str(archive):
            raise ValueError("Input archive differs from the saved run")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    points = np.load(frozen / "corpus_vectors.npy", mmap_mode="r")
    metadata = read_csv(frozen / "query_ids.csv")
    corpus_ids = {int(row["coco_image_id"]) for row in read_csv(frozen / "corpus_ids.csv")}
    queries, query_rows, old_counts = {}, [], {}
    with h5py.File(archive / "data/embeddings/coco_imagebind.h5", "r") as source:
        val = np.flatnonzero(source["splits"][:].astype(str) == "val")
        image_ids = source["image_ids"][:]
        if corpus_ids & set(map(int, image_ids[val])):
            raise ValueError("Training corpus and validation queries overlap")
        positions = {int(index): position for position, index in enumerate(val)}
        for modality in ("image", "text"):
            old_rows = sorted((r for r in metadata if r["modality"] == modality),
                              key=lambda r: int(r["query_index"]))
            order = expanded_indices([r["hdf5_index"] for r in old_rows], val)
            all_validation = source[f"{modality}_embeddings"][val]
            raw = all_validation[[positions[i] for i in order]]
            normalized = normalize_queries(raw)
            historical = np.load(frozen / f"{modality}_queries.npy")
            if not np.array_equal(normalized[:len(historical)], historical):
                raise ValueError(f"{modality}: archived query vectors do not match")
            old_counts[modality] = len(historical)
            queries[modality] = normalized
            np.save(destination / f"{modality}_queries.npy", normalized)
            for index, hdf5_index in enumerate(order):
                qid = f"{modality}:val:{hdf5_index}"
                if index < len(old_rows) and qid != old_rows[index]["query_id"]:
                    raise ValueError("Archived query ID changed")
                query_rows.append(dict(modality=modality, query_index=index, query_id=qid,
                    hdf5_index=hdf5_index, coco_image_id=int(image_ids[hdf5_index]),
                    split="screening" if index < 100 else "holdout"))
    with (destination / "query_ids.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(query_rows[0]))
        writer.writeheader()
        writer.writerows(query_rows)
    truth = {}
    for modality, values in queries.items():
        truth[modality] = [np.empty((len(values), 20), dtype=d) for d in ("<i8", "<f8", "<i8")]
        for array, suffix in zip(truth[modality], ("ids", "scores", "quantized_scores")):
            array[:old_counts[modality]] = np.load(frozen / f"{modality}_gt_{suffix}.npy")
        for index in (0, 100, old_counts[modality] - 1):
            actual = exact_neighbors(points, values[index])
            for saved, checked in zip(truth[modality], actual):
                if not np.array_equal(saved[index], checked):
                    raise ValueError("Archived ground truth does not match the exact ranking rule")
    tasks = [(m, i) for m in queries for i in range(old_counts[m], len(queries[m]))]
    GT_STATE = points, queries
    print(f"Preparing exact neighbors for {len(tasks)} additional queries", flush=True)
    with mp.get_context("fork").Pool(workers, initializer=worker_threads) as pool:
        for done, (modality, index, arrays) in enumerate(pool.imap_unordered(gt_one, tasks, chunksize=4), 1):
            for destination_array, value in zip(truth[modality], arrays):
                destination_array[index] = value
            if done % 200 == 0:
                print(f"GROUND TRUTH {done}/{len(tasks)}", flush=True)
    for modality, arrays in truth.items():
        for array, suffix in zip(arrays, ("ids", "scores", "quantized_scores")):
            np.save(destination / f"{modality}_gt_{suffix}.npy", array)
    write_json(complete, dict(complete=True, archive=str(archive),
        corpus_path=str(frozen / "corpus_vectors.npy"), corpus_size=len(points),
        counts={m: len(v) for m, v in queries.items()}, original_query_prefix=old_counts,
        screening_per_modality=100, holdout_per_modality={m: len(v)-100 for m, v in queries.items()},
        corpus_query_image_overlap=0, archived_queries_bitwise_equal=True,
        score="float64 einsum, 1e-8 quantization, record-ID ascending tie break",
        frozen_configuration=dict(trustmr="B2-p8-m8-coarse1000-K20", vgann="R2", vsq="V4-exact-32x8")))
    GT_STATE = None
    return destination
