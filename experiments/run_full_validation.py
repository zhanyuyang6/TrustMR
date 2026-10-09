"""TrustMR experiments: fixed-configuration evaluation on all COCO validation queries."""
import argparse
import csv
import gc
import json
import multiprocessing as mp
import os
import pickle
import platform
import struct
import sys
import time
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import compact_wire, protocol_compact as compact, protocol_v6 as ref
from baselines.vgann.reproduction_v3 import core_r2 as vgann
from baselines.vsq.vector import core as vsq
from experiments.full_validation_inputs import prepare_inputs, read_csv, worker_threads, write_json

STATE = None
METHODS = ("V-GANN", "VSQ", "TrustMR")


def load_vgann(archive, points):
    folder = archive / "results/vgann_final_screen_v1"
    index = vgann.Index.__new__(vgann.Index)
    index.points = points
    index.n, index.d = points.shape
    index.size = 1 << (index.n - 1).bit_length()
    data = (folder / f"tuples_{index.n}.bin").read_bytes()
    index.tuplebytes, position = [], 0
    while position < len(data):
        length = struct.unpack_from("<I", data, position)[0]
        position += 4
        index.tuplebytes.append(data[position:position + length])
        position += length
    if position != len(data) or len(index.tuplebytes) != index.n:
        raise ValueError("Invalid saved V-GANN tuples")
    index.tuples = [vgann.unpack_tuple(value) for value in index.tuplebytes]
    data = (folder / f"mht_{index.n}.bin").read_bytes()
    index.levels, position, width = [], 0, index.size
    while width:
        index.levels.append([data[j:j+32] for j in range(position, position+width*32, 32)])
        position += width * 32
        width //= 2
    anchor = json.loads((folder / f"anchor_{index.n}.json").read_text())
    index.root = bytes.fromhex(anchor["root"])
    index.signature = bytes.fromhex(anchor["signature"])
    index.public = Ed25519PublicKey.from_public_bytes(bytes.fromhex(anchor["public_key"]))
    if position != len(data) or index.levels[-1][0] != index.root:
        raise ValueError("Invalid saved V-GANN Merkle root")
    index.public.verify(index.signature, index.root)
    return index


def load_index(method, archive, points):
    if method == "V-GANN":
        return load_vgann(archive, points)
    if method == "VSQ":
        folder = archive / "results/vsq_vector_v1"
        index = vsq.Index.__new__(vsq.Index)
        index.points = points
        index.n, index.d = points.shape
        index.bands, index.width = 32, 8
        index.signatures = np.load(folder / "shared/signatures.npy", mmap_mode="r")
        index.hashes = np.load(folder / "shared/vector_hashes.npy", mmap_mode="r")
        index.projection = np.load(folder / "shared/projection.npy")
        previous = sys.modules.get("core")
        sys.modules["core"] = vsq
        try:
            with (folder / "V4/tree_cache.pkl").open("rb") as stream:
                index.tree = pickle.load(stream)
        finally:
            if previous is None:
                sys.modules.pop("core", None)
            else:
                sys.modules["core"] = previous
        anchor = json.loads((folder / "V4/anchor.json").read_text())
        if index.tree.root.hex() != anchor["root"]:
            raise ValueError("VSQ cached root differs from its saved anchor")
        return index
    with (archive / "results/vgann_final_screen_v1/trustmr_store.pkl").open("rb") as stream:
        store = pickle.load(stream)
    if store.env.prefix != 8 or store.env.projection.shape != (256, points.shape[1]):
        raise ValueError("TrustMR cache has a different configuration")
    if len(store.vectors) != len(points):
        raise ValueError("TrustMR cache has a different corpus")
    for i in range(len(points)):
        if not np.array_equal(store.vectors[i], points[i]):
            raise ValueError(f"TrustMR vector mismatch at RID {i}")
    return store


def trustmr_response(q, store):
    started = time.perf_counter()
    if not ref.valid_query(q, store.env):
        raise ValueError("Invalid query")
    probes = ref.probes(q, store.env)
    buckets = tuple(store.buckets[i] for i in probes)
    candidates = ref.filtered((r for b in buckets for r in b.records), q)
    coarse = compact.select_coarse(candidates, q, store.env)
    openings = tuple((i, store.vectors[i]) for i in coarse)
    result = compact.top(openings, q)
    search = time.perf_counter() - started
    started = time.perf_counter()
    response = compact.MainResponse(q.commitment(), store.env.commitment(), probes,
                                    buckets, coarse, result, openings)
    proof = time.perf_counter() - started
    started = time.perf_counter()
    wire = compact_wire.encode(response)
    serialization = time.perf_counter() - started
    return response, wire, dict(search_seconds=search, proof_generation_seconds=proof,
        serialization_seconds=serialization, candidate_count=len(candidates))


def one_query(task):
    modality, query_index, repeat, timing_mode = task
    method, index, queries, truth, metadata, corpus_size = STATE
    vector = queries[modality][query_index]
    meta = metadata[modality, query_index]
    base = dict(method=method, modality=modality, query_index=query_index,
        query_id=meta["query_id"], split=meta["split"], repeat=repeat,
        timing_mode=timing_mode, valid=False, result_count=0)
    try:
        if method == "TrustMR":
            q = ref.Query(vector, modality, 1, 0, corpus_size-1,
                tuple(s for s, _ in index.env.registry), 20, 1000, 8, ("image",))
            response, wire, stats = trustmr_response(q, index)
            result = response.result
        elif method == "V-GANN":
            response, wire, native, _ = index.response(vector, 20, meta["query_id"])
            result = response["result"]
            stats = dict(search_seconds=native["search_seconds"],
                proof_generation_seconds=native["raw_vo_seconds"]+native["collapse_seconds"],
                serialization_seconds=native["serialize_seconds"], candidate_count=native["probed_points"])
        else:
            response, wire, native, _ = index.response(vector, k=20, radius=0)
            result = response["result"]
            stats = {key: native[key] for key in ("search_seconds", "proof_generation_seconds", "serialization_seconds")}
            stats["candidate_count"] = native["unique_candidates"]
        base.update(stats, result_ids=list(map(int, result)), result_count=len(result),
            recall=len(set(result) & set(map(int, truth[modality][query_index]))) / 20,
            result_plus_proof_bytes=len(wire), search_e2e_directly_comparable=method != "V-GANN")
        started = time.perf_counter()
        if method == "TrustMR":
            received = compact_wire.decode(wire)
            verdict = compact.verify(q, received, index.env, index.ledger)
            if not verdict.valid:
                raise ValueError("TrustMR verifier rejected the response")
        elif method == "V-GANN":
            vgann.verify(wire, vector, 20, meta["query_id"], index.n, index.d, index.public, index.root)
        else:
            vsq.verify(wire, vector, vsq.anchor(index), index.projection, k=20, radius=0)
        verification = time.perf_counter() - started
        base.update(valid=True, verification_seconds=verification,
            end_to_end_seconds=sum(stats[k] for k in ("search_seconds", "proof_generation_seconds", "serialization_seconds"))+verification)
    except Exception as error:
        base["error"] = f"{type(error).__name__}: {error}"
    return base


def rows_from_log(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def preflight(archive, output):
    global STATE
    frozen = archive / "results/evaluation_manifest_v1"
    points = np.load(frozen / "corpus_vectors.npy", mmap_mode="r")
    queries = {m: np.load(frozen / f"{m}_queries.npy", mmap_mode="r") for m in ("image", "text")}
    truth = {m: np.load(frozen / f"{m}_gt_ids.npy", mmap_mode="r") for m in queries}
    metadata = {(r["modality"], int(r["query_index"])): dict(r, split="screening")
                for r in read_csv(frozen / "query_ids.csv")}
    sources = {"TrustMR": "vgann_final_screen_v1/trustmr_raw.csv",
               "V-GANN": "vgann_final_screen_v1/vgann_raw.csv",
               "VSQ": "vsq_vector_v1/V4_screening_raw.csv"}
    checks = []
    for method in METHODS:
        index = load_index(method, archive, points)
        STATE = method, index, queries, truth, metadata, len(points)
        historical = {(r["modality"], int(r["query_index"])): r
                      for r in read_csv(archive / "results" / sources[method])}
        for modality in queries:
            row = one_query((modality, 0, 0, "preflight"))
            old = historical[modality, 0]
            if not row["valid"] or row["recall"] != float(old["recall"]) or row["result_plus_proof_bytes"] != int(old["result_plus_proof_bytes"]):
                raise ValueError(f"Preflight differs from archived {method}/{modality}: {row}")
            checks.append(dict(method=method, modality=modality, valid=True,
                recall=row["recall"], result_plus_proof_bytes=row["result_plus_proof_bytes"]))
        print(f"PREFLIGHT {method} PASS", flush=True)
        STATE = None
        del index
        gc.collect()
    write_json(output / "preflight.json", dict(passed=True, checks=checks))


def evaluate(method, archive, inputs, output, workers, latency_queries, latency_repeats, blas_threads):
    global STATE
    points = np.load(archive / "results/evaluation_manifest_v1/corpus_vectors.npy", mmap_mode="r")
    queries = {m: np.load(inputs / f"{m}_queries.npy", mmap_mode="r") for m in ("image", "text")}
    truth = {m: np.load(inputs / f"{m}_gt_ids.npy", mmap_mode="r") for m in queries}
    metadata = {(r["modality"], int(r["query_index"])): r for r in read_csv(inputs / "query_ids.csv")}
    print(f"LOAD {method}", flush=True)
    index = load_index(method, archive, points)
    STATE = method, index, queries, truth, metadata, len(points)
    directory = output / method.lower().replace("-", "")
    directory.mkdir(exist_ok=True)
    for m in queries:
        checked = one_query((m, 0, 0, "warmup"))
        if not checked["valid"]:
            raise ValueError(f"{method} warmup failed: {checked}")
        if method == "TrustMR":
            q = ref.Query(queries[m][0], m, 1, 0, len(points)-1,
                tuple(s for s, _ in index.env.registry), 20, 1000, 8, ("image",))
            if trustmr_response(q, index)[1] != compact_wire.encode(compact.retrieve(q, index)):
                raise ValueError("Instrumented TrustMR response differs from native retrieval")
    log = directory / "queries.jsonl"
    existing = rows_from_log(log)
    done = {(r["modality"], r["query_index"]) for r in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate query rows in saved results")
    tasks = [(m, i, 0, f"parallel_{workers}_workers") for m in queries
             for i in range(len(queries[m])) if (m, i) not in done]
    with log.open("a", buffering=1) as stream:
        with mp.get_context("fork").Pool(workers, initializer=worker_threads) as pool:
            for count, row in enumerate(pool.imap_unordered(one_query, tasks, chunksize=1), len(done)+1):
                stream.write(json.dumps(row) + "\n")
                if count % 100 == 0:
                    print(f"EVAL {method} {count}/{sum(map(len, queries.values()))}", flush=True)
                    write_json(output / "status.json", dict(phase="evaluation", method=method, completed=count))
    gc.collect()
    timing_log = directory / "timing.jsonl"
    timing_rows = rows_from_log(timing_log)
    done_timing = {(r["modality"], r["query_index"], r["repeat"]) for r in timing_rows}
    if len(done_timing) != len(timing_rows):
        raise ValueError("Duplicate timing rows")
    threadpool_limits(limits=blas_threads)
    write_json(directory / "timing_environment.json", dict(threadpools=threadpool_info(),
        worker_count=1, queries_per_modality=latency_queries, repeats=latency_repeats))
    with timing_log.open("a", buffering=1) as stream:
        for repeat in range(latency_repeats):
            for modality in queries:
                one_query((modality, 100, repeat, "warmup"))
                for i in range(100, min(100+latency_queries, len(queries[modality]))):
                    if (modality, i, repeat) in done_timing:
                        continue
                    stream.write(json.dumps(one_query((modality, i, repeat, "single_worker"))) + "\n")
                    if (i-99) % 25 == 0:
                        print(f"TIMING {method} {modality} repeat={repeat+1} {i-99}/{latency_queries}", flush=True)
                        write_json(output / "status.json", dict(phase="timing", method=method,
                            modality=modality, repeat=repeat+1, completed=i-99))
    STATE = None
    del index
    gc.collect()


def summarize(output, expected_queries, expected_timing):
    summary = []
    all_complete = True
    for method in METHODS:
        directory = output / method.lower().replace("-", "")
        rows = rows_from_log(directory / "queries.jsonl")
        timed = rows_from_log(directory / "timing.jsonl")
        for modality in ("image", "text"):
            modal = [r for r in rows if r["modality"] == modality]
            timing = [r for r in timed if r["modality"] == modality]
            complete = len(modal) == expected_queries and len(timing) == expected_timing
            all_complete &= complete
            for split in ("all_validation", "holdout", "screening"):
                chosen = modal if split == "all_validation" else [r for r in modal if r["split"] == split]
                valid = [r for r in chosen if r["valid"]]
                row = dict(method=method, modality=modality, split=split, queries=len(chosen),
                    valid_queries=len(valid), invalid_queries=len(chosen)-len(valid), complete=complete)
                if valid:
                    row.update(recall_mean_valid=float(np.mean([r["recall"] for r in valid])),
                        response_mib_mean_valid=float(np.mean([r["result_plus_proof_bytes"] for r in valid])) / 2**20)
                if split == "holdout":
                    good_timing = [r for r in timing if r["valid"]]
                    row.update(timing_observations=len(timing), valid_timing_observations=len(good_timing),
                        timing_unique_queries=len({r["query_id"] for r in timing}))
                    if good_timing:
                        row.update(verification_ms_mean=float(np.mean([r["verification_seconds"] for r in good_timing]))*1000,
                            verification_ms_p95=float(np.percentile([r["verification_seconds"] for r in good_timing], 95))*1000)
                summary.append(row)
    fields = list(dict.fromkeys(key for row in summary for key in row))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    write_json(output / "complete.json", dict(complete=bool(all_complete),
        all_responses_valid=all(r["invalid_queries"] == 0 for r in summary)
            and all(r.get("timing_observations", 0) == r.get("valid_timing_observations", 0) for r in summary),
        methods=list(METHODS), queries_per_modality=expected_queries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--latency-queries", type=int, default=200)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=64)
    args = parser.parse_args()
    args.archive, args.output = args.archive.resolve(), args.output.resolve()
    if args.output.is_relative_to(args.archive):
        parser.error("Use an output directory outside the archived experiment project")
    if min(args.workers, args.latency_queries, args.latency_repeats, args.blas_threads) < 1:
        parser.error("Worker, query, repeat and thread counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    run = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = args.output / "run_config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != run:
        raise ValueError("Existing run configuration differs")
    write_json(manifest, run)
    write_json(args.output / "environment.json", dict(python=sys.version, numpy=np.__version__,
        platform=platform.platform(), cpu_count=os.cpu_count()))
    try:
        write_json(args.output / "status.json", dict(phase="preflight"))
        preflight(args.archive, args.output)
        write_json(args.output / "status.json", dict(phase="preparing_inputs"))
        inputs = prepare_inputs(args.archive, args.output, args.workers)
        counts = json.loads((inputs / "complete.json").read_text())["counts"]
        if counts != {"image": 5000, "text": 5000}:
            raise ValueError(f"Expected complete COCO validation set, found {counts}")
        for method in METHODS:
            evaluate(method, args.archive, inputs, args.output, args.workers,
                     args.latency_queries, args.latency_repeats, args.blas_threads)
        summarize(args.output, 5000, min(args.latency_queries, 4900)*args.latency_repeats)
        write_json(args.output / "status.json", dict(phase="complete"))
        print("FULL VALIDATION COMPLETE", flush=True)
    except BaseException as error:
        write_json(args.output / "status.json", dict(phase="failed", error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    main()
