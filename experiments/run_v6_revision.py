"""TrustMR experiments: full-vector reference measurements and audit simulation."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.protocol_v6 import (
    setup, Record, Query, unit, signatures, commit, digest, retrieve,
    verify, filtered, cosine_top, select_coarse, probes, challenge,
    answer_audit, wire_size, miss_probability,
)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def load(n):
    with h5py.File(ROOT / "data/embeddings/coco_imagebind.h5", "r") as f:
        mask = np.asarray(f["splits"]).astype(str) == "train"
        indices = np.flatnonzero(mask)[:n]
        vectors = np.asarray(f["image_embeddings"])[indices].astype(np.float32)
        paths = np.asarray(f["image_paths"])[indices].astype(str)
    images = np.load(ROOT / "data/queries/image_queries.npy")
    texts = np.load(ROOT / "data/queries/text_queries.npy")
    return vectors, paths, {"image": images, "text": texts}


def build(vectors, paths, out):
    env = setup(registry=tuple((f"S{i:02}", f"L{i:02}") for i in range(16)))
    vectors = np.stack([unit(row) for row in vectors])
    start = perf_counter()
    sigs = signatures(vectors, env)
    bids = sigs[:, :env.prefix].astype(np.int64) @ (1 << np.arange(env.prefix))
    records = []
    for rid, (vec, path, bid) in enumerate(zip(vectors, paths, bids)):
        raw = (ROOT / path).read_bytes()
        records.append(Record(rid, digest("content", raw), vec, "image", rid,
                              f"S{rid % 16:02}", 1, int(bid)))
    raw_hash_seconds = perf_counter() - start
    start = perf_counter()
    store = commit(env, records)
    build_seconds = perf_counter() - start
    save(out / "trusted_ledger_snapshot.json", dict(epoch=1,
         environment_hash=store.ledger.environment_hash.hex(),
         anchors=[dict(bucket=i, count=n, root=r.hex())
                  for i, (n, r) in enumerate(store.ledger.anchors)]))
    save(out / "build.json", dict(n=len(records), embedding_dimension=env.dimension,
         content_and_initial_projection_seconds=raw_hash_seconds,
         commitment_build_seconds=build_seconds,
         metadata="SYNTHETIC: timestamp=record index, source=record index modulo16",
         content="SHA-256 domain-separated digest of actual COCO JPEG bytes",
         ledger="independently trusted local snapshot; no network consensus latency"))
    print(f"COMMIT n={len(records)} elapsed={build_seconds:.2f}s", flush=True)
    return env, records, store


def query(vector, modal, n, flips=4, coarse=1000, restricted=False):
    return Query(unit(vector), modal, 1, n//4 if restricted else 0,
                 3*n//4 if restricted else n-1,
                 tuple(f"S{i:02}" for i in range(8 if restricted else 16)),
                 20, coarse, flips, ("image",))


def metric(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1)) if len(a)>1 else 0.0,
                p50=float(np.quantile(a, .5)), p95=float(np.quantile(a, .95)), n=len(a))


def run_retrieval(env, records, store, queries, args, out):
    summary = []
    for modal, vectors in queries.items():
        for m in (4, 8):
            for restricted in (False, True):
                label = f"{modal}_m{m}_range{int(restricted)}"
                rows = []
                for i, vector in enumerate(vectors[:args.queries]):
                    q = query(vector, modal, len(records), m, restricted=restricted)
                    all_records = filtered(records, q)
                    start = perf_counter()
                    truth = cosine_top(all_records, q)
                    exact_ms = (perf_counter()-start)*1000
                    start = perf_counter()
                    vo = retrieve(q, store)
                    query_ms = (perf_counter()-start)*1000
                    recalled = len(set(vo.result) & set(truth))/max(1, len(truth))
                    bucket_records = filtered((r for b in vo.buckets for r in b.records), q)
                    # Same probe coverage, with no Hamming truncation.
                    start = perf_counter()
                    full_bucket_result = cosine_top(bucket_records, q)
                    exact_bucket_ms = (perf_counter()-start)*1000
                    row = dict(i=i, recall=recalled, query_ms=query_ms, exact_ms=exact_ms,
                               probes=len(vo.probe_ids), members=sum(len(b.records) for b in vo.buckets),
                               filtered=len(bucket_records), coarse=len(vo.coarse_ids),
                               same_coverage_no_trim_recall=len(set(full_bucket_result)&set(truth))/max(1,len(truth)),
                               bucket_cosine_ms=exact_bucket_ms)
                    if m == 8 and full_bucket_result != truth:
                        raise AssertionError("all-bucket exact result differs from ground truth")
                    if i < args.verify_queries:
                        start = perf_counter()
                        verdict = verify(q, vo, env, store.ledger)
                        row["verify_ms"] = (perf_counter()-start)*1000
                        row["valid"] = verdict.valid
                        row["wire_bytes"] = wire_size(vo)
                        row["steps"] = verdict.steps
                        if not verdict.valid:
                            raise AssertionError(verdict.steps)
                    rows.append(row)
                    if i % 25 == 0:
                        print(f"RETRIEVAL {label} {i+1}/{args.queries} recall={recalled:.3f}", flush=True)
                    with (out / f"{label}.jsonl").open("a") as f:
                        f.write(json.dumps(row)+"\n")
                record = dict(label=label, modality=modal, flips=m, restricted=restricted,
                              database_size=len(records), queries=len(rows),
                              metrics={k:metric([r[k] for r in rows if k in r]) for k in
                                       ("recall","query_ms","exact_ms","probes","members","filtered","coarse",
                                        "same_coverage_no_trim_recall","bucket_cosine_ms","verify_ms","wire_bytes")
                                       if any(k in r for r in rows)})
                summary.append(record)
                save(out / "retrieval_summary.json", summary)
    return summary


def tamper(vo, attack, q):
    if attack == "honest":
        return vo
    if attack == "query_modality":
        return replace(vo, query_hash=digest("query", b"wrong modality"))
    if attack == "epoch":
        return replace(vo, environment_hash=b"x"*32)
    if attack == "routing":
        return replace(vo, probe_ids=vo.probe_ids[:-1])
    if attack == "bucket_missing":
        return replace(vo, buckets=vo.buckets[:-1])
    if attack in ("ranking", "lower_topk", "coarse_selection", "filter_bypass"):
        rows = [r for b in vo.buckets for r in b.records]
        if attack == "ranking":
            return replace(vo, result=tuple(reversed(vo.result)))
        if attack == "lower_topk":
            selected = [r for r in rows if r.rid in vo.coarse_ids and r.rid != vo.result[0]]
            return replace(vo, result=cosine_top(selected, q))
        if attack == "filter_bypass":
            return replace(vo, coarse_ids=tuple(r.rid for r in rows[:q.coarse]),
                           result=cosine_top(rows[:q.coarse], q))
        ids = vo.coarse_ids[1:]
        return replace(vo, coarse_ids=ids, result=cosine_top([r for r in rows if r.rid in ids], q))
    bs = list(vo.buckets)
    idx = next(i for i,b in enumerate(bs) if b.records)
    b = bs[idx]
    if attack == "member_missing":
        bs[idx] = replace(b, records=b.records[1:], leaves=b.leaves[1:])
    else:
        r = b.records[0]
        if attack == "vector":
            r = replace(r, vector=unit(-r.vector))
        elif attack == "timestamp":
            r = replace(r, timestamp=r.timestamp+999999)
        elif attack == "source":
            r = replace(r, source="S15" if r.source != "S15" else "S00")
        elif attack == "record_modality":
            r = replace(r, modality="text")
        elif attack == "content":
            r = replace(r, content_hash=b"z"*32)
        elif attack == "joint_tree_replacement":
            r = replace(r, timestamp=r.timestamp+999999)
        else:
            raise ValueError(attack)
        leaves = (r.leaf(),)+b.leaves[1:] if attack == "joint_tree_replacement" else b.leaves
        bs[idx] = replace(b, records=(r,)+b.records[1:], leaves=leaves)
    return replace(vo, buckets=tuple(bs))


def run_security(env, records, store, queries, args, out):
    attacks = ("honest","query_modality","epoch","routing","bucket_missing","member_missing",
               "vector","timestamp","source","record_modality","content","ranking","lower_topk",
               "coarse_selection","filter_bypass","joint_tree_replacement")
    variants = {"full":set(), **{f"skip_check_{i}":{i} for i in range(6)}}
    counts = {v:{a:[] for a in attacks} for v in variants}
    for i, vector in enumerate(queries["image"][:args.attack_queries]):
        q = query(vector,"image",len(records),restricted=True)
        vo = retrieve(q,store)
        for attack in attacks:
            bad = tamper(vo,attack,q)
            for variant,skip in variants.items():
                verdict = verify(q,bad,env,store.ledger,skip_steps=skip)
                counts[variant][attack].append(dict(valid=verdict.valid,
                    executed=[s["step"] for s in verdict.steps if not s["skipped"]],
                    failed=[s["step"] for s in verdict.steps if not s["passed"]]))
        print(f"SECURITY {i+1}/{args.attack_queries}",flush=True)
        save(out/"security_raw.json",counts)
    table = []
    for variant, results in counts.items():
        for attack, values in results.items():
            table.append(dict(variant=variant,attack=attack,n=len(values),
                acceptance_rate=sum(v["valid"] for v in values)/len(values),
                detection_rate=None if attack=="honest" else sum(not v["valid"] for v in values)/len(values)))
    save(out/"security_summary.json",table)


def run_audit_sim(out):
    rows=[]
    for h in (1,4,16):
        for alpha,t in ((.05,50),(.1,20),(.2,10),(.3,10)):
            n=240
            s=round(n*alpha)
            rng=np.random.default_rng(20260911)
            # Exactly equivalent to sampling s distinct bucket IDs each round.
            hits=rng.hypergeometric(h,n-h,s,size=(10000,t))
            observed=float(np.mean(np.any(hits>0,axis=1)))
            rows.append(dict(n=n,h=h,s=s,t=t,alpha=s/n,trials=10000,
                             observed=observed,theory=1-miss_probability(n,h,s,t),
                             generator="numpy hypergeometric, independent rounds"))
    save(out/"audit_probability.json",rows)


def run_audit_protocol(env,records,store,queries,args,out):
    rows=[]
    for i,vector in enumerate(queries["text"][:args.audit_queries]):
        q=query(vector,"text",len(records),restricted=True)
        vo=retrieve(q,store)
        start=perf_counter()
        ch=challenge(vo,store.ledger,24,20)
        lock_challenge_ms=(perf_counter()-start)*1000
        start=perf_counter()
        response=answer_audit(ch,store)
        response_ms=(perf_counter()-start)*1000
        start=perf_counter()
        verdict=verify(q,vo,env,store.ledger,ch,response)
        verify_ms=(perf_counter()-start)*1000
        if not verdict.valid:
            raise AssertionError(verdict.steps)
        missing=verify(q,vo,env,store.ledger,ch,response[:-1])
        badvo=replace(vo,result=tuple(reversed(vo.result)))
        transcript=verify(q,badvo,env,store.ledger,ch,response,skip_steps={5})
        audit_vo=replace(vo,buckets=response,probe_ids=tuple(b.bid for b in response),coarse_ids=(),result=())
        rows.append(dict(i=i,lock_challenge_ms=lock_challenge_ms,response_ms=response_ms,
                         verify_with_audit_ms=verify_ms,buckets=len(response),valid=verdict.valid,
                         material_omission=verdict.material_omission,
                         missing_response_rejected=not missing.valid,
                         transcript_replacement_rejected=not transcript.valid,
                         audit_disclosure_bytes=wire_size(audit_vo),
                         seeds=[s.hex() for s in ch.seeds],rounds=ch.rounds,
                         initial=vo.result,augmented=verdict.augmented))
        print(f"AUDIT {i+1}/{args.audit_queries}",flush=True)
        save(out/"audit_protocol.json",rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--n",type=int,default=118287)
    parser.add_argument("--queries",type=int,default=1000)
    parser.add_argument("--verify-queries",type=int,default=100)
    parser.add_argument("--attack-queries",type=int,default=100)
    parser.add_argument("--audit-queries",type=int,default=10)
    parser.add_argument("--mode",choices=("all","retrieval","security","audit"),default="all")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    save(out/"run_metadata.json",dict(args=vars(args),time=datetime.now(timezone.utc).isoformat(),
         python=sys.version,platform=platform.platform(),numpy=np.__version__,
         thread_env={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
         dataset="COCO train JPEGs + existing frozen ImageBind embeddings; fixed val queries",
         ground_truth="float64 cosine quantized 1e-8, rid ascending; same filtered corpus",
         source_time_metadata="synthetic controlled variables, not real collection metadata"))
    vectors,paths,queries=load(args.n)
    env,records,store=build(vectors,paths,out)
    if args.mode in ("all","retrieval"):
        run_retrieval(env,records,store,queries,args,out)
    if args.mode in ("all","security"):
        run_security(env,records,store,queries,args,out)
    if args.mode in ("all","audit"):
        run_audit_sim(out)
        run_audit_protocol(env,records,store,queries,args,out)
    save(out/"COMPLETE.json",dict(completed=datetime.now(timezone.utc).isoformat()))
    print("COMPLETE",out,flush=True)


if __name__=="__main__":
    main()
