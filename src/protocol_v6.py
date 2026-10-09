"""TrustMR experiments: reference protocol with full-vector bucket responses."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
import secrets
import struct
from time import perf_counter
from typing import Iterable

import numpy as np


def pack(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(tag: str, *parts: bytes) -> bytes:
    h = hashlib.sha256(tag.encode("ascii") + b"\0")
    for part in parts:
        h.update(struct.pack(">Q", len(part)))
        h.update(part)
    return h.digest()


def vector_bytes(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype="<f4").tobytes(order="C")


def unit(v: np.ndarray) -> np.ndarray:
    out = np.array(v, dtype="<f4", copy=True)
    norm = np.linalg.norm(out.astype(np.float64))
    if out.ndim != 1 or not np.isfinite(norm) or norm == 0:
        raise ValueError("invalid embedding")
    out = np.asarray(out / norm, dtype="<f4")
    out.setflags(write=False)
    return out


def merkle(leaves: Iterable[bytes]) -> bytes:
    layer = list(leaves)
    if not layer:
        return digest("empty")
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [digest("node", layer[i], layer[i + 1])
                 for i in range(0, len(layer), 2)]
    return layer[0]


@dataclass(frozen=True)
class Environment:
    epoch: int
    encoder_id: str
    projection: np.ndarray
    prefix: int
    registry: tuple[tuple[str, str], ...]
    modalities: tuple[str, ...] = ("image", "text")

    @property
    def dimension(self):
        return self.projection.shape[1]

    def commitment(self) -> bytes:
        manifest = dict(epoch=self.epoch, encoder=self.encoder_id,
                        shape=self.projection.shape, prefix=self.prefix,
                        registry=self.registry, modalities=self.modalities,
                        protocol="TrustMR-V6-20260911", vector="float32-le",
                        score="float64-sum-products-quantized-1e-8-rid-ascending",
                        hamming_tie="rid-ascending", tree="duplicate-last",
                        time="integer-source-timestamp-inclusive")
        return digest("environment", pack(manifest), vector_bytes(self.projection))


def setup(d=1024, bits=256, prefix=8, epoch=1, seed=42,
          registry=(("S0", "L0"),), encoder_id="ImageBind-frozen"):
    if not 0 <= prefix <= bits or d < 1:
        raise ValueError("invalid projection dimensions")
    if len(dict(registry)) != len(registry):
        raise ValueError("duplicate source")
    projection = np.random.default_rng(seed).standard_normal((bits, d)).astype("<f4")
    projection.setflags(write=False)
    return Environment(epoch, encoder_id, projection, prefix, tuple(sorted(registry)))


def signatures(vectors: np.ndarray, env: Environment) -> np.ndarray:
    # Use identical float64 accumulation for single-vector and batch signatures.
    return np.einsum("ij,kj->ik", np.atleast_2d(vectors), env.projection,
                     dtype=np.float64, optimize=False) >= 0


def bucket_id(v: np.ndarray, env: Environment) -> int:
    bits = signatures(v, env)[0, :env.prefix]
    return sum(int(bit) << j for j, bit in enumerate(bits))


@dataclass(frozen=True)
class Record:
    rid: int
    content_hash: bytes
    vector: np.ndarray
    modality: str
    timestamp: int
    source: str
    epoch: int
    bucket: int

    def metadata(self):
        return [self.rid, self.content_hash.hex(), self.modality,
                self.timestamp, self.source, self.epoch, self.bucket]

    def leaf(self):
        return digest("record", pack(self.metadata()),
                      digest("vector", vector_bytes(self.vector)))


@dataclass(frozen=True)
class Bucket:
    bid: int
    records: tuple[Record, ...]
    leaves: tuple[bytes, ...]


def bucket_commitment(epoch, bid, leaves):
    return digest("bucket", pack([epoch, bid, len(leaves)]), merkle(leaves))


@dataclass(frozen=True)
class Ledger:
    epoch: int
    environment_hash: bytes
    anchors: tuple[tuple[int, bytes], ...]

    def commitment(self):
        return digest("ledger", pack([self.epoch, self.environment_hash.hex(),
                                      [(n, root.hex()) for n, root in self.anchors]]))


@dataclass
class Store:
    env: Environment
    buckets: dict[int, Bucket]
    ledger: Ledger
    signature_cache: dict[int, np.ndarray]


def commit(env: Environment, records: Iterable[Record]) -> Store:
    grouped = {b: [] for b in range(1 << env.prefix)}
    seen = set()
    sources = dict(env.registry)
    signature_cache = {}
    for rec in records:
        if (rec.rid in seen or rec.epoch != env.epoch or rec.source not in sources
                or rec.modality not in env.modalities or type(rec.timestamp) is not int
                or len(rec.content_hash) != 32 or rec.rid < 0):
            raise ValueError("invalid record metadata")
        if rec.vector.shape != (env.dimension,) or not np.isfinite(rec.vector).all():
            raise ValueError("invalid record vector")
        if abs(np.linalg.norm(rec.vector.astype(np.float64)) - 1) > 1e-5:
            raise ValueError("record vector is not normalized")
        sig = signatures(rec.vector, env)[0]
        computed_bucket = sum(int(bit) << j for j, bit in enumerate(sig[:env.prefix]))
        if rec.bucket != computed_bucket:
            raise ValueError("record bucket mismatch")
        signature_cache[rec.rid] = sig
        grouped[rec.bucket].append(rec)
        seen.add(rec.rid)
    buckets = {}
    anchors = []
    for bid, rows in grouped.items():
        ordered = tuple(sorted(rows, key=lambda r: r.rid))
        leaves = tuple(r.leaf() for r in ordered)
        buckets[bid] = Bucket(bid, ordered, leaves)
        anchors.append((len(rows), bucket_commitment(env.epoch, bid, leaves)))
    return Store(env, buckets, Ledger(env.epoch, env.commitment(), tuple(anchors)), signature_cache)


@dataclass(frozen=True)
class Query:
    vector: np.ndarray
    modality: str
    epoch: int
    start: int
    end: int
    sources: tuple[str, ...]
    k: int = 20
    coarse: int = 1000
    flips: int = 4
    target_modalities: tuple[str, ...] = ("image",)

    def commitment(self):
        fields = [self.modality, self.epoch, self.start, self.end, self.sources,
                  self.k, self.coarse, self.flips, self.target_modalities]
        return digest("query", pack(fields), vector_bytes(self.vector))


def valid_query(q, env):
    return (q.vector.shape == (env.dimension,) and np.isfinite(q.vector).all()
            and abs(np.linalg.norm(q.vector.astype(np.float64)) - 1) <= 1e-5
            and q.epoch == env.epoch and q.modality in env.modalities
            and type(q.start) is int and type(q.end) is int and q.start <= q.end
            and 0 < q.k <= q.coarse and 0 <= q.flips <= env.prefix
            and tuple(sorted(set(q.sources))) == q.sources
            and set(q.sources) <= set(dict(env.registry))
            and tuple(sorted(set(q.target_modalities))) == q.target_modalities
            and set(q.target_modalities) <= set(env.modalities))


def probes(q: Query, env: Environment) -> tuple[int, ...]:
    margin = np.einsum("ij,j->i", env.projection, q.vector,
                       dtype=np.float64, optimize=False)
    main = sum(int(x >= 0) << j for j, x in enumerate(margin[:env.prefix]))
    flip = sorted(range(env.prefix), key=lambda j: (abs(margin[j]), j))[:q.flips]
    result = [main]
    for size in range(1, len(flip) + 1):
        for subset in itertools.combinations(flip, size):
            mask = sum(1 << j for j in subset)
            result.append(main ^ mask)
    return tuple(sorted(result))


def filtered(records, q):
    sources, modalities = set(q.sources), set(q.target_modalities)
    return tuple(sorted((r for r in records if q.start <= r.timestamp <= q.end
                         and r.source in sources and r.modality in modalities),
                        key=lambda r: r.rid))


def cosine_top(records, q):
    if not records:
        return ()
    matrix = np.stack([r.vector for r in records])
    scores = np.einsum("ij,j->i", matrix, q.vector, dtype=np.float64, optimize=False)
    scores = np.rint(scores * 1e8).astype(np.int64)
    ids = np.array([r.rid for r in records], dtype=np.int64)
    return tuple(int(ids[i]) for i in np.lexsort((ids, -scores))[:q.k])


def select_coarse(records, q, env, signature_cache=None):
    if not records:
        return ()
    if len(records) <= q.coarse:
        return tuple(r.rid for r in records)
    sig = (signatures(np.stack([r.vector for r in records]), env)
           if signature_cache is None else np.stack([signature_cache[r.rid] for r in records]))
    query_sig = signatures(q.vector, env)[0]
    scores = np.count_nonzero(sig == query_sig, axis=1)
    ids = np.array([r.rid for r in records], dtype=np.int64)
    keep = np.lexsort((ids, -scores))[:q.coarse]
    return tuple(sorted(int(ids[i]) for i in keep))


@dataclass(frozen=True)
class MainResponse:
    query_hash: bytes
    environment_hash: bytes
    probe_ids: tuple[int, ...]
    buckets: tuple[Bucket, ...]
    coarse_ids: tuple[int, ...]
    result: tuple[int, ...]


def retrieve(q: Query, store: Store) -> MainResponse:
    if not valid_query(q, store.env):
        raise ValueError("invalid query")
    ps = probes(q, store.env)
    bs = tuple(store.buckets[b] for b in ps)
    rows = filtered((r for b in bs for r in b.records), q)
    coarse = select_coarse(rows, q, store.env, store.signature_cache)
    selected = set(coarse)
    result = cosine_top([r for r in rows if r.rid in selected], q)
    return MainResponse(q.commitment(), store.env.commitment(), ps, bs, coarse, result)


def response_digest(vo: MainResponse, ledger: Ledger):
    parts = [vo.query_hash, vo.environment_hash, ledger.commitment(),
             pack([vo.probe_ids, vo.coarse_ids, vo.result])]
    for b in vo.buckets:
        parts.append(pack(b.bid))
        # Recompute from supplied bytes, not cached leaves, to bind actual disclosure.
        parts.append(digest("disclosure", *(r.leaf() for r in b.records)))
        parts.append(digest("leaves", *b.leaves))
    return digest("transcript", *parts)


@dataclass(frozen=True)
class Challenge:
    transcript: bytes
    seeds: tuple[bytes, ...]
    rounds: tuple[tuple[int, ...], ...]
    sample_size: int


def sample_subset(pool, size, seed):
    # Rejection sampling avoids modulo bias; each round samples without replacement.
    pool = list(pool)
    if not 0 <= size <= len(pool):
        raise ValueError("invalid sample size")
    counter = 0
    for i in range(size):
        remaining = len(pool) - i
        bound = (1 << 256) - (1 << 256) % remaining
        while True:
            value = int.from_bytes(digest("random", seed, pack(counter)), "big")
            counter += 1
            if value < bound:
                break
        j = i + value % remaining
        pool[i], pool[j] = pool[j], pool[i]
    return tuple(sorted(pool[:size]))


def challenge(vo, ledger, sample_size, rounds, *, seeds=None):
    """Called by verifier only after receipt/locking of the full main response."""
    if rounds < 1:
        raise ValueError("at least one round required")
    pool = sorted(set(range(len(ledger.anchors))) - set(vo.probe_ids))
    size = min(sample_size, len(pool))
    if seeds is None:
        seeds = tuple(secrets.token_bytes(32) for _ in range(rounds))
    if len(seeds) != rounds or any(len(s) != 32 for s in seeds):
        raise ValueError("invalid independent round seeds")
    return Challenge(response_digest(vo, ledger), tuple(seeds),
                     tuple(sample_subset(pool, size, s) for s in seeds), size)


def answer_audit(ch: Challenge, store: Store):
    ids = sorted({b for row in ch.rounds for b in row})
    return tuple(store.buckets[b] for b in ids)


@dataclass
class Verdict:
    valid: bool
    material_omission: bool
    augmented: tuple[int, ...]
    steps: list[dict]


def bucket_structure(bs, expected_ids, ledger):
    if tuple(b.bid for b in bs) != tuple(expected_ids):
        return False
    for b in bs:
        if not 0 <= b.bid < len(ledger.anchors):
            return False
        n, anchor = ledger.anchors[b.bid]
        ids = [r.rid for r in b.records]
        if (len(ids) != n or len(b.leaves) != n or ids != sorted(set(ids))
                or any(len(leaf) != 32 for leaf in b.leaves)
                or bucket_commitment(ledger.epoch, b.bid, b.leaves) != anchor):
            return False
    return True


def record_bindings(bs, env):
    seen = set()
    sources = dict(env.registry)
    for b in bs:
        if len(b.records) != len(b.leaves):
            return False
        for rec, leaf in zip(b.records, b.leaves):
            if (rec.rid in seen or rec.rid < 0 or rec.epoch != env.epoch
                    or rec.bucket != b.bid or rec.source not in sources
                    or rec.modality not in env.modalities
                    or type(rec.timestamp) is not int or len(rec.content_hash) != 32
                    or rec.vector.shape != (env.dimension,)
                    or not np.isfinite(rec.vector).all()
                    or abs(np.linalg.norm(rec.vector.astype(np.float64)) - 1) > 1e-5
                    or rec.leaf() != leaf):
                return False
            seen.add(rec.rid)
    return True


def verify(q, vo, env, ledger, ch=None, audit=(), *, skip_steps=frozenset()):
    """skip_steps is an experiment-only ablation hook; checks continue normally."""
    steps = []
    candidates = ()
    coarse = ()

    def check(index, fn):
        if index in skip_steps:
            steps.append(dict(step=index, passed=True, skipped=True, ms=0.0))
            return True
        start = perf_counter()
        reason = ""
        try:
            passed = bool(fn())
        except (ValueError, TypeError, IndexError, KeyError, AttributeError, OverflowError) as exc:
            passed, reason = False, type(exc).__name__
        steps.append(dict(step=index, passed=passed, skipped=False,
                          ms=(perf_counter() - start) * 1000, reason=reason))
        return passed

    def fail():
        return Verdict(False, False, (), steps)

    if not check(0, lambda: valid_query(q, env) and vo.query_hash == q.commitment()):
        return fail()
    if not check(1, lambda: env.commitment() == vo.environment_hash == ledger.environment_hash
                 and env.epoch == ledger.epoch and len(ledger.anchors) == 1 << env.prefix):
        return fail()
    if not check(2, lambda: vo.probe_ids == probes(q, env)):
        return fail()
    if not check(3, lambda: bucket_structure(vo.buckets, vo.probe_ids, ledger)):
        return fail()
    if not check(4, lambda: record_bindings(vo.buckets, env)):
        return fail()

    def ranking():
        nonlocal candidates, coarse
        candidates = filtered((r for b in vo.buckets for r in b.records), q)
        coarse = select_coarse(candidates, q, env)
        selected = set(coarse)
        return (coarse == vo.coarse_ids and vo.result ==
                cosine_top([r for r in candidates if r.rid in selected], q))

    if not check(5, ranking):
        return fail()

    def audit_check():
        if ch is None:
            return not audit
        pool = sorted(set(range(len(ledger.anchors))) - set(vo.probe_ids))
        if (not ch.seeds or len(ch.seeds) != len(ch.rounds)
                or any(len(s) != 32 for s in ch.seeds)
                or ch.transcript != response_digest(vo, ledger)
                or ch.rounds != tuple(sample_subset(pool, ch.sample_size, s) for s in ch.seeds)):
            return False
        ids = tuple(sorted({b for row in ch.rounds for b in row}))
        return bucket_structure(audit, ids, ledger) and record_bindings(audit, env)

    if not check(6, audit_check):
        return fail()
    # Augmentation preserves the protocol's Hamming-selected starting candidates.
    original = {r.rid: r for b in vo.buckets for r in b.records if r.rid in vo.coarse_ids}
    original.update({r.rid: r for r in filtered((r for b in audit for r in b.records), q)})
    augmented = cosine_top(list(original.values()), q) if ch is not None else vo.result
    return Verdict(True, augmented != vo.result, augmented, steps)


def check_payload(record: Record, content: bytes):
    return digest("content", content) == record.content_hash


def wire_size(vo):
    """Uncompressed serialized main response, excluding separately fetched payloads."""
    return sum(len(part) for part in wire_parts(vo))


def wire_parts(vo):
    header = pack([vo.query_hash.hex(), vo.environment_hash.hex(), vo.probe_ids,
                   vo.coarse_ids, vo.result])
    yield struct.pack(">Q", len(header))
    yield header
    for b in vo.buckets:
        yield struct.pack(">QQ", b.bid, len(b.records))
        for r, leaf in zip(b.records, b.leaves):
            meta, vec = pack(r.metadata()), vector_bytes(r.vector)
            yield struct.pack(">QQ", len(meta), len(vec))
            yield meta
            yield vec
            yield leaf


def miss_probability(n, h, s, t):
    if not (0 <= h <= n and 0 <= s <= n and t >= 1):
        raise ValueError("invalid audit parameters")
    if n == 0 or h == 0:
        return 1.0
    return (math.comb(n - h, s) / math.comb(n, s)) ** t if s <= n - h else 0.0
