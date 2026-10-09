"""TrustMR experiments: signature-bound headers and selective vector verification."""
from dataclasses import dataclass, replace
import struct
from time import perf_counter

import numpy as np

from . import protocol_v6 as ref


@dataclass(frozen=True)
class Environment(ref.Environment):
    def commitment(self):
        return ref.digest("compact-environment-v1", super().commitment(),
                          b"signature-packbits-little;trusted-commit;selective-vector-openings")


def setup(**kwargs):
    return Environment(**vars(ref.setup(**kwargs)))


@dataclass(frozen=True)
class Entry:
    rid: int
    content_hash: bytes
    modality: str
    timestamp: int
    source: str
    epoch: int
    bucket: int
    vector_hash: bytes
    signature: bytes

    def metadata(self):
        return [self.rid, self.content_hash.hex(), self.modality,
                self.timestamp, self.source, self.epoch, self.bucket]

    def leaf(self):
        return ref.digest("signed-record-v1", ref.pack(self.metadata()),
                          self.vector_hash, self.signature)


@dataclass
class Store:
    env: Environment
    ledger: ref.Ledger
    buckets: dict
    vectors: dict


def from_validated_store(original):
    if not isinstance(original.env, Environment):
        raise ValueError("compact environment required")
    buckets, anchors, vectors = {}, [], {}
    for bid, bucket in original.buckets.items():
        rows = tuple(Entry(r.rid, r.content_hash, r.modality, r.timestamp,
                           r.source, r.epoch, r.bucket,
                           ref.digest("vector", ref.vector_bytes(r.vector)),
                           np.packbits(original.signature_cache[r.rid], bitorder="little").tobytes())
                     for r in bucket.records)
        leaves = tuple(r.leaf() for r in rows)
        buckets[bid] = ref.Bucket(bid, rows, leaves)
        anchors.append((len(rows), ref.bucket_commitment(original.env.epoch, bid, leaves)))
        vectors.update((r.rid, r.vector) for r in bucket.records)
    return Store(original.env, ref.Ledger(original.env.epoch,
                 original.env.commitment(), tuple(anchors)), buckets, vectors)


def commit(env, records):
    # Validate vector-to-bucket placement before constructing compact commitments.
    return from_validated_store(ref.commit(env, records))


POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def select_coarse(rows, q, env):
    if len(rows) <= q.coarse:
        return tuple(r.rid for r in rows)
    width = (env.projection.shape[0] + 7)//8
    packed = np.frombuffer(b"".join(r.signature for r in rows), dtype=np.uint8).reshape(-1, width)
    qs = np.packbits(ref.signatures(q.vector, env)[0], bitorder="little")
    distances = POPCOUNT[packed ^ qs].sum(axis=1)
    ids = np.array([r.rid for r in rows], dtype=np.int64)
    chosen = np.lexsort((ids, distances))[:q.coarse]
    return tuple(sorted(int(ids[i]) for i in chosen))


def top(openings, q):
    if not openings:
        return ()
    ids = np.array([rid for rid, _ in openings], dtype=np.int64)
    scores = np.rint(np.einsum("ij,j->i", np.stack([v for _, v in openings]), q.vector,
                               dtype=np.float64, optimize=False)*1e8).astype(np.int64)
    return tuple(int(ids[i]) for i in np.lexsort((ids, -scores))[:q.k])


@dataclass(frozen=True)
class MainResponse:
    query_hash: bytes
    environment_hash: bytes
    probe_ids: tuple
    buckets: tuple
    coarse_ids: tuple
    result: tuple
    openings: tuple


def retrieve(q, store):
    if not ref.valid_query(q, store.env):
        raise ValueError("invalid query")
    ps = ref.probes(q, store.env)
    buckets = tuple(store.buckets[i] for i in ps)
    rows = ref.filtered((r for b in buckets for r in b.records), q)
    coarse = select_coarse(rows, q, store.env)
    openings = tuple((i, store.vectors[i]) for i in coarse)
    return MainResponse(q.commitment(), store.env.commitment(), ps, buckets,
                        coarse, top(openings, q), openings)


def response_digest(vo, ledger):
    return ref.digest("compact-transcript-v1", ref.response_digest(vo, ledger),
                      ref.digest("openings", *(ref.digest("opening", ref.pack(i), ref.vector_bytes(v))
                                               for i, v in vo.openings)))


def challenge(vo, ledger, sample_size, rounds, *, seeds=None):
    ch = ref.challenge(vo, ledger, sample_size, rounds, seeds=seeds)
    return replace(ch, transcript=response_digest(vo, ledger))


@dataclass(frozen=True)
class AuditResponse:
    buckets: tuple
    openings: tuple


def answer_audit(q, ch, store):
    ids = sorted({i for row in ch.rounds for i in row})
    buckets = tuple(store.buckets[i] for i in ids)
    rows = ref.filtered((r for b in buckets for r in b.records), q)
    return AuditResponse(buckets, tuple((r.rid, store.vectors[r.rid]) for r in rows))


def bindings(buckets, env):
    seen = set()
    width = (env.projection.shape[0]+7)//8
    for b in buckets:
        if len(b.records) != len(b.leaves):
            return False
        for r, leaf in zip(b.records, b.leaves):
            if (type(r.rid) is not int or r.rid < 0 or r.rid in seen
                    or r.bucket != b.bid or r.epoch != env.epoch
                    or r.source not in dict(env.registry) or r.modality not in env.modalities
                    or type(r.timestamp) is not int or len(r.content_hash) != 32
                    or len(r.vector_hash) != 32 or len(r.signature) != width or r.leaf() != leaf):
                return False
            bits = np.unpackbits(np.frombuffer(r.signature, dtype=np.uint8), bitorder="little")
            if (bits[env.projection.shape[0]:].any()
                    or sum(int(v)<<j for j, v in enumerate(bits[:env.prefix])) != b.bid):
                return False
            seen.add(r.rid)
    return True


def validate_openings(openings, entries, expected, env):
    if tuple(i for i, _ in openings) != tuple(expected):
        return False
    for i, vec in openings:
        if (vec.shape != (env.dimension,) or not np.isfinite(vec).all()
                or abs(np.linalg.norm(vec.astype(np.float64))-1) > 1e-5
                or ref.digest("vector", ref.vector_bytes(vec)) != entries[i].vector_hash):
            return False
    return True


def verify(q, vo, env, ledger, ch=None, audit=None, *, skip_steps=frozenset()):
    steps = []

    def check(i, fn):
        if i in skip_steps:
            steps.append(dict(step=i, passed=True, skipped=True, ms=0.0))
            return True
        start = perf_counter()
        try:
            ok = bool(fn())
        except (ValueError, TypeError, IndexError, KeyError, AttributeError, OverflowError):
            ok = False
        steps.append(dict(step=i, passed=ok, skipped=False, ms=(perf_counter()-start)*1000))
        return ok

    def fail():
        return ref.Verdict(False, False, (), steps)

    if not check(0, lambda: ref.valid_query(q, env) and vo.query_hash == q.commitment()):
        return fail()
    if not check(1, lambda: isinstance(env, Environment) and env.epoch == ledger.epoch
                 and len(ledger.anchors) == 1<<env.prefix
                 and env.commitment() == ledger.environment_hash == vo.environment_hash):
        return fail()
    if not check(2, lambda: vo.probe_ids == ref.probes(q, env)):
        return fail()
    if not check(3, lambda: ref.bucket_structure(vo.buckets, vo.probe_ids, ledger)):
        return fail()
    if not check(4, lambda: bindings(vo.buckets, env)):
        return fail()

    def ranking():
        rows = ref.filtered((r for b in vo.buckets for r in b.records), q)
        expected = select_coarse(rows, q, env)
        entries = {r.rid:r for r in rows}
        return (vo.coarse_ids == expected and validate_openings(vo.openings, entries, expected, env)
                and top(vo.openings, q) == vo.result)

    if not check(5, ranking):
        return fail()

    def audit_check():
        if ch is None:
            return audit is None
        pool = sorted(set(range(len(ledger.anchors)))-set(vo.probe_ids))
        if (audit is None or not ch.seeds or len(ch.seeds) != len(ch.rounds)
                or any(len(s) != 32 for s in ch.seeds)
                or ch.transcript != response_digest(vo, ledger)
                or ch.rounds != tuple(ref.sample_subset(pool, ch.sample_size, s) for s in ch.seeds)):
            return False
        ids = tuple(sorted({i for row in ch.rounds for i in row}))
        if not ref.bucket_structure(audit.buckets, ids, ledger) or not bindings(audit.buckets, env):
            return False
        rows = ref.filtered((r for b in audit.buckets for r in b.records), q)
        return validate_openings(audit.openings, {r.rid:r for r in rows}, tuple(r.rid for r in rows), env)

    if not check(6, audit_check):
        return fail()
    augmented = top(vo.openings + audit.openings, q) if ch is not None else vo.result
    return ref.Verdict(True, augmented != vo.result, augmented, steps)


def disclosure_parts(buckets, openings):
    yield struct.pack(">QQ", len(buckets), len(openings))
    for b in buckets:
        yield struct.pack(">QQ", b.bid, len(b.records))
        for r, leaf in zip(b.records, b.leaves):
            meta = ref.pack(r.metadata())
            yield struct.pack(">Q", len(meta))
            yield meta
            yield r.vector_hash
            yield struct.pack(">Q", len(r.signature))
            yield r.signature
            yield leaf
    for rid, vec in openings:
        raw = ref.vector_bytes(vec)
        yield struct.pack(">QQ", rid, len(raw))
        yield raw


def wire_size(vo):
    header = ref.pack([vo.query_hash.hex(), vo.environment_hash.hex(),
                       vo.probe_ids, vo.coarse_ids, vo.result])
    return 8+len(header)+sum(map(len, disclosure_parts(vo.buckets, vo.openings)))


def audit_wire_size(audit):
    return sum(map(len, disclosure_parts(audit.buckets, audit.openings)))
