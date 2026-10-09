"""TrustMR experiments: compact-protocol equivalence and tamper-rejection tests."""
from dataclasses import replace
import numpy as np
import pytest

from tests.test_protocol_v6 import case
from src import protocol_v6 as ref
from src import protocol_compact as c


@pytest.fixture
def compact(case):
    env, records, original, q = case
    env = c.Environment(**vars(env))
    store = c.commit(env, records)
    return env, original, store, q


@pytest.mark.parametrize("flips", [0, 1, 2, 3])
@pytest.mark.parametrize("coarse", [4, 12, 1000])
def test_same_retrieval_and_audit(compact, flips, coarse):
    env, original, store, q = compact
    q = replace(q, flips=flips, coarse=coarse)
    old = ref.retrieve(q, original)
    vo = c.retrieve(q, store)
    assert vo.coarse_ids == old.coarse_ids and vo.result == old.result
    assert c.verify(q, vo, env, store.ledger).valid
    seeds = (b"a"*32, b"b"*32)
    ch = c.challenge(vo, store.ledger, 2, 2, seeds=seeds)
    oldch = ref.challenge(old, original.ledger, 2, 2, seeds=seeds)
    result = c.verify(q, vo, env, store.ledger, ch, c.answer_audit(q, ch, store))
    expected = ref.verify(q, old, original.env, original.ledger, oldch,
                          ref.answer_audit(oldch, original))
    assert result.valid and expected.valid
    assert result.augmented == expected.augmented
    assert result.material_omission == expected.material_omission


@pytest.mark.parametrize("field,value", [("signature", b"xx"), ("vector_hash", b"x"*32),
                         ("timestamp", 999), ("source", "absent"),
                         ("content_hash", b"x"*32), ("epoch", 2), ("rid", 999)])
def test_entry_tamper(compact, field, value):
    env, _, store, q = compact
    vo = c.retrieve(q, store)
    b = vo.buckets[0]
    altered = replace(b.records[0], **{field:value})
    bad = replace(vo, buckets=(replace(b, records=(altered,)+b.records[1:]),)+vo.buckets[1:])
    assert not c.verify(q, bad, env, store.ledger).valid
    # Rehashing the leaf must not permit a forged tree against the trusted anchor.
    bad = replace(bad, buckets=(replace(bad.buckets[0], leaves=(altered.leaf(),)+b.leaves[1:]),)+bad.buckets[1:])
    assert not c.verify(q, bad, env, store.ledger).valid


def test_openings_and_rankings(compact):
    env, _, store, q = compact
    vo = c.retrieve(q, store)
    rid, vec = vo.openings[0]
    changed = ((rid, -vec),)+vo.openings[1:]
    for bad in (replace(vo, openings=changed), replace(vo, openings=vo.openings[1:]),
                replace(vo, result=vo.result[::-1]), replace(vo, coarse_ids=vo.coarse_ids[1:])):
        assert not c.verify(q, bad, env, store.ledger).valid
    lower = tuple((i,v) for i,v in vo.openings if i != vo.result[0])
    assert not c.verify(q, replace(vo, result=c.top(lower,q)), env, store.ledger).valid


def test_audit_missing_opening_and_transcript(compact):
    env, _, store, q = compact
    vo = c.retrieve(q, store)
    ch = c.challenge(vo, store.ledger, 4, 1, seeds=(b"z"*32,))
    audit = c.answer_audit(q, ch, store)
    assert audit.openings
    assert not c.verify(q, vo, env, store.ledger, ch, replace(audit, openings=audit.openings[1:])).valid
    i, v = vo.openings[0]
    bad = replace(vo, openings=((i,-v),)+vo.openings[1:])
    assert not c.verify(q, bad, env, store.ledger, ch, audit, skip_steps={5}).valid


def test_compact_root_not_legacy_root(compact):
    env, old, store, q = compact
    assert not c.verify(q, c.retrieve(q,store), env, old.ledger).valid


def test_coherent_routing_skip(compact):
    env, _, store, q = compact
    vo = c.retrieve(q, store)
    buckets = vo.buckets[1:]
    rows = ref.filtered((r for b in buckets for r in b.records), q)
    coarse = c.select_coarse(rows,q,env)
    openings = tuple((i,store.vectors[i]) for i in coarse)
    bad = replace(vo,buckets=buckets,probe_ids=tuple(b.bid for b in buckets),
                  coarse_ids=coarse,openings=openings,result=c.top(openings,q))
    assert not c.verify(q,bad,env,store.ledger).valid
    assert c.verify(q,bad,env,store.ledger,skip_steps={2}).valid


def test_ingestion_rejects_wrong_bucket(compact):
    env, original, _, _ = compact
    r = next(r for b in original.buckets.values() for r in b.records)
    with pytest.raises(ValueError):
        c.commit(env,[replace(r,bucket=r.bucket ^ 1)])
