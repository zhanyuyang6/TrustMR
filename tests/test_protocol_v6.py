"""TrustMR experiments: reference-protocol retrieval, filtering, and security tests."""
from dataclasses import replace

import numpy as np
import pytest

from src.protocol_v6 import (
    Record, Query, setup, unit, bucket_id, commit, digest, retrieve,
    verify, challenge, answer_audit, cosine_top, filtered, select_coarse,
    miss_probability, sample_subset, check_payload, response_digest,
)


@pytest.fixture
def case():
    env = setup(d=12, bits=16, prefix=3, registry=(("S0", "A"), ("S1", "B")))
    rng = np.random.default_rng(19)
    records = []
    for i, vector in enumerate(rng.normal(size=(120, 12))):
        vector = unit(vector)
        records.append(Record(i, digest("content", str(i).encode()), vector,
                              "image" if i % 3 else "text", i, f"S{i % 2}",
                              env.epoch, bucket_id(vector, env)))
    store = commit(env, records)
    q = Query(unit(rng.normal(size=12)), "text", 1, 0, 119,
              ("S0", "S1"), k=4, coarse=12, flips=2)
    return env, records, store, q


def edit_record(vo, name, value):
    idx = next(i for i, b in enumerate(vo.buckets) if b.records)
    bs = list(vo.buckets)
    rows = list(bs[idx].records)
    rows[0] = replace(rows[0], **{name: value})
    bs[idx] = replace(bs[idx], records=tuple(rows))
    return replace(vo, buckets=tuple(bs))


def test_honest_main_and_audit(case):
    env, _, store, q = case
    vo = retrieve(q, store)
    assert verify(q, vo, env, store.ledger).valid
    ch = challenge(vo, store.ledger, 2, 3, seeds=(b"a"*32, b"b"*32, b"c"*32))
    result = verify(q, vo, env, store.ledger, ch, answer_audit(ch, store))
    assert result.valid
    assert [s["step"] for s in result.steps] == list(range(7))


@pytest.mark.parametrize("name,value", [
    ("timestamp", 8888), ("source", "S1"), ("content_hash", b"x"*32),
    ("rid", 99999), ("epoch", 2), ("modality", "image"), ("bucket", 99),
])
def test_record_fields_are_bound(case, name, value):
    env, _, store, q = case
    vo = retrieve(q, store)
    original = next(b.records[0] for b in vo.buckets if b.records)
    if getattr(original, name) == value:
        value = "text" if name == "modality" else "S0"
    assert not verify(q, edit_record(vo, name, value), env, store.ledger).valid


def test_vector_tamper(case):
    env, _, store, q = case
    vo = retrieve(q, store)
    assert not verify(q, edit_record(vo, "vector", unit(np.ones(12))), env, store.ledger).valid


def test_forged_response_consistent_with_forged_tree_rejected(case):
    env, records, store, q = case
    changed = list(records)
    r = changed[0]
    changed[0] = replace(r, timestamp=9999)
    forged = commit(env, changed)
    vo = retrieve(q, forged)
    q = replace(q, flips=3)
    vo = retrieve(q, forged)
    assert verify(q, vo, env, forged.ledger).valid
    assert not verify(q, vo, env, store.ledger).valid


def test_missing_entire_bucket(case):
    env, _, store, q = case
    vo = retrieve(q, store)
    assert not verify(q, replace(vo, buckets=vo.buckets[1:]), env, store.ledger).valid


def test_missing_member_and_leaf(case):
    env, _, store, q = case
    vo = retrieve(q, store)
    idx = next(i for i, b in enumerate(vo.buckets) if b.records)
    bs = list(vo.buckets)
    bs[idx] = replace(bs[idx], records=bs[idx].records[1:], leaves=bs[idx].leaves[1:])
    assert not verify(q, replace(vo, buckets=tuple(bs)), env, store.ledger).valid


def test_legal_lower_ranked_records_rejected(case):
    env, _, store, q = case
    vo = retrieve(q, store)
    rows = [r for b in vo.buckets for r in b.records
            if r.rid in vo.coarse_ids and r.rid != vo.result[0]]
    bad = replace(vo, result=cosine_top(rows, q))
    assert bad.result != vo.result
    assert not verify(q, bad, env, store.ledger).valid


def test_hamming_selection_cannot_be_chosen_by_presenter(case):
    env, _, store, q = case
    vo = retrieve(q, store)
    ids = vo.coarse_ids[1:]
    rows = [r for b in vo.buckets for r in b.records if r.rid in ids]
    bad = replace(vo, coarse_ids=ids, result=cosine_top(rows, q))
    assert not verify(q, bad, env, store.ledger).valid


def test_filter_and_target_modality(case):
    env, records, store, q = case
    q = replace(q, start=30, end=50, sources=("S1",), flips=3, coarse=1000)
    vo = retrieve(q, store)
    assert vo.result == cosine_top(filtered(records, q), q)
    assert verify(q, vo, env, store.ledger).valid
    assert all(records[i].source == "S1" and 30 <= i <= 50
               and records[i].modality == "image" for i in vo.result)


def test_full_buckets_no_truncation_equals_exact(case):
    env, records, store, q = case
    q = replace(q, flips=3, coarse=1000)
    vo = retrieve(q, store)
    assert len(vo.probe_ids) == 8
    assert vo.result == cosine_top(filtered(records, q), q)


def test_audit_discovery_is_valid_and_augmented(case):
    env, records, store, q = case
    q = replace(q, flips=0, coarse=1000)
    vo = retrieve(q, store)
    ch = challenge(vo, store.ledger, 8, 1, seeds=(b"d"*32,))
    out = verify(q, vo, env, store.ledger, ch, answer_audit(ch, store))
    assert out.valid
    assert out.augmented == cosine_top(filtered(records, q), q)
    assert out.material_omission == (out.augmented != vo.result)


def test_missing_audit_rejected(case):
    env, _, store, q = case
    vo = retrieve(q, store)
    ch = challenge(vo, store.ledger, 2, 1, seeds=(b"e"*32,))
    assert not verify(q, vo, env, store.ledger, ch, ()).valid


def test_transcript_replacement_detected_even_if_ranking_skipped(case):
    env, _, store, q = case
    vo = retrieve(q, store)
    ch = challenge(vo, store.ledger, 1, 1, seeds=(b"f"*32,))
    bad = replace(vo, result=tuple(reversed(vo.result)))
    out = verify(q, bad, env, store.ledger, ch, answer_audit(ch, store), skip_steps={5})
    assert not out.valid and out.steps[-1]["step"] == 6


def test_skip_is_real_control_flow(case):
    env, _, store, q = case
    vo = replace(retrieve(q, store), query_hash=b"x"*32)
    out = verify(q, vo, env, store.ledger, skip_steps={0})
    assert out.valid
    assert out.steps[0]["skipped"]
    assert len(out.steps) == 7
    bad = edit_record(vo, "timestamp", 9999)
    out = verify(q, bad, env, store.ledger, skip_steps={0})
    assert not out.valid and out.steps[-1]["step"] == 4


def test_registry_replacement(case):
    env, _, store, q = case
    vo = retrieve(q, store)
    altered = replace(env, registry=(("S0", "other"), ("S1", "B")))
    assert not verify(q, vo, altered, store.ledger).valid


def test_no_candidates_and_no_unprobed(case):
    env, _, store, q = case
    q = replace(q, sources=(), flips=3)
    vo = retrieve(q, store)
    ch = challenge(vo, store.ledger, 2, 1, seeds=(b"g"*32,))
    out = verify(q, vo, env, store.ledger, ch, answer_audit(ch, store))
    assert out.valid and out.augmented == () and not out.material_omission


def test_payload_check(case):
    _, records, _, _ = case
    assert check_payload(records[0], b"0")
    assert not check_payload(records[0], b"1")


def test_sampling_and_formula():
    for i in range(30):
        chosen = sample_subset(range(10), 4, i.to_bytes(32, "big"))
        assert len(chosen) == len(set(chosen)) == 4
    assert miss_probability(10, 1, 2, 3) == pytest.approx(0.8**3)
    assert miss_probability(10, 3, 10, 1) == 0
    assert miss_probability(0, 0, 0, 1) == 1
