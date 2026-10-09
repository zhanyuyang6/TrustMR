"""TrustMR experiments: regression test for coherent probe-scope omission."""
from dataclasses import replace

from tests.test_protocol_v6 import case
from src.protocol_v6 import cosine_top, filtered, retrieve, select_coarse, verify


def test_coherent_route_omission_requires_route_check(case):
    env, _, store, q = case
    vo = retrieve(q, store)
    # Keep the disclosed buckets and their rankings mutually consistent.
    buckets = vo.buckets[1:]
    rows = filtered((r for b in buckets for r in b.records), q)
    coarse = select_coarse(rows, q, env)
    selected = set(coarse)
    bad = replace(vo, probe_ids=tuple(b.bid for b in buckets), buckets=buckets,
                  coarse_ids=coarse,
                  result=cosine_top([r for r in rows if r.rid in selected], q))
    full = verify(q, bad, env, store.ledger)
    assert not full.valid
    assert full.steps[-1]["step"] == 2
    skipped = verify(q, bad, env, store.ledger, skip_steps={2})
    assert skipped.valid
    assert len(skipped.steps) == 7
    assert skipped.steps[2]["skipped"]
