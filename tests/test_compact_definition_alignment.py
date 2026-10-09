"""TrustMR experiments: compact-protocol tests for commitments, coarse selection, and audits."""
import unittest
from dataclasses import replace

import numpy as np

from src import protocol_compact as c
from src import protocol_v6 as ref


class DefinitionAlignment(unittest.TestCase):
    def setUp(self):
        self.env = c.setup(d=12, bits=16, prefix=3,
                           registry=(("S0", "A"), ("S1", "B")))
        rng = np.random.default_rng(19)
        self.records = []
        for i, raw in enumerate(rng.normal(size=(120, 12))):
            v = ref.unit(raw)
            self.records.append(ref.Record(i, ref.digest("content", str(i).encode()),
                v, "image", i, f"S{i % 2}", 1, ref.bucket_id(v, self.env)))
        self.store = c.commit(self.env, self.records)
        self.q = ref.Query(ref.unit(rng.normal(size=12)), "text", 1, 0, 119,
                           ("S0", "S1"), k=4, coarse=12, flips=1)

    def exact(self, ids):
        q = self.q
        def key(i):
            score = sum(float(a)*float(b) for a, b in zip(self.store.vectors[i], q.vector))
            return (-int(np.rint(score*1e8)), i)
        return tuple(sorted(set(ids), key=key)[:q.k])

    def test_commit_derives_signature_and_vector_hash(self):
        for bucket in self.store.buckets.values():
            for row in bucket.records:
                vector = self.store.vectors[row.rid]
                expected = np.packbits(ref.signatures(vector, self.env)[0],
                                       bitorder="little").tobytes()
                self.assertEqual(row.signature, expected)
                self.assertEqual(row.vector_hash, ref.digest("vector", ref.vector_bytes(vector)))

    def test_commit_rejects_incorrect_bucket(self):
        row = self.records[0]
        with self.assertRaises(ValueError):
            c.commit(self.env, [replace(row, bucket=row.bucket ^ 1)])

    def test_coarse_definition_matches_independent_hamming(self):
        self.q = replace(self.q, start=20, end=100, sources=("S1",), coarse=4)
        vo = c.retrieve(self.q, self.store)
        pool = [r for b in vo.buckets for r in b.records
                if 20 <= r.timestamp <= 100 and r.source == "S1"]
        qs = np.packbits(ref.signatures(self.q.vector, self.env)[0], bitorder="little").tobytes()
        key = lambda r: (sum((a ^ b).bit_count() for a, b in zip(r.signature, qs)), r.rid)
        expected = tuple(sorted(r.rid for r in sorted(pool, key=key)[:4]))
        self.assertEqual(vo.coarse_ids, expected)
        self.assertEqual(vo.result, self.exact(expected))
        self.assertTrue(c.verify(self.q, vo, self.env, self.store.ledger).valid)

    def test_full_pool_topk_can_differ_from_actual_result(self):
        self.q = replace(self.q, coarse=4, flips=3)
        vo = c.retrieve(self.q, self.store)
        all_ids = [r.rid for b in vo.buckets for r in b.records]
        self.assertNotEqual(self.exact(all_ids), vo.result)
        self.assertEqual(self.exact(vo.coarse_ids), vo.result)

    def test_audit_starts_from_coarse_set_and_does_not_rerun_hamming(self):
        vo = c.retrieve(self.q, self.store)
        ch = c.challenge(vo, self.store.ledger, 2, 1, seeds=(b"a"*32,))
        audit = c.answer_audit(self.q, ch, self.store)
        verdict = c.verify(self.q, vo, self.env, self.store.ledger, ch, audit)
        ids = set(vo.coarse_ids) | {r.rid for b in audit.buckets for r in b.records}
        self.assertTrue(verdict.valid)
        self.assertEqual(verdict.augmented, self.exact(ids))
        self.assertEqual(verdict.material_omission, verdict.augmented != vo.result)
        self.assertEqual({i for i, _ in audit.openings}, ids - set(vo.coarse_ids))

    def test_audit_filters_vector_openings_but_keeps_complete_headers(self):
        self.q = replace(self.q, start=20, end=100, sources=("S1",))
        vo = c.retrieve(self.q, self.store)
        ch = c.challenge(vo, self.store.ledger, 2, 1, seeds=(b"a"*32,))
        audit = c.answer_audit(self.q, ch, self.store)
        rows = [r for b in audit.buckets for r in b.records]
        expected = {r.rid for r in rows if 20 <= r.timestamp <= 100 and r.source == "S1"}
        self.assertGreater(len(rows), len(expected))
        self.assertEqual({i for i, _ in audit.openings}, expected)
        self.assertTrue(c.verify(self.q, vo, self.env, self.store.ledger, ch, audit).valid)

    def test_signature_rewrite_after_fixed_commit_is_rejected(self):
        vo = c.retrieve(self.q, self.store)
        b = next(b for b in vo.buckets if b.records)
        r = b.records[0]
        signature = bytearray(r.signature)
        signature[-1] ^= 128
        changed = replace(r, signature=bytes(signature))
        rows = (changed,) + b.records[1:]
        forged = replace(b, records=rows, leaves=tuple(r.leaf() for r in rows))
        vo = replace(vo, buckets=tuple(forged if old.bid == b.bid else old for old in vo.buckets))
        self.assertFalse(c.verify(self.q, vo, self.env, self.store.ledger).valid)

    def test_bad_initial_signature_is_outside_current_guarantee(self):
        # Bypass ingestion to test the assumption that the initial commitment is trusted.
        original = ref.commit(self.env, self.records)
        for signature in original.signature_cache.values():
            signature[-1] = ~signature[-1]
        bad = c.from_validated_store(original)
        self.q = replace(self.q, coarse=1000, flips=3)
        vo = c.retrieve(self.q, bad)
        self.assertTrue(c.verify(self.q, vo, self.env, bad.ledger).valid)
        mismatches = [r for b in vo.buckets for r in b.records
                      if r.signature != np.packbits(ref.signatures(bad.vectors[r.rid], self.env)[0],
                                                    bitorder="little").tobytes()]
        self.assertEqual(len(mismatches), len(self.records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
