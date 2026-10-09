"""TrustMR experiments: compact-response serialization and round-trip verification tests."""
import unittest

from src import compact_wire as wire
from src import protocol_compact as c
from tests import test_compact_definition_alignment as fixtures


class CompactWireTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DefinitionAlignment()
        self.fixture.setUp()
        f = self.fixture
        self.vo = c.retrieve(f.q, f.store)
        self.raw = wire.encode(self.vo)

    def test_exact_size_and_roundtrip(self):
        f = self.fixture
        self.assertEqual(len(self.raw), c.wire_size(self.vo))
        received = wire.decode(self.raw)
        self.assertEqual(received.result, self.vo.result)
        self.assertEqual(wire.encode(received), self.raw)
        self.assertTrue(c.verify(f.q, received, f.env, f.store.ledger).valid)

    def test_truncated_and_trailing(self):
        for raw in (self.raw[:-1], self.raw+b"x", b""):
            with self.assertRaises(ValueError):
                wire.decode(raw)

    def test_vector_tamper_in_received_bytes(self):
        f = self.fixture
        changed = self.raw[:-1]+bytes([self.raw[-1]^1])
        self.assertFalse(c.verify(f.q, wire.decode(changed), f.env, f.store.ledger).valid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
