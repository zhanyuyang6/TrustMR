"""TrustMR experiments: full-validation query expansion and exact-ranking tests."""
import numpy as np
import pytest

from experiments.full_validation_inputs import expanded_indices, exact_neighbors, normalize_queries
from src.protocol_v6 import unit


def test_full_query_order_preserves_archived_prefix():
    assert expanded_indices([8, 6], [5, 6, 7, 8, 9]) == [8, 6, 5, 7, 9]
    with pytest.raises(ValueError):
        expanded_indices([8, 8], [5, 6, 7, 8, 9])
    with pytest.raises(ValueError):
        expanded_indices([1], [5, 6, 7, 8, 9])


def test_query_normalization_matches_frozen_pipeline():
    raw = np.random.default_rng(4).normal(size=(10, 1024)).astype(np.float32)
    selected = raw[[7, 3, 1]].copy()
    selected /= np.linalg.norm(selected, axis=1, keepdims=True) + 1e-12
    expected = np.asarray([unit(v) for v in selected])
    np.testing.assert_array_equal(normalize_queries(raw)[[7, 3, 1]], expected)


def test_exact_ground_truth_ties():
    points = np.asarray([[1, 0], [0, 1], [1, 0], [-1, 0]], dtype=np.float32)
    ids, scores, quantized = exact_neighbors(points, points[0], 3)
    np.testing.assert_array_equal(ids, [0, 2, 1])
    np.testing.assert_array_equal(scores, [1, 1, 0])
    np.testing.assert_array_equal(quantized, [100000000, 100000000, 0])


def test_instrumented_trustmr_matches_native_response():
    from experiments.run_full_validation import trustmr_response
    from src import compact_wire, protocol_compact
    from tests.test_compact_definition_alignment import DefinitionAlignment

    fixture = DefinitionAlignment()
    fixture.setUp()
    _, wire, stats = trustmr_response(fixture.q, fixture.store)
    expected = compact_wire.encode(protocol_compact.retrieve(fixture.q, fixture.store))
    assert wire == expected
    assert stats["candidate_count"] > 0
