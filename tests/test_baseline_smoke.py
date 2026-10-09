"""TrustMR experiments: V-GANN and VSQ response-verification smoke tests."""
import numpy as np
import pytest

from baselines.vgann.reproduction_v3 import core_r2 as vgann
from baselines.vsq.vector import core as vsq
from src.protocol_v6 import unit


def test_vgann_r2_response_and_verifier(tmp_path):
    rng = np.random.default_rng(42)
    points = np.asarray([unit(v) for v in rng.normal(size=(40, 8))])
    graph = [np.asarray([j for j in range(40) if j != i]) for i in range(40)]
    index = vgann.Index(points, graph, tmp_path)
    query = unit(rng.normal(size=8))
    response, wire, _, sizes = index.response(query, 5, "smoke:0")
    assert len(response["result"]) == 5
    assert len(wire) == sizes["result_plus_proof_bytes"]
    vgann.verify(wire, query, 5, "smoke:0", 40, 8, index.public, index.root)
    with pytest.raises(ValueError):
        vgann.verify(wire[:-1], query, 5, "smoke:0", 40, 8, index.public, index.root)


def test_vsq_vector_response_and_verifier():
    rng = np.random.default_rng(42)
    points = np.asarray([unit(v) for v in rng.normal(size=(40, 1024))])
    projection = rng.normal(size=(256, 1024)).astype("<f4")
    signatures = np.asarray([np.frombuffer(vsq.signature(v, projection), dtype="u1") for v in points])
    hashes = np.asarray([np.frombuffer(vsq.vh(v), dtype="u1") for v in points])
    index = vsq.Index(points, signatures, hashes, projection, 32, 8)
    response, wire, _, sizes = index.response(points[0], k=5)
    assert response["result"][0] == 0
    assert len(wire) == sizes["result_plus_proof_bytes"]
    vsq.verify(wire, points[0], vsq.anchor(index), projection, k=5)
    with pytest.raises(ValueError):
        vsq.verify(wire[:-1], points[0], vsq.anchor(index), projection, k=5)
