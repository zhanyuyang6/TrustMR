"""TrustMR experiments: exact-neighbor ranking and input-ID consistency tests."""
import numpy as np
import h5py
import pytest

from scripts.prepare_benchmark_inputs import exact_topk, read_embeddings


def test_exact_topk_quantized_ties_and_scores():
    points = np.asarray([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype="<f4")
    queries = np.asarray([[1, 0], [0, 1]], dtype="<f4")
    ids, scores, quantized = exact_topk(points, queries, k=3)
    np.testing.assert_array_equal(ids, [[0, 1, 2], [2, 0, 1]])
    np.testing.assert_array_equal(scores, [[1, 1, 0], [1, 0, 0]])
    np.testing.assert_array_equal(quantized, np.rint(scores * 1e8).astype(np.int64))


def test_input_id_order_and_split_binding(tmp_path):
    with h5py.File(tmp_path / "embeddings.h5", "w") as h5:
        h5["image_ids"] = [100, 200, 300]
        h5["splits"] = np.asarray([b"train", b"val", b"val"])
        h5["image_embeddings"] = np.eye(3, dtype="<f4")
        rows = [{"hdf5_index": "2", "coco_image_id": "300"},
                {"hdf5_index": "1", "coco_image_id": "200"}]
        np.testing.assert_array_equal(read_embeddings(h5, rows, "image", "val"),
                                      np.eye(3)[[2, 1]])
        with pytest.raises(ValueError, match="split"):
            read_embeddings(h5, rows, "image", "train")
        rows[0]["coco_image_id"] = "200"
        with pytest.raises(ValueError, match="image IDs"):
            read_embeddings(h5, rows, "image", "val")
