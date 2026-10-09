"""TrustMR experiments: binary encoding and decoding of compact query responses."""
import json
import struct

import numpy as np

from . import protocol_compact as c
from . import protocol_v6 as ref


def encode(vo):
    header = ref.pack([vo.query_hash.hex(), vo.environment_hash.hex(),
                       vo.probe_ids, vo.coarse_ids, vo.result])
    return b"".join((struct.pack(">Q", len(header)), header,
                     *c.disclosure_parts(vo.buckets, vo.openings)))


def decode(raw):
    view, offset = memoryview(raw), 0

    def take(n):
        nonlocal offset
        if n < 0 or n > len(view)-offset:
            raise ValueError("truncated compact response")
        value = view[offset:offset+n]
        offset += n
        return value

    def u64():
        return struct.unpack(">Q", take(8))[0]

    header = json.loads(take(u64()).tobytes())
    bucket_count, opening_count = u64(), u64()
    buckets = []
    for _ in range(bucket_count):
        bid, count = u64(), u64()
        rows, leaves = [], []
        for _ in range(count):
            meta = json.loads(take(u64()).tobytes())
            vector_hash = take(32).tobytes()
            signature = take(u64()).tobytes()
            leaves.append(take(32).tobytes())
            rows.append(c.Entry(meta[0], bytes.fromhex(meta[1]), *meta[2:], vector_hash, signature))
        buckets.append(ref.Bucket(bid, tuple(rows), tuple(leaves)))
    openings = []
    for _ in range(opening_count):
        rid, size = u64(), u64()
        openings.append((rid, np.frombuffer(take(size), dtype="<f4")))
    if offset != len(view):
        raise ValueError("trailing compact response")
    return c.MainResponse(bytes.fromhex(header[0]), bytes.fromhex(header[1]),
                          tuple(header[2]), tuple(buckets), tuple(header[3]),
                          tuple(header[4]), tuple(openings))
