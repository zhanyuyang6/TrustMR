# Baselines and Measurement

## V-GANN

The implementation combines HCNNG, guided neighbor tuples, Merkle authentication,
and verifier-side replay. HCNNG is fetched from the authors' repository at the
commit recorded in the README. `reproducibility.patch` fixes the clustering RNG
and equal-weight edge ordering without changing the partition or MST rules.
Graph construction uses a cluster size of 1,000, 20 iterations, and the native
degree-three MST construction.

The preserved `faithful_v0.py` records the initial interpretation of the paper.
R1 starts with an empty result set and evaluates its maximum only after reaching
K results. R2 adds original graph neighbors at a guided-search dead end while
the result is underfull. R3 adds deterministic restarts. The runner chooses the
first version passing all correctness gates at 1,000, 5,000, 20,000, and 118,287
records. The archived experiments selected R2, which does not use R3 restarts.
Selection uses correctness, not Recall. R1 paired diagnostics remain separate
from the final competitor.

Authentication uses SHA-256, an Ed25519-signed root, a padded binary Merkle tree,
and a canonical binary response. The public query-ID-derived start is checked by
the verifier. Distance ordering uses float64 squared L2 and record-ID tie breaks.
Query vectors are already normalized. `reference_check.py` independently checks
the traversal against the raw graph; `validate.py` reloads the saved index and
checks result identities, payload sizes, and reproducibility.

## VSQ

`original/run.py` implements the original set-valued MinHash/Jaccard setting and
its authentication checks on synthetic sets. This is separate from vector
performance evaluation.

`vector/core.py` adapts the authenticated bucket construction to dense vectors.
It uses the same 256-bit projection as TrustMR, authenticates selected bucket
members and missing bucket boundaries, and recomputes cosine ranking over the
candidate union. The four configurations are `(bands, bits per band)` =
`(4,64)`, `(8,32)`, `(16,16)`, and `(32,8)`. The archived exact-match comparison
selected `(32,8)` from the common screening queries before holdout evaluation.

The implementation repairs collection-before-sorting, retains band identity,
and authenticates missing buckets with ordered boundary proofs. `radius=1`
RANGE expansion is a separate sensitivity experiment and is not substituted
for exact-match results.

## Measurement Boundaries

- Communication is the actual encoded result-plus-proof payload. Returned
  vectors already present in authenticated openings are not duplicated.
- Byte-to-verdict verification includes deserialization, authentication,
  reconstruction or replay, and final result comparisons.
- TrustMR parameter/security experiments in `run_compact_revision.py` time an
  already decoded response. Their object-to-verdict times are not the external
  comparison's byte-to-verdict times.
- V-GANN search includes its native graph traversal. Its full-scan proof
  assembly remains part of measured proof generation. Strict search and E2E
  columns are not directly interchangeable with different boundary definitions
  in other wrappers; use the explicitly matching boundaries for comparison.
- Image/text queries 0-99 are screening queries. The full-validation evaluation
  compares all three methods on 5,000 queries per modality and separately reports
  the 4,900 held-out queries at indices 100-4999. Latency uses indices 100-299
  with `timing_mode=single_worker`, repeated three times. Each query contributes
  its mean across repetitions; parallel-stage timings are excluded.
- All algorithms consume the same frozen corpus/query vectors and quantized
  cosine ground truth. The published COCO/query mapping is in
  `evaluation_metadata/`; the full-validation extension is recorded in
  `benchmarks/full_validation/query_ids.csv`. No original image or checkpoint is
  distributed here.

The core protocol and baseline source files retain the evaluated server
implementations. `prepare_benchmark_inputs.py` and `build_hcnng.py` provide
portable input preparation and dependency build entry points; they do not
modify retrieval or authentication algorithms.
