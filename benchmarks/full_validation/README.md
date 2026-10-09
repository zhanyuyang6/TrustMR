# Full COCO Validation Results

The benchmark uses 118,287 MSCOCO 2017 training images as the retrieval corpus
and 5,000 validation images with one text query per image as queries. The corpus
and query images are disjoint. TrustMR, V-GANN, and VSQ share the frozen ImageBind
vectors, query IDs, and exact top-20 reference results.

## Files

- `query_ids.csv`: all image/text query IDs, COCO IDs, and split labels.
- `summary.csv`: means for the full validation set, held-out set, and screening set.
- `paper_metrics.csv`: phase timings and candidate counts used in the paper.
- `bootstrap_intervals.csv`: query-level 95% bootstrap intervals.
- `paired_recall_differences.csv`: paired TrustMR-versus-baseline recall differences.
- `trustmr/`, `vgann/`, `vsq/`: compressed, unmodified per-query JSONL observations.
- `complete.json`: completion and response-validity checks.

Each `queries.jsonl.gz` contains 10,000 rows: 5,000 image queries and 5,000 text
queries. Each `timing.jsonl.gz` contains 1,200 rows: 200 queries per modality,
measured three times. All methods use the same query and timing IDs, and all
responses passed their respective verifiers.

## Measurement

The first 100 queries per modality are configuration-screening queries. The
remaining 4,900 per modality form the held-out set. These subsets are included
in, rather than additional to, the 5,000-query full-validation set.

Recall and response size use all queries in each stated subset. Response size
includes both results and proof, measured in MiB (2^20 bytes). Verification time
starts from received response bytes and includes deserialization and all checks.
Latency uses one query worker and 64 OpenBLAS threads. The mean of three
measurements is computed for each query before averaging across queries.
Bootstrap intervals use 5,000 resamples and seed 20261009; latency resampling uses
the 200 query means, not 600 independent observations.

Search, proof generation, serialization, and verification exclude embedding
generation, network transfer, consensus, and offline index construction.
V-GANN's query-ID start hash falls outside its search timer, so its strict search
and end-to-end columns are not directly comparable. The raw phase sum is retained
in the observations and flagged by `search_e2e_directly_comparable=false`.

The main configurations are TrustMR with `p=8`, `mmax=8`, 256-bit unweighted
Hamming signatures, `Kcoarse=1000`, and `K=20`; V-GANN with the fixed HCNNG index;
and VSQ with 32 exact-match bands of 8 bits. No time/source filtering is applied.
V-GANN and VSQ are the paper-principle implementations described in
[`docs/REPRODUCTION.md`](../../docs/REPRODUCTION.md).

The main protocol implementations and the full-validation runner are included
in this repository. Parameter sweeps, attacks, ablations, and limited-probe audit
cases are separate experiments with their own sample counts.
