# SciComm MPI Motif Recorder

SciComm is a standalone, metadata-only PMPI recorder for scientific MPI
communication motifs. It is intended for AMG2023, Quicksilver, miniAMR,
miniVite, and similar proxy applications.

The recorder is attached at runtime with `LD_PRELOAD`; the application does not
need to be rebuilt.

```bash
cd scicomm
make

SCICOMM_TRACE_DIR=traces \
LD_PRELOAD=$PWD/libscicomm_pmpi.so \
mpirun -n 128 ./miniAMR.x ...
```

For Flux-style launches, scope the preload to the MPI job:

```bash
flux run --env=SCICOMM_TRACE_DIR=traces \
         --env=LD_PRELOAD=/path/to/libscicomm_pmpi.so \
         ...
```

## What It Records

The first version records descriptors only:

- rank, communicator id, communicator size;
- host and local rank placement;
- MPI operation name;
- count, datatype, and byte counts;
- peer rank and tag for point-to-point calls;
- per-rank aggregate counts for `Allgather(v)` and `Alltoall(v)`;
- nonblocking request start/completion timing;
- optional callsite return address;
- optional user phase label.

It does not record payloads.

Each rank writes one JSONL shard:

```text
${SCICOMM_TRACE_DIR:-.}/${SCICOMM_TRACE_PREFIX:-scicomm}.rank000123.jsonl
```

## Useful Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCICOMM_TRACE_DIR` | `.` | Output directory. Created if missing. |
| `SCICOMM_TRACE_PREFIX` | `scicomm` | Output filename prefix. |
| `SCICOMM_PHASE` | `unknown` | Static phase label for all events. |
| `SCICOMM_CALLSITE` | `0` | Set to `1` to record a return-address callsite. |
| `SCICOMM_FLUSH_EVERY` | `0` | Flush every N events. `0` means flush at finalize. |
| `SCICOMM_BUFFER_LOCATION` | `unknown` | Force `host`, `device`, `managed`, or `unknown`. |

`SCICOMM_BUFFER_LOCATION` is a coarse override. For AMG GPU residency studies, a
future build should add HIP pointer classification with `hipPointerGetAttributes`.
This initial implementation intentionally avoids a ROCm/CUDA build dependency.

## Postprocess

Convert raw events into motif summaries:

```bash
python3 scicomm_postprocess.py traces --out traces_summary
```

Outputs:

- `events_by_op.csv`: aggregate event counts, bytes, and time by raw operation.
- `rank_peer_edges.csv`: per-rank point-to-point peer graph summary.
- `motifs.json`: inferred high-level communication motifs.

The motif inference is intentionally conservative. It identifies scalar
reductions, sparse allgather, irregular all-to-all-v, dense all-to-all,
barriers, halo exchange, and repeated neighbor-like point-to-point exchange.
Application phase labeling and stronger callsite attribution can refine this
later without changing the raw trace format.

## Runtime Size

This first implementation is small enough to audit:

- C PMPI recorder: about 1.3k lines.
- Python postprocessor: about 300 lines.
- Schema/docs/tests: small.

It is not production-perfect. It is meant to produce useful scientific
communication metadata quickly without changing the profiled applications.
