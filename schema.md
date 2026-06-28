# SciComm Trace Schema v1

SciComm uses two levels of data:

1. Raw PMPI events, one JSON object per line.
2. Postprocessed motif summaries.

The raw trace is intentionally simple so it can be emitted from C with low
overhead.

## Raw Event Fields

Common fields:

```json
{
  "schema": "scicomm.raw.v1",
  "event": "collective",
  "seq": 42,
  "rank": 7,
  "size": 128,
  "time_us": 12345.6,
  "host": "node001",
  "local_rank": 3,
  "phase": "unknown"
}
```

Important event types:

| Event | Meaning |
| --- | --- |
| `rank_info` | One record per rank at initialization. |
| `collective` | Blocking collective completion. |
| `request_start` | Nonblocking point-to-point or collective started. |
| `request_complete` | Nonblocking request completed by wait/test. |
| `p2p` | Blocking point-to-point completion. |
| `comm_create` | New communicator observed. |
| `comm_free` | Communicator freed. |
| `finalize` | Rank is finalizing MPI. |

Common operation-specific fields:

```json
{
  "op": "allreduce",
  "comm_id": 0,
  "comm_size": 128,
  "count": 1,
  "dtype": "float64",
  "bytes": 8,
  "reduction": "sum",
  "duration_us": 17.2,
  "buffer_location": "unknown"
}
```

Point-to-point fields:

```json
{
  "op": "isend",
  "peer": 12,
  "tag": 99,
  "bytes": 8192,
  "dtype": "float64"
}
```

All-to-all-v fields:

```json
{
  "op": "alltoallv",
  "send_bytes": 4096,
  "recv_bytes": 12288,
  "send_nonzero": 2,
  "recv_nonzero": 5,
  "send_max_bytes": 2048,
  "recv_max_bytes": 4096
}
```

Allgather-v fields:

```json
{
  "op": "allgatherv",
  "send_bytes": 64,
  "recv_bytes": 8192,
  "recv_nonzero": 7,
  "recv_max_bytes": 2048,
  "recv_min_nonzero_bytes": 16
}
```

## Motif Summary

The postprocessor emits motif objects like:

```json
{
  "schema": "scicomm.motif.v1",
  "motif": "scalar_allreduce",
  "op": "allreduce",
  "events": 327202,
  "total_bytes": 2944818,
  "avg_bytes": 9.0,
  "total_duration_us": 737000000.0
}
```

Supported motifs in the first version:

- `scalar_allreduce`
- `allreduce`
- `allgather`
- `sparse_allgather`
- `dense_alltoall`
- `irregular_alltoallv`
- `neighbor_exchange`
- `halo_exchange`
- `barrier_or_sync`
- `other_collective`

## Phase Labels

Without application changes, `phase` is usually `unknown`. There are two simple
ways to label a run:

1. Set `SCICOMM_PHASE` for a whole run.
2. Call `scicomm_set_phase("phase_name")` from an application or small shim.

The second option requires the application to be linked or loaded in a way that
can resolve the `scicomm_set_phase` symbol. It is optional.
