#!/usr/bin/env python3
"""Postprocess SciComm raw PMPI JSONL traces into communication motifs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import glob
import json
from pathlib import Path
from typing import Any


@dataclass
class OpStats:
    events: int = 0
    total_bytes: int = 0
    total_send_bytes: int = 0
    total_recv_bytes: int = 0
    total_duration_us: float = 0.0
    max_bytes: int = 0

    def add(self, event: dict[str, Any]) -> None:
        self.events += 1
        bytes_value = int(event.get("bytes") or 0)
        send_bytes = int(event.get("send_bytes") or 0)
        recv_bytes = int(event.get("recv_bytes") or 0)
        self.total_bytes += bytes_value
        self.total_send_bytes += send_bytes
        self.total_recv_bytes += recv_bytes
        self.total_duration_us += float(
            event.get("duration_us")
            or event.get("elapsed_us")
            or event.get("wait_duration_us")
            or 0.0
        )
        self.max_bytes = max(self.max_bytes, bytes_value, send_bytes, recv_bytes)


@dataclass
class EdgeStats:
    events: int = 0
    bytes: int = 0
    sends: int = 0
    recvs: int = 0

    def add(self, op: str, bytes_value: int) -> None:
        self.events += 1
        self.bytes += bytes_value
        if op in {"send", "isend"}:
            self.sends += 1
        elif op in {"recv", "irecv"}:
            self.recvs += 1


@dataclass
class TraceSummary:
    op_stats: dict[str, OpStats] = field(default_factory=dict)
    edge_stats: dict[tuple[int, int, int], EdgeStats] = field(default_factory=dict)
    scalar_allreduce_events: list[dict[str, Any]] = field(default_factory=list)
    allreduce_events: list[dict[str, Any]] = field(default_factory=list)
    allgather_events: list[dict[str, Any]] = field(default_factory=list)
    allgatherv_events: list[dict[str, Any]] = field(default_factory=list)
    alltoallv_events: list[dict[str, Any]] = field(default_factory=list)
    alltoall_events: list[dict[str, Any]] = field(default_factory=list)
    barrier_events: list[dict[str, Any]] = field(default_factory=list)
    ranks: set[int] = field(default_factory=set)
    hosts: set[str] = field(default_factory=set)


def expand_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
            continue
        matches = [Path(match) for match in glob.glob(item)]
        if matches:
            files.extend(sorted(matches))
        else:
            files.append(path)
    return [path for path in files if path.exists()]


def load_events(files: list[Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                event["_file"] = str(path)
                event["_lineno"] = lineno
                events.append(event)
    return events


def event_op(event: dict[str, Any]) -> str:
    return str(event.get("op") or event.get("event") or "unknown")


def event_bytes(event: dict[str, Any]) -> int:
    if event.get("bytes") is not None:
        return int(event.get("bytes") or 0)
    send_bytes = int(event.get("send_bytes") or 0)
    recv_bytes = int(event.get("recv_bytes") or 0)
    return max(send_bytes, recv_bytes)


def summarize(events: list[dict[str, Any]], scalar_threshold: int, alltoallv_skew: float) -> TraceSummary:
    summary = TraceSummary()

    for event in events:
        event_kind = str(event.get("event") or "unknown")
        rank = int(event.get("rank", -1))
        if rank >= 0:
            summary.ranks.add(rank)
        host = event.get("host")
        if host:
            summary.hosts.add(str(host))

        op = event_op(event)
        stats = summary.op_stats.setdefault(f"{event_kind}:{op}", OpStats())
        stats.add(event)

        # Nonblocking start records are useful for timelines, but motif summaries
        # should count the completed communication once.
        if event_kind == "request_start":
            continue

        if op in {"allreduce", "iallreduce"}:
            if event_bytes(event) <= scalar_threshold:
                summary.scalar_allreduce_events.append(event)
            else:
                summary.allreduce_events.append(event)
        elif op in {"allgather", "iallgather"}:
            summary.allgather_events.append(event)
        elif op in {"allgatherv", "iallgatherv"}:
            summary.allgatherv_events.append(event)
        elif op in {"alltoallv", "ialltoallv", "neighbor_alltoallv"}:
            summary.alltoallv_events.append(event)
        elif op in {"alltoall", "ialltoall"}:
            summary.alltoall_events.append(event)
        elif op == "barrier":
            summary.barrier_events.append(event)

        if op in {"send", "recv", "isend", "irecv"}:
            peer = int(event.get("peer", -1))
            comm_id = int(event.get("comm_id", -1))
            if rank >= 0 and peer >= 0:
                key = (comm_id, rank, peer)
                summary.edge_stats.setdefault(key, EdgeStats()).add(op, event_bytes(event))

    return summary


def motif_from_events(
    motif: str,
    op: str,
    events: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_bytes = sum(event_bytes(event) for event in events)
    total_duration = sum(
        float(event.get("duration_us") or event.get("elapsed_us") or event.get("wait_duration_us") or 0.0)
        for event in events
    )
    payload = {
        "schema": "scicomm.motif.v1",
        "motif": motif,
        "op": op,
        "events": len(events),
        "total_bytes": total_bytes,
        "avg_bytes": total_bytes / len(events) if events else 0.0,
        "total_duration_us": total_duration,
    }
    if extra:
        payload.update(extra)
    return payload


def alltoallv_extra(events: list[dict[str, Any]], skew_threshold: float) -> dict[str, Any]:
    irregular = 0
    sparse = 0
    for event in events:
        comm_size = int(event.get("comm_size") or 0)
        send_nonzero = int(event.get("send_nonzero") or 0)
        recv_nonzero = int(event.get("recv_nonzero") or 0)
        send_total = int(event.get("send_bytes") or 0)
        recv_total = int(event.get("recv_bytes") or 0)
        send_max = int(event.get("send_max_bytes") or 0)
        recv_max = int(event.get("recv_max_bytes") or 0)
        if comm_size and (send_nonzero < comm_size or recv_nonzero < comm_size):
            sparse += 1
        send_mean = send_total / send_nonzero if send_nonzero else 0.0
        recv_mean = recv_total / recv_nonzero if recv_nonzero else 0.0
        if (send_mean and send_max / send_mean >= skew_threshold) or (
            recv_mean and recv_max / recv_mean >= skew_threshold
        ):
            irregular += 1
    return {
        "irregular_events": irregular,
        "sparse_events": sparse,
        "irregular_fraction": irregular / len(events) if events else 0.0,
        "sparse_fraction": sparse / len(events) if events else 0.0,
    }


def allgatherv_extra(events: list[dict[str, Any]], skew_threshold: float) -> dict[str, Any]:
    irregular = 0
    sparse = 0
    for event in events:
        comm_size = int(event.get("comm_size") or 0)
        recv_nonzero = int(event.get("recv_nonzero") or 0)
        recv_total = int(event.get("recv_bytes") or 0)
        recv_max = int(event.get("recv_max_bytes") or 0)
        if comm_size and recv_nonzero < comm_size:
            sparse += 1
        recv_mean = recv_total / recv_nonzero if recv_nonzero else 0.0
        if recv_mean and recv_max / recv_mean >= skew_threshold:
            irregular += 1
    return {
        "irregular_events": irregular,
        "sparse_events": sparse,
        "irregular_fraction": irregular / len(events) if events else 0.0,
        "sparse_fraction": sparse / len(events) if events else 0.0,
    }


def neighbor_exchange_motif(summary: TraceSummary) -> dict[str, Any] | None:
    if not summary.edge_stats:
        return None

    rank_to_peers: dict[tuple[int, int], set[int]] = {}
    total_events = 0
    total_bytes = 0
    bidirectional_edges = 0
    for (comm_id, rank, peer), edge in summary.edge_stats.items():
        rank_to_peers.setdefault((comm_id, rank), set()).add(peer)
        total_events += edge.events
        total_bytes += edge.bytes
        if edge.sends and edge.recvs:
            bidirectional_edges += 1

    degrees = [len(peers) for peers in rank_to_peers.values()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
    max_degree = max(degrees) if degrees else 0

    motif = "neighbor_exchange"
    if max_degree <= 26 and bidirectional_edges:
        motif = "halo_exchange"

    return {
        "schema": "scicomm.motif.v1",
        "motif": motif,
        "op": "p2p",
        "events": total_events,
        "total_bytes": total_bytes,
        "avg_bytes": total_bytes / total_events if total_events else 0.0,
        "rank_peer_sets": len(rank_to_peers),
        "avg_peer_degree": avg_degree,
        "max_peer_degree": max_degree,
        "bidirectional_rank_peer_edges": bidirectional_edges,
    }


def build_motifs(summary: TraceSummary, alltoallv_skew: float) -> list[dict[str, Any]]:
    motifs: list[dict[str, Any]] = []
    if summary.scalar_allreduce_events:
        motifs.append(motif_from_events("scalar_allreduce", "allreduce", summary.scalar_allreduce_events))
    if summary.allreduce_events:
        motifs.append(motif_from_events("allreduce", "allreduce", summary.allreduce_events))
    if summary.allgather_events:
        motifs.append(motif_from_events("allgather", "allgather", summary.allgather_events))
    if summary.allgatherv_events:
        extra = allgatherv_extra(summary.allgatherv_events, alltoallv_skew)
        motif_name = "sparse_allgather" if extra["sparse_events"] or extra["irregular_events"] else "allgather"
        motifs.append(
            motif_from_events(
                motif_name,
                "allgatherv",
                summary.allgatherv_events,
                extra,
            )
        )
    if summary.alltoall_events:
        motifs.append(motif_from_events("dense_alltoall", "alltoall", summary.alltoall_events))
    if summary.alltoallv_events:
        motifs.append(
            motif_from_events(
                "irregular_alltoallv",
                "alltoallv",
                summary.alltoallv_events,
                alltoallv_extra(summary.alltoallv_events, alltoallv_skew),
            )
        )
    if summary.barrier_events:
        motifs.append(motif_from_events("barrier_or_sync", "barrier", summary.barrier_events))
    neighbor = neighbor_exchange_motif(summary)
    if neighbor:
        motifs.append(neighbor)
    return motifs


def write_op_csv(path: Path, summary: TraceSummary) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "op",
                "events",
                "total_bytes",
                "total_send_bytes",
                "total_recv_bytes",
                "max_bytes",
                "total_duration_us",
            ]
        )
        for op, stats in sorted(summary.op_stats.items()):
            writer.writerow(
                [
                    op,
                    stats.events,
                    stats.total_bytes,
                    stats.total_send_bytes,
                    stats.total_recv_bytes,
                    stats.max_bytes,
                    f"{stats.total_duration_us:.3f}",
                ]
            )


def write_edges_csv(path: Path, summary: TraceSummary) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comm_id", "rank", "peer", "events", "bytes", "sends", "recvs"])
        for (comm_id, rank, peer), stats in sorted(summary.edge_stats.items()):
            writer.writerow([comm_id, rank, peer, stats.events, stats.bytes, stats.sends, stats.recvs])


def write_motifs(path: Path, motifs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(motifs, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess SciComm PMPI JSONL traces")
    parser.add_argument("inputs", nargs="+", help="Trace JSONL files, globs, or directories")
    parser.add_argument("--out", default="scicomm_summary", help="Output directory")
    parser.add_argument("--scalar-threshold", type=int, default=64, help="Bytes threshold for scalar allreduce")
    parser.add_argument(
        "--alltoallv-skew",
        type=float,
        default=2.0,
        help="Max/mean ratio marking vector-collective skew",
    )
    args = parser.parse_args()

    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("No SciComm JSONL files found")

    events = load_events(files)
    summary = summarize(events, args.scalar_threshold, args.alltoallv_skew)
    motifs = build_motifs(summary, args.alltoallv_skew)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_op_csv(out_dir / "events_by_op.csv", summary)
    write_edges_csv(out_dir / "rank_peer_edges.csv", summary)
    write_motifs(out_dir / "motifs.json", motifs)

    print(f"read_events={len(events)} files={len(files)} ranks={len(summary.ranks)} hosts={len(summary.hosts)}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
