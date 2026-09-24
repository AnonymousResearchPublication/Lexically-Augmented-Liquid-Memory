from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def conversation_id(example_id: str) -> str:
    return example_id.rsplit("-qa-", 1)[0]


def agent_scores(run_dir: Path, agent: str) -> dict[str, float]:
    rows = read_jsonl(run_dir / "predictions.jsonl")
    return {
        row["example_id"]: float(row["locomo_official_qa_score"])
        for row in rows
        if row["agent"] == agent
    }


def clustered_difference(
    candidate: dict[str, float],
    comparator: dict[str, float],
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    common = sorted(set(candidate) & set(comparator))
    if not common:
        raise ValueError("candidate and comparator have no shared examples")
    clusters: dict[str, list[str]] = defaultdict(list)
    for example_id in common:
        clusters[conversation_id(example_id)].append(example_id)
    cluster_names = sorted(clusters)
    differences = [candidate[key] - comparator[key] for key in common]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sampled_ids = [
            example_id
            for _ in cluster_names
            for example_id in clusters[rng.choice(cluster_names)]
        ]
        draws.append(statistics.mean(candidate[key] - comparator[key] for key in sampled_ids))
    return {
        "questions": len(common),
        "conversations": len(cluster_names),
        "difference": statistics.mean(differences),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
        "bootstrap_samples": samples,
    }


def latency_summary(run_dir: Path) -> list[dict[str, float | int | str]]:
    with (run_dir / "latency.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["agent"]].append(row)
    output = []
    for agent, selected in sorted(grouped.items()):
        result: dict[str, float | int | str] = {"agent": agent, "examples": len(selected)}
        for field in ("update_ms", "read_ms", "generation_ms", "end_to_end_ms"):
            values = [float(row[field]) for row in selected]
            result[f"{field}_mean"] = statistics.mean(values)
            result[f"{field}_median"] = statistics.median(values)
            result[f"{field}_p95"] = percentile(values, 0.95)
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera-ready clustered LoCoMo uncertainty and runtime summaries."
    )
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--candidate-agent", default="lalm")
    parser.add_argument("--comparators", nargs="+", default=["rag", "lexical_only"])
    parser.add_argument("--latency-run", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/camera_ready"))
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    candidate = agent_scores(args.candidate_run, args.candidate_agent)
    comparisons = []
    for comparator_name in args.comparators:
        result = clustered_difference(
            candidate,
            agent_scores(args.baseline_run, comparator_name),
            args.bootstrap_samples,
            args.seed,
        )
        comparisons.append({
            "candidate": args.candidate_agent,
            "comparator": comparator_name,
            **result,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "locomo_cluster_bootstrap.csv", comparisons)

    if args.latency_run is not None:
        write_csv(
            args.output_dir / "latency_summary.csv",
            latency_summary(args.latency_run),
        )
    print(f"Wrote camera-ready analyses to {args.output_dir}")


if __name__ == "__main__":
    main()
