from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_directories(root: Path) -> set[Path]:
    return {
        path.resolve()
        for path in root.glob("*_longmemeval")
        if path.is_dir()
    }


def newest_created(root: Path, before: set[Path]) -> Path:
    created = [
        path.resolve()
        for path in root.glob("*_longmemeval")
        if path.is_dir()
        and path.resolve() not in before
        and (path / "metrics.csv").is_file()
    ]
    if len(created) != 1:
        raise RuntimeError(
            f"expected exactly one completed LongMemEval run, found {created}"
        )
    return created[0]


def read_metrics(run_dir: Path) -> dict[str, dict[str, str]]:
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        return {row["agent"]: row for row in csv.DictReader(handle)}


def invoke(command: list[str], results_root: Path) -> Path:
    before = run_directories(results_root)
    print("[development sweep]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return newest_created(results_root, before)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune LALM retrieval depth and prefix strength on LongMemEval "
            "development indices 0-49 only"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/longmemeval.yaml"),
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--prefix-scales",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2],
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if any(value < 1 for value in args.top_k):
        parser.error("--top-k values must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.prefix_scales):
        parser.error("--prefix-scales values must be between 0 and 1")

    results_root = PROJECT_ROOT / "results"
    base = [
        sys.executable,
        "scripts/run_real_benchmark.py",
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--start-index",
        "0",
        "--examples",
        "50",
    ]
    rag_run = invoke([*base, "--agents", "rag"], results_root)
    rag = read_metrics(rag_run)["rag"]
    rows = []
    for top_k in args.top_k:
        lexical_run = invoke(
            [
                *base,
                "--agents",
                "lexical_only",
                "--lexical-top-k",
                str(top_k),
            ],
            results_root,
        )
        lexical = read_metrics(lexical_run)["lexical_only"]
        for scale in args.prefix_scales:
            run = invoke(
                [
                    *base,
                    "--agents",
                    "lalm",
                    "--lexical-top-k",
                    str(top_k),
                    "--prefix-scale",
                    str(scale),
                ],
                results_root,
            )
            metrics = read_metrics(run)
            for agent in ("rag", "lexical_only", "lalm"):
                source = (
                    rag
                    if agent == "rag"
                    else lexical
                    if agent == "lexical_only"
                    else metrics[agent]
                )
                rows.append({
                    "top_k": top_k,
                    "prefix_scale": scale,
                    "agent": agent,
                    "examples": int(source["examples"]),
                    "diagnostic_token_f1": float(
                        source["diagnostic_token_f1"]
                    ),
                    "diagnostic_exact_match": float(
                        source["diagnostic_exact_match"]
                    ),
                    "run_dir": str(
                        rag_run
                        if agent == "rag"
                        else lexical_run
                        if agent == "lexical_only"
                        else run
                    ),
                })

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = results_root / f"{timestamp}_lalm_dev_sweep"
    output.mkdir(parents=False, exist_ok=False)
    with (output / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metadata.json").write_text(
        json.dumps({
            "scope": "development_only",
            "dataset_indices": "0-49",
            "checkpoint": str(args.checkpoint.resolve()),
            "config": str(args.config),
            "rag_run": str(rag_run),
            "top_k": args.top_k,
            "prefix_scales": args.prefix_scales,
        }, indent=2),
        encoding="utf-8",
    )

    rag_f1 = float(rag["diagnostic_token_f1"])
    lalm_rows = [row for row in rows if row["agent"] == "lalm"]
    for row in sorted(
        lalm_rows,
        key=lambda value: (
            value["diagnostic_token_f1"],
            value["diagnostic_exact_match"],
        ),
        reverse=True,
    ):
        delta = 100 * (row["diagnostic_token_f1"] - rag_f1)
        print(
            f"top_k={row['top_k']} scale={row['prefix_scale']}: "
            f"LALM token_F1={100 * row['diagnostic_token_f1']:.2f}% "
            f"exact={100 * row['diagnostic_exact_match']:.2f}% "
            f"delta_vs_RAG={delta:+.2f} points"
        )
    print(f"Development sweep complete: {output}")


if __name__ == "__main__":
    main()
