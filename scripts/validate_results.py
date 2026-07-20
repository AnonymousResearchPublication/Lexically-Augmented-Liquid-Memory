from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REQUIRED = {
    "config.yaml", "environment.json", "hardware.json", "dataset_manifest.json",
    "predictions.jsonl", "metrics.csv", "per_type.csv", "latency.csv",
    "memory.csv", "errors.jsonl",
}


def validate_run(path: Path) -> list[str]:
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8")) or {}
    if config.get("run_kind") == "training":
        required = {"config.yaml", "environment.json", "hardware.json", "dataset_manifest.json", "training_metrics.json"}
        problems = [f"missing {name}" for name in sorted(required) if not (path / name).is_file()]
        if not any((path / "checkpoints").glob("*.pt")):
            problems.append("missing checkpoint")
        return problems
    problems = [f"missing {name}" for name in sorted(REQUIRED) if not (path / name).is_file()]
    prediction_path = path / "predictions.jsonl"
    if prediction_path.is_file():
        for line_number, line in enumerate(prediction_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"predictions.jsonl:{line_number}: {exc}")
                continue
            for field in ("example_id", "agent", "prediction", "reference", "prompt"):
                if field not in row:
                    problems.append(f"predictions.jsonl:{line_number}: missing {field}")
            if (
                row.get("agent") == "pure_liquid"
                and row.get("memory_context")
            ):
                problems.append(
                    f"predictions.jsonl:{line_number}: pure Liquid contains textual memory context"
                )
            if row.get("official_score") is not None and not (path / "official_evaluator.json").exists():
                problems.append(
                    f"predictions.jsonl:{line_number}: official score lacks evaluator artifact"
                )
    memory_path = path / "memory.csv"
    if memory_path.is_file() and memory_path.stat().st_size:
        import csv
        with memory_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for agent_name in ("lalm", "pure_liquid", "lexical_only"):
            sizes = {
                row.get("bytes") for row in rows
                if row.get("agent") == agent_name
            }
            if len(sizes) > 1:
                problems.append(
                    f"{agent_name} persistent bytes vary across examples/horizons: "
                    f"{sorted(sizes)}"
                )
    llm = config.get("answer_llm", {})
    if config.get("run_kind") == "evaluation" and llm.get("provider") != "transformers_local":
        problems.append("primary evaluation does not use transformers_local answer LLM")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        help="Validate only these run directories instead of every directory under --results",
    )
    args = parser.parse_args()
    runs = (
        args.run_dir
        if args.run_dir
        else [
            path for path in args.results.iterdir()
            if path.is_dir() and (path / "config.yaml").exists()
        ]
    )
    if not runs:
        raise SystemExit("No immutable run directories found; no result is validated.")
    failed = False
    for run in runs:
        problems = validate_run(run)
        print(f"{run}: {'PASS' if not problems else 'FAIL'}")
        for problem in problems:
            print(f"  - {problem}")
        failed |= bool(problems)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
