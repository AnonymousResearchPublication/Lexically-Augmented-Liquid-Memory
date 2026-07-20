from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invoke the pinned upstream LongMemEval QA evaluator"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt-4o")
    args = parser.parse_args()
    run_dir, evaluator, dataset = (
        args.run_dir.resolve(),
        args.evaluator.resolve(),
        args.dataset.resolve(),
    )
    if not evaluator.is_file() or not dataset.is_file():
        raise SystemExit("Evaluator or dataset path does not exist")
    inputs = sorted(run_dir.glob("official_input_*.jsonl"))
    if not inputs:
        raise SystemExit(f"No official_input_*.jsonl files in {run_dir}")
    evaluator_repo = evaluator.parents[2]
    commit = subprocess.run(
        ["git", "-C", str(evaluator_repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    records = []
    for index, hypothesis in enumerate(inputs, 1):
        command = [
            sys.executable,
            str(evaluator),
            args.judge_model,
            str(hypothesis.resolve()),
            str(dataset),
        ]
        print(f"[official evaluator] {index}/{len(inputs)} {hypothesis.name}", flush=True)
        result = subprocess.run(
            command,
            cwd=evaluator.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        records.append({
            "hypothesis": hypothesis.name,
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        })
        if result.returncode:
            (run_dir / "official_evaluator.json").write_text(
                json.dumps({
                    "status": "failed",
                    "judge_model": args.judge_model,
                    "evaluator": str(evaluator),
                    "evaluator_commit": commit,
                    "dataset": str(dataset),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "records": records,
                }, indent=2),
                encoding="utf-8",
            )
            raise SystemExit(f"Official evaluator failed for {hypothesis.name}")
    artifact = {
        "status": "completed",
        "judge_model": args.judge_model,
        "evaluator": str(evaluator),
        "evaluator_commit": commit,
        "dataset": str(dataset),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    (run_dir / "official_evaluator.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    print(f"Official evaluator completed: {run_dir / 'official_evaluator.json'}")


if __name__ == "__main__":
    main()
