from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liquid_memory_agents.utils import create_run_directory, write_environment
from liquid_memory_agents.utils.config import load_config


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_one(config: dict, checkpoint: Path, config_dir: Path) -> Path:
    root = Path(config["run"]["output_root"])
    before = {path.resolve() for path in root.glob("*") if path.is_dir()}
    config_path = config_dir / f"seed_{config['run']['seed']}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_synthetic.py")),
        "--config", str(config_path),
        "--checkpoint", str(checkpoint),
    ]
    subprocess.run(command, check=True)
    created = [
        path for path in root.glob("*")
        if path.is_dir() and path.resolve() not in before and (path / "metrics.csv").exists()
    ]
    if len(created) != 1:
        raise RuntimeError(f"expected one new run for seed {config['run']['seed']}, found {created}")
    return created[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-seed learned-model evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 13, 17])
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=(
            "vanilla", "window", "rag",
            "pure_liquid", "lexical_only", "lalm",
            "lalm_zero_prefix", "lalm_random_prefix", "lalm_permuted_prefix",
        ),
        help="Override the primary agents for every child run",
    )
    args = parser.parse_args()
    base = load_config(args.config)
    if args.agents:
        base.setdefault("evaluation", {})["agents"] = args.agents
    child_runs: list[tuple[int, Path]] = []
    with tempfile.TemporaryDirectory(prefix="liquid-multiseed-") as temp:
        temp_dir = Path(temp)
        for index, seed in enumerate(args.seeds, 1):
            print(f"[multi-seed] seed {seed} ({index}/{len(args.seeds)})", flush=True)
            config = json.loads(json.dumps(base))
            config["run"]["seed"] = seed
            child_runs.append((seed, _run_one(config, args.checkpoint, temp_dir)))

    aggregate_dir = create_run_directory(base["run"]["output_root"], "multi_seed")
    aggregate_config = {
        **base,
        "run_kind": "aggregate",
        "aggregate_type": "multi_seed",
        "seeds": args.seeds,
        "source_run_ids": [path.name for _, path in child_runs],
        "checkpoint": str(args.checkpoint),
    }
    (aggregate_dir / "config.yaml").write_text(
        yaml.safe_dump(aggregate_config, sort_keys=True), encoding="utf-8"
    )
    write_environment(aggregate_dir)
    all_metrics, all_predictions = [], []
    for seed, run in child_runs:
        all_metrics.extend({**row, "seed": seed, "run_id": run.name} for row in _read_csv(run / "metrics.csv"))
        for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            all_predictions.append({**row, "seed": seed, "source_run_id": run.name})
    grouped = defaultdict(list)
    for row in all_metrics:
        grouped[(row["horizon"], row["agent"])].append(float(row["accuracy"]))
    aggregate = []
    import statistics
    for (horizon, agent), values in sorted(grouped.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        aggregate.append({
            "horizon": horizon,
            "agent": agent,
            "seeds": len(values),
            "accuracy_mean": statistics.mean(values),
            "accuracy_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    _write_csv(aggregate_dir / "metrics.csv", aggregate)
    _write_csv(aggregate_dir / "per_type.csv", [])
    _write_csv(aggregate_dir / "latency.csv", [])
    _write_csv(aggregate_dir / "memory.csv", [])
    (aggregate_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_predictions),
        encoding="utf-8",
    )
    (aggregate_dir / "errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_predictions if not row["correct"]),
        encoding="utf-8",
    )
    manifests = [
        json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
        for _, run in child_runs
    ]
    (aggregate_dir / "dataset_manifest.json").write_text(
        json.dumps({"source_runs": [run.name for _, run in child_runs], "manifests": manifests}, indent=2),
        encoding="utf-8",
    )
    print(f"Completed multi-seed aggregate: {aggregate_dir}")


if __name__ == "__main__":
    main()
