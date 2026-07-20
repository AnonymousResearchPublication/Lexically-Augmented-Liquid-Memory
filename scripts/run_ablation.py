from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import tempfile
import math
import statistics
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liquid_memory_agents.utils import create_run_directory, write_environment
from liquid_memory_agents.utils.config import load_config


def key_for(configuration: dict) -> str:
    return "__".join(f"{key}={str(value).lower()}" for key, value in sorted(configuration.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-selected Liquid ablations with explicit trained checkpoints"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit ablation-grid YAML; no unpublished default is bundled.",
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        required=True,
        help="JSON object mapping canonical configuration keys to checkpoint paths",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--full-factorial",
        action="store_true",
        help="Run the complete Cartesian grid instead of the default one-factor-at-a-time study",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = json.loads(args.checkpoint_manifest.read_text(encoding="utf-8"))
    grid = config["grid"]
    fields = ("adaptive_tau", "trained", "correction_loss", "state_dim", "slots", "update")
    if args.full_factorial:
        combinations = [
            dict(zip(fields, values))
            for values in itertools.product(*(grid[field] for field in fields))
        ]
    else:
        primary = dict(config["primary"])
        combinations = [primary]
        for field in fields:
            for value in grid[field]:
                if value == primary[field]:
                    continue
                variant = dict(primary)
                variant[field] = value
                combinations.append(variant)
    if args.smoke:
        combinations = combinations[:2]
    missing = [key_for(item) for item in combinations if key_for(item) not in manifest]
    if missing:
        preview = "\n".join(missing[:10])
        raise SystemExit(
            f"Checkpoint manifest is missing {len(missing)} configurations. First entries:\n{preview}"
        )

    source_runs, result_rows = [], []
    with tempfile.TemporaryDirectory(prefix="liquid-ablation-") as temp:
        for index, ablation in enumerate(combinations, 1):
            print(f"[ablation] {index}/{len(combinations)} {key_for(ablation)}", flush=True)
            run_config = json.loads(json.dumps(config))
            run_config["run"]["seed"] = int(config.get("validation_seed", 2000))
            run_config["dataset"] = {"name": "synthetic", "turns": [100, 500, 1000, 5000]}
            run_config["evaluation"] = {"agents": ["lalm"]}
            run_config["liquid"].update({
                "state_dim": ablation["state_dim"],
                "slots": ablation["slots"],
                "adaptive_tau": ablation["adaptive_tau"],
                "update": ablation["update"],
            })
            run_config.setdefault("training", {})["correction_loss"] = ablation["correction_loss"]
            run_config["training"]["trained"] = ablation["trained"]
            config_path = Path(temp) / f"{index}.yaml"
            config_path.write_text(yaml.safe_dump(run_config, sort_keys=True), encoding="utf-8")
            root = Path(run_config["run"]["output_root"])
            before = {path.resolve() for path in root.glob("*") if path.is_dir()}
            command = [
                sys.executable, str(Path(__file__).with_name("run_synthetic.py")),
                "--config", str(config_path), "--checkpoint", str(manifest[key_for(ablation)]),
            ]
            if args.smoke:
                command.append("--smoke")
            if not ablation["trained"]:
                command.append("--allow-untrained")
            subprocess.run(command, check=True)
            created = [
                path for path in root.glob("*")
                if path.is_dir() and path.resolve() not in before and (path / "metrics.csv").exists()
            ]
            if len(created) != 1:
                raise RuntimeError(f"expected one new run, found {created}")
            source_runs.append(created[0])
            with (created[0] / "metrics.csv").open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["agent"] == "lalm":
                        result_rows.append({
                            **ablation, **row, "run_id": created[0].name,
                            "checkpoint": manifest[key_for(ablation)],
                            "persistent_bytes": (
                                int(ablation["state_dim"]) * int(ablation["slots"])
                                * (4 if run_config["liquid"].get("storage_dtype", "float32") == "float32" else 2)
                            ),
                        })

    output = create_run_directory(config["run"]["output_root"], "ablation")
    by_configuration = {}
    for row in result_rows:
        by_configuration.setdefault(key_for({field: row[field] for field in fields}), []).append(row)
    scores = {}
    for key, rows in by_configuration.items():
        endpoint = max(rows, key=lambda row: int(row["horizon"]))
        average = statistics.mean(float(row["accuracy"]) for row in rows)
        memory_bytes = int(endpoint["persistent_bytes"])
        scores[key] = {
            "average": average,
            "endpoint": float(endpoint["accuracy"]),
            "efficiency": average / math.log2(memory_bytes + 1),
        }
    ranks = {
        criterion: {
            key: rank
            for rank, (key, _) in enumerate(
                sorted(scores.items(), key=lambda item: item[1][criterion], reverse=True), 1
            )
        }
        for criterion in ("average", "endpoint", "efficiency")
    }
    for key, rows in by_configuration.items():
        for row in rows:
            row["average_horizon_rank"] = ranks["average"][key]
            row["endpoint_rank"] = ranks["endpoint"][key]
            row["memory_efficiency_rank"] = ranks["efficiency"][key]
    output_config = {
        **config, "run_kind": "aggregate", "aggregate_type": "ablation",
        "source_run_ids": [path.name for path in source_runs],
        "checkpoint_manifest": str(args.checkpoint_manifest),
    }
    (output / "config.yaml").write_text(
        yaml.safe_dump(output_config, sort_keys=True), encoding="utf-8"
    )
    write_environment(output)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    for name in ("per_type.csv", "latency.csv", "memory.csv", "predictions.jsonl", "errors.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    (output / "dataset_manifest.json").write_text(
        json.dumps({"source_runs": [path.name for path in source_runs]}, indent=2),
        encoding="utf-8",
    )
    print(f"Completed ablation aggregate: {output}")


if __name__ == "__main__":
    main()
