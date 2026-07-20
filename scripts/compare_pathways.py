from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Callable


def read_predictions(run_dir: Path, agent: str) -> list[dict]:
    path = run_dir / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing predictions: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("agent") == agent]
    if not selected:
        raise ValueError(f"{path} contains no predictions for agent={agent!r}")
    return selected


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_metrics(
    lalm: dict[tuple, dict],
    control: dict[tuple, dict],
    metric: Callable[[dict], float],
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if set(lalm) != set(control):
        missing_control = sorted(set(lalm) - set(control))[:5]
        missing_lalm = sorted(set(control) - set(lalm))[:5]
        raise ValueError(
            "paired runs contain different examples; "
            f"missing_control={missing_control}, missing_lalm={missing_lalm}"
        )
    keys = sorted(lalm)
    lalm_values = [metric(lalm[key]) for key in keys]
    control_values = [metric(control[key]) for key in keys]
    differences = [
        lalm_value - control_value
        for lalm_value, control_value in zip(lalm_values, control_values)
    ]
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        bootstrap.append(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    return {
        "examples": len(keys),
        "lalm": sum(lalm_values) / len(lalm_values),
        "control_score": sum(control_values) / len(control_values),
        "delta_lalm_minus_control": sum(differences) / len(differences),
        "ci95_low": percentile(bootstrap, 0.025),
        "ci95_high": percentile(bootstrap, 0.975),
        "paired_bootstrap_samples": samples,
    }


def keyed(rows: list[dict], fields: tuple[str, ...]) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate prediction key: {key}")
        result[key] = row
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired LALM comparisons against pathway ablations and RAG"
    )
    parser.add_argument("--synthetic-main", type=Path, required=True)
    parser.add_argument("--longmemeval-main", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/pathway_comparison.csv"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    synthetic_lalm_rows = read_predictions(args.synthetic_main, "lalm")
    synthetic_controls = {
        "pure_liquid": read_predictions(args.synthetic_main, "pure_liquid"),
        "lexical_only": read_predictions(args.synthetic_main, "lexical_only"),
        "rag": read_predictions(args.synthetic_main, "rag"),
    }
    synthetic_results = []
    horizons = sorted({int(row["horizon"]) for row in synthetic_lalm_rows})
    for horizon in [*horizons, None]:
        selected_lalm = [
            row for row in synthetic_lalm_rows
            if horizon is None or int(row["horizon"]) == horizon
        ]
        lalm_map = keyed(selected_lalm, ("seed", "example_id", "horizon"))
        for offset, (control_name, rows) in enumerate(synthetic_controls.items()):
            selected_control = [
                row for row in rows
                if horizon is None or int(row["horizon"]) == horizon
            ]
            metrics = paired_metrics(
                lalm_map,
                keyed(selected_control, ("seed", "example_id", "horizon")),
                lambda row: float(bool(row["correct"])),
                args.bootstrap_samples,
                args.seed + offset + (horizon or 0),
            )
            synthetic_results.append({
                "dataset": "synthetic",
                "scope": "overall" if horizon is None else str(horizon),
                "metric": "exact_value_accuracy",
                "control": control_name,
                **metrics,
            })

    real_lalm = keyed(
        read_predictions(args.longmemeval_main, "lalm"),
        ("example_id",),
    )
    real_controls = {
        "pure_liquid": keyed(
            read_predictions(args.longmemeval_main, "pure_liquid"),
            ("example_id",),
        ),
        "lexical_only": keyed(
            read_predictions(args.longmemeval_main, "lexical_only"),
            ("example_id",),
        ),
        "rag": keyed(
            read_predictions(args.longmemeval_main, "rag"),
            ("example_id",),
        ),
    }
    real_metrics = {
        "diagnostic_exact_match": lambda row: float(bool(row["correct"])),
        "diagnostic_token_f1": lambda row: float(row["token_f1_diagnostic"]),
    }
    real_results = []
    for control_offset, (control_name, control) in enumerate(real_controls.items()):
        for metric_offset, (metric_name, metric) in enumerate(real_metrics.items()):
            metrics = paired_metrics(
                real_lalm,
                control,
                metric,
                args.bootstrap_samples,
                args.seed + 10_000 + 100 * control_offset + metric_offset,
            )
            real_results.append({
                "dataset": "longmemeval_heldout_50_499",
                "scope": "overall",
                "metric": metric_name,
                "control": control_name,
                **metrics,
            })

    results = synthetic_results + real_results
    write_csv(args.output, results)
    run_dirs = {
        "synthetic_main": args.synthetic_main,
        "longmemeval_main": args.longmemeval_main,
    }
    source_hashes = {
        name: hashlib.sha256(
            (run_dir / "predictions.jsonl").read_bytes()
        ).hexdigest()
        for name, run_dir in run_dirs.items()
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps({
            "output_name": args.output.name,
            "bootstrap_seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "run_ids": {name: path.name for name, path in run_dirs.items()},
            "source_prediction_sha256": source_hashes,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
