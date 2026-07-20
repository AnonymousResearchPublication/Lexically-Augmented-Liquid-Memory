from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def read_predictions(path: Path) -> list[dict]:
    predictions = path / "predictions.jsonl"
    if not predictions.is_file():
        raise FileNotFoundError(f"missing predictions.jsonl in {path}")
    return [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def key(row: dict) -> tuple:
    return (
        str(row.get("seed", "")),
        str(row.get("source_run_id", "")),
        str(row["example_id"]),
        int(row["horizon"]),
    )


def paired_delta(
    rows: list[dict],
    horizon: int,
    control: str,
    treatment: str = "lalm",
    bootstrap_samples: int = 10_000,
) -> dict:
    by_agent = {
        agent: {
            key(row): bool(row["correct"])
            for row in rows
            if int(row["horizon"]) == horizon and row["agent"] == agent
        }
        for agent in (control, treatment)
    }
    treatment_keys = set(by_agent[treatment])
    control_keys = set(by_agent[control])
    shared = sorted(treatment_keys & control_keys)
    if not shared:
        raise ValueError(f"no paired rows for horizon={horizon}, control={control}")
    differences = [
        float(by_agent[treatment][item]) - float(by_agent[control][item])
        for item in shared
    ]
    rng = random.Random(20260707 + horizon + len(control))
    boot = []
    for _ in range(bootstrap_samples):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    lower = boot[int(0.025 * (len(boot) - 1))]
    upper = boot[int(0.975 * (len(boot) - 1))]
    return {
        "horizon": horizon,
        "treatment": treatment,
        "control": control,
        "paired_examples": len(shared),
        "treatment_accuracy": sum(by_agent[treatment][item] for item in shared) / len(shared),
        "control_accuracy": sum(by_agent[control][item] for item in shared) / len(shared),
        "delta_treatment_minus_control": sum(differences) / len(differences),
        "ci95_low": lower,
        "ci95_high": upper,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired comparison for LALM sham-prefix controls"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--controls",
        nargs="+",
        default=[
            "lexical_only",
            "lalm_zero_prefix",
            "lalm_random_prefix",
            "lalm_permuted_prefix",
        ],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_predictions(args.run_dir)
    horizons = sorted({int(row["horizon"]) for row in rows if row["agent"] == "lalm"})
    comparisons = [
        paired_delta(rows, horizon, control)
        for horizon in horizons
        for control in args.controls
    ]
    output = args.output or args.run_dir / "prefix_control_comparison.csv"
    write_csv(output, comparisons)
    print(f"Wrote {output}")
    for row in comparisons:
        print(
            f"h={row['horizon']} lalm - {row['control']}: "
            f"{100 * row['delta_treatment_minus_control']:.2f} pp "
            f"[{100 * row['ci95_low']:.2f}, {100 * row['ci95_high']:.2f}] "
            f"n={row['paired_examples']}"
        )


if __name__ == "__main__":
    main()
