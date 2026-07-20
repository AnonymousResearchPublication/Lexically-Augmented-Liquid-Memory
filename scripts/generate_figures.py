from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liquid_memory_agents.datasets.synthetic import generate_examples
from liquid_memory_agents.utils.config import load_config


AGENT_LABELS = {
    "vanilla": "Vanilla",
    "window": "Window",
    "rag": "Growing-history RAG",
    "pure_liquid": "Pure Liquid",
    "lexical_only": "Lexical-only",
    "lalm": "LALM",
}

AGENT_ORDER = ["lalm", "lexical_only", "rag", "pure_liquid", "window", "vanilla"]
AGENT_STYLES = {
    "lalm": {"color": "#1f77b4", "marker": "o", "linewidth": 2.4},
    "lexical_only": {"color": "#ff7f0e", "marker": "s", "linewidth": 2.0},
    "rag": {"color": "#2ca02c", "marker": "^", "linewidth": 2.0},
    "pure_liquid": {"color": "#9467bd", "marker": "D", "linewidth": 1.5, "alpha": 0.8},
    "window": {"color": "#8c564b", "marker": "v", "linewidth": 1.5, "alpha": 0.8},
    "vanilla": {"color": "#7f7f7f", "marker": "x", "linewidth": 1.2, "alpha": 0.8},
}

LALM_BYTES = 8 * 256 * 4 + 512 * (384 * 4 + 1024)


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def synthetic_metric(row: dict[str, str]) -> tuple[float, float]:
    if "accuracy_mean" in row:
        return 100.0 * float(row["accuracy_mean"]), 100.0 * float(row["accuracy_std"])
    return 100.0 * float(row["accuracy"]), 0.0


def infer_examples_by_horizon(config: dict) -> dict[int, int]:
    dataset = config.get("dataset", {})
    configured = dataset.get("examples_by_horizon", {})
    default = int(dataset.get("examples", 100))
    horizons = [int(value) for value in dataset.get("turns", [])]
    return {
        horizon: int(
            configured.get(str(horizon), configured.get(horizon, default))
        )
        for horizon in horizons
    }


def rag_payload_bytes(horizon: int, count: int, seed: int) -> float:
    examples = generate_examples(count, horizon, seed)
    payloads = []
    for example in examples:
        total = 0
        for turn, role, timestamp in zip(
            example.turns, example.roles, example.timestamps
        ):
            formatted = f"[{timestamp or 'unknown'}] {role}: {turn}"
            total += 384 * 4
            total += len(formatted.encode("utf-8"))
            total += 8
        payloads.append(total)
    return sum(payloads) / max(1, len(payloads))


def plot_synthetic_crossover(metrics: list[dict[str, str]], output: Path) -> None:
    rows = [
        row for row in metrics
        if row.get("agent") in AGENT_LABELS and row.get("horizon")
    ]
    if not rows:
        raise SystemExit("No synthetic rows with horizon/agent metrics found")
    horizons = sorted({int(row["horizon"]) for row in rows})
    by_agent = {(row["agent"], int(row["horizon"])): row for row in rows}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for agent in AGENT_ORDER:
        values, errors, xs = [], [], []
        for horizon in horizons:
            row = by_agent.get((agent, horizon))
            if row is None:
                continue
            value, error = synthetic_metric(row)
            xs.append(horizon)
            values.append(value)
            errors.append(error)
        if not xs:
            continue
        style = AGENT_STYLES[agent]
        ax.errorbar(
            xs,
            values,
            yerr=errors,
            label=AGENT_LABELS[agent],
            capsize=3,
            markersize=5,
            **style,
        )

    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels(["100", "500", "1K", "5K"])
    ax.set_xlabel("Conversation horizon (turns)")
    ax.set_ylabel("Exact-value accuracy (%)")
    ax.set_ylim(-2, 85)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("Synthetic long-horizon crossover")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_memory_scaling(config: dict, metrics: list[dict[str, str]], output: Path) -> None:
    horizons = sorted({
        int(row["horizon"]) for row in metrics
        if row.get("horizon") and row.get("agent") == "lalm"
    })
    if not horizons:
        horizons = [100, 500, 1000, 5000]
    counts = infer_examples_by_horizon(config)
    seeds = [int(seed) for seed in config.get("seeds", config.get("dataset", {}).get("test_seeds", [11, 13, 17]))]
    rag_values = []
    for horizon in horizons:
        per_seed = [
            rag_payload_bytes(horizon, counts.get(horizon, 50), seed)
            for seed in seeds
        ]
        rag_values.append(sum(per_seed) / len(per_seed))
    lalm_values = [LALM_BYTES for _ in horizons]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(horizons, rag_values, marker="o", linewidth=2.2, label="Growing-history RAG")
    ax.plot(horizons, lalm_values, marker="s", linewidth=2.2, label="LALM")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels(["100", "500", "1K", "5K"])
    ax.set_xlabel("Conversation horizon (turns)")
    ax.set_ylabel("Logical user-memory payload (bytes)")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Logical memory scaling")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from regenerated evaluation metrics"
    )
    parser.add_argument("--synthetic-run", "--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Figures"))
    args = parser.parse_args()

    metrics_path = args.synthetic_run / "metrics.csv"
    config_path = args.synthetic_run / "config.yaml"
    if not metrics_path.is_file():
        raise SystemExit(f"Missing metrics file: {metrics_path}")
    if not config_path.is_file():
        raise SystemExit(f"Missing run config: {config_path}")

    metrics = read_metrics(metrics_path)
    config = load_config(config_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crossover = args.output_dir / "synthetic_crossover.png"
    memory = args.output_dir / "memory_scaling.png"
    plot_synthetic_crossover(metrics, crossover)
    plot_memory_scaling(config, metrics, memory)

    metadata = {
        "figures": [crossover.name, memory.name],
        "synthetic_run_id": args.synthetic_run.name,
        "generation_script": "scripts/generate_figures.py",
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {crossover}")
    print(f"Wrote {memory}")


if __name__ == "__main__":
    main()
