from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


AGENT_LABELS = {
    "vanilla": "Vanilla",
    "window": "Window",
    "rag": "RAG",
    "pure_liquid": "Pure Liquid",
    "lexical_only": "Lexical-only",
    "lalm": "LALM",
    "lalm_zero_prefix": "Zero prefix",
    "lalm_random_prefix": "Random prefix",
    "lalm_permuted_prefix": "Permuted prefix",
}

PRIMARY_AGENTS = ["vanilla", "window", "rag", "pure_liquid", "lexical_only", "lalm"]
PREFIX_AGENTS = [
    "lexical_only",
    "lalm_zero_prefix",
    "lalm_random_prefix",
    "lalm_permuted_prefix",
    "lalm",
]
LOCOMO_TYPES = [
    ("category-1", "Multi-hop"),
    ("category-2", "Temporal"),
    ("category-3", "Open-domain"),
    ("category-4", "Single-hop"),
    ("category-5", "Adversarial"),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pct(value: str | float) -> float:
    return 100.0 * float(value)


def fmt_number(value: float) -> str:
    return f"{value:.2f}"


def fmt_pm(mean: float, std: float, bold: bool = False) -> str:
    body = f"{fmt_number(mean)}\\!\\pm\\!{fmt_number(std)}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    return f"${body}$"


def fmt_cell(value: float, bold: bool = False) -> str:
    body = fmt_number(value)
    if bold:
        body = f"\\textbf{{{body}}}"
    return body


def metric_map(rows: list[dict[str, str]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[field] for field in key_fields): row for row in rows}


def best_agents_by_column(
    rows_by_agent: dict[str, list[float]], agents: list[str], column_count: int
) -> list[str | None]:
    best = []
    for index in range(column_count):
        candidates = [
            (agent, rows_by_agent[agent][index])
            for agent in agents
            if agent in rows_by_agent
        ]
        best.append(max(candidates, key=lambda item: item[1])[0] if candidates else None)
    return best


def synthetic_rows(run_dir: Path, agents: list[str], horizons: list[str]) -> list[str]:
    rows = read_csv(run_dir / "metrics.csv")
    by_key = metric_map(rows, ("horizon", "agent"))
    means_by_agent = {
        agent: [
            pct(by_key[(horizon, agent)]["accuracy_mean"])
            for horizon in horizons
        ]
        for agent in agents
    }
    best = best_agents_by_column(means_by_agent, agents, len(horizons))
    output = []
    for agent in agents:
        cells = []
        for index, horizon in enumerate(horizons):
            row = by_key[(horizon, agent)]
            cells.append(
                fmt_pm(
                    pct(row["accuracy_mean"]),
                    pct(row["accuracy_std"]),
                    bold=best[index] == agent,
                )
            )
        output.append(f"{AGENT_LABELS[agent]} & " + " & ".join(cells) + r"\\")
    return output


def longmemeval_rows(run_dir: Path) -> list[str]:
    rows = [row for row in read_csv(run_dir / "metrics.csv") if row["agent"] in PRIMARY_AGENTS]
    best_f1 = max(rows, key=lambda row: float(row["diagnostic_token_f1"]))["agent"]
    best_exact = max(rows, key=lambda row: float(row["diagnostic_exact_match"]))["agent"]
    by_agent = {row["agent"]: row for row in rows}
    output = []
    for agent in PRIMARY_AGENTS:
        row = by_agent[agent]
        output.append(
            f"{AGENT_LABELS[agent]} & "
            f"{fmt_cell(pct(row['diagnostic_token_f1']), best_f1 == agent)} & "
            f"{fmt_cell(pct(row['diagnostic_exact_match']), best_exact == agent)}" + r"\\"
        )
    return output


def locomo_rows(run_dir: Path) -> list[str]:
    metrics = {
        row["agent"]: row
        for row in read_csv(run_dir / "metrics.csv")
        if row["agent"] in PRIMARY_AGENTS
    }
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in predictions:
        agent = row["agent"]
        question_type = row["question_type"]
        if agent in PRIMARY_AGENTS and question_type.startswith("category-"):
            scores[(agent, question_type)].append(float(row["locomo_official_qa_score"]))

    values_by_agent = {}
    for agent in PRIMARY_AGENTS:
        values_by_agent[agent] = [
            pct(metrics[agent]["locomo_official_qa_score"]),
            *[
                100.0 * statistics.mean(scores[(agent, question_type)])
                for question_type, _label in LOCOMO_TYPES
            ],
        ]
    best = best_agents_by_column(values_by_agent, PRIMARY_AGENTS, 6)
    output = []
    for agent in PRIMARY_AGENTS:
        cells = [
            fmt_cell(value, best[index] == agent)
            for index, value in enumerate(values_by_agent[agent])
        ]
        output.append(f"{AGENT_LABELS[agent]} & " + " & ".join(cells) + r"\\")
    return output


def efficiency_rows(run_dir: Path) -> list[str]:
    memory_rows = read_csv(run_dir / "memory.csv")
    latency_rows = read_csv(run_dir / "latency.csv")
    memory_by_agent: dict[str, list[int]] = defaultdict(list)
    latency_by_agent: dict[str, list[float]] = defaultdict(list)
    for row in memory_rows:
        if row["agent"] in PRIMARY_AGENTS:
            memory_by_agent[row["agent"]].append(int(float(row["bytes"])))
    for row in latency_rows:
        if row["agent"] in PRIMARY_AGENTS:
            latency_by_agent[row["agent"]].append(float(row["end_to_end_ms"]))

    output = []
    for agent in PRIMARY_AGENTS:
        memory_values = memory_by_agent[agent]
        mean_memory = round(statistics.mean(memory_values))
        memory_suffix = "fixed" if len(set(memory_values)) == 1 else "mean"
        memory_text = "0" if mean_memory == 0 else f"{mean_memory:,} {memory_suffix}"
        latency_text = fmt_number(statistics.mean(latency_by_agent[agent]))
        output.append(f"{AGENT_LABELS[agent]} & {memory_text} & {latency_text}" + r"\\")
    return output


def checkpoint_seed(run_dir: Path) -> int | None:
    manifest_path = run_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if "checkpoint_seed" in manifest:
        return int(manifest["checkpoint_seed"])
    manifests = manifest.get("manifests", [])
    seeds = {
        int(item["checkpoint_seed"])
        for item in manifests
        if item.get("checkpoint_seed") is not None
    }
    if len(seeds) == 1:
        return next(iter(seeds))
    return None


def validate_checkpoint_seeds(run_dirs: list[Path], expected_seed: int) -> None:
    mismatches = []
    for run_dir in run_dirs:
        actual_seed = checkpoint_seed(run_dir)
        if actual_seed is not None and actual_seed != expected_seed:
            mismatches.append(f"{run_dir} has checkpoint_seed={actual_seed}")
    if mismatches:
        joined = "\n  - ".join(mismatches)
        raise SystemExit(
            f"Run/checkpoint seed mismatch for requested training seed {expected_seed}:\n"
            f"  - {joined}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write seed-specific paper.tex replacement rows from completed result runs"
    )
    parser.add_argument("--synthetic-run", type=Path, required=True)
    parser.add_argument(
        "--prefix-1k-run",
        type=Path,
        help="Optional separate 1,000-turn sham-prefix control run",
    )
    parser.add_argument("--prefix-run", type=Path, required=True)
    parser.add_argument("--longmemeval-run", type=Path, required=True)
    parser.add_argument("--locomo-run", type=Path, required=True)
    parser.add_argument(
        "--training-seed",
        type=int,
        required=True,
        help="Training checkpoint seed for the supplied run directories",
    )
    parser.add_argument("--seed-label")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dirs = [
        args.synthetic_run,
        args.longmemeval_run,
        args.locomo_run,
    ]
    if args.prefix_1k_run is not None:
        run_dirs.append(args.prefix_1k_run)
    run_dirs.append(args.prefix_run)
    validate_checkpoint_seeds(run_dirs, args.training_seed)
    seed_label = args.seed_label or f"Training seed {args.training_seed}"
    output = args.output or Path(f"analysis/paper_seed{args.training_seed}_values.tex")

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated rows for paper.tex. Paste these into the matching seed blocks.",
        f"% {seed_label}",
        "",
        "% Synthetic horizon effect",
        *synthetic_rows(args.synthetic_run, PRIMARY_AGENTS, ["100", "500", "1000", "5000"]),
        "",
        *(
            [
                "% Sham-prefix control at 1K",
                *synthetic_rows(args.prefix_1k_run, PREFIX_AGENTS, ["1000"]),
                "",
            ]
            if args.prefix_1k_run is not None
            else []
        ),
        "% Sham-prefix control at 5K",
        *synthetic_rows(args.prefix_run, PREFIX_AGENTS, ["5000"]),
        "",
        "% LongMemEval local diagnostics",
        *longmemeval_rows(args.longmemeval_run),
        "",
        "% LoCoMo category-aware QA score",
        *locomo_rows(args.locomo_run),
        "",
        "% LongMemEval memory and latency",
        *efficiency_rows(args.longmemeval_run),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
