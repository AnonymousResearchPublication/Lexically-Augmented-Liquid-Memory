from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from one validated run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/tables"))
    args = parser.parse_args()
    with (args.run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("metrics.csv is empty; refusing to generate a table")
    columns = list(rows[0])
    # Synthetic exact-value output intentionally contains accuracy, not redundant P/R/F1.
    preferred = [
        column for column in
        ("horizon", "agent", "examples", "accuracy", "seeds", "accuracy_mean", "accuracy_std",
         "official_score", "diagnostic_token_f1", "diagnostic_exact_match",
         "locomo_official_qa_score")
        if column in columns
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.run_dir.name}_metrics.tex"
    lines = [
        r"\begin{tabular}{" + "l" * len(preferred) + "}",
        r"\toprule",
        " & ".join(escape(column) for column in preferred) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(escape(row[column]) for column in preferred) + r" \\" for row in rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {
        "artifact_name": output.name,
        "run_ids": [args.run_dir.name],
        "generation_script": "scripts/generate_tables.py",
        "dataset_hash": json.loads(
            (args.run_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        ).get("sha256"),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
