from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liquid_memory_agents.evaluation.error_analysis import error_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = [error_record(row) for row in rows if not row.get("correct")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors),
        encoding="utf-8",
    )
    counts = Counter(category for row in errors for category in row["categories"])
    print(json.dumps({"errors": len(errors), "categories": counts}, indent=2))


if __name__ == "__main__":
    main()
