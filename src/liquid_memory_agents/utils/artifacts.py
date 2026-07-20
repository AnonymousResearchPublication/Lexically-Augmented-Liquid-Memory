from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_RUN_FILES = {
    "config.yaml", "environment.json", "hardware.json", "dataset_manifest.json",
    "predictions.jsonl", "metrics.csv", "per_type.csv", "latency.csv",
    "memory.csv", "errors.jsonl",
}


def create_run_directory(root: str | Path, label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(root) / f"{stamp}_{label}"
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = Path(root) / f"{stamp}_{label}_{suffix}"
    destination.mkdir(parents=True)
    (destination / "logs").mkdir()
    (destination / "checkpoints").mkdir()
    return destination


def write_environment(run_dir: str | Path) -> None:
    import torch

    run_dir = Path(run_dir)
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": packages,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    hardware = {
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "gpus": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }
    (run_dir / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
