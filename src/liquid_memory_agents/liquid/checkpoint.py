from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def dataset_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    modules: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    config: dict[str, Any],
    validation_metrics: dict[str, float],
    seed: int,
    dataset_hashes: dict[str, str],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "epoch": epoch,
            "modules": {name: module.state_dict() for name, module in modules.items()},
            "optimizer": optimizer.state_dict(),
            "config": config,
            "validation_metrics": validation_metrics,
            "seed": seed,
            "dataset_hashes": dataset_hashes,
        },
        destination,
    )
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "validation_metrics": validation_metrics,
                "seed": seed,
                "dataset_hashes": dataset_hashes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return destination


def load_checkpoint(path: str | Path, modules, optimizer=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    for name, module in modules.items():
        module.load_state_dict(payload["modules"][name])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
