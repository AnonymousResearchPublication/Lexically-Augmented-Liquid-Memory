from __future__ import annotations

import argparse
import platform
import sys
import importlib
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liquid_memory_agents.datasets.longmemeval import load_longmemeval
from liquid_memory_agents.utils.config import load_config


EXPECTED_AGENTS = {
    "vanilla", "window", "rag",
    "pure_liquid", "lexical_only", "lalm",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a fresh experiment environment")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="?",
        metavar="CHECKPOINT",
        help="Path to a trained checkpoint to validate",
    )
    parser.add_argument("--require-longmemeval", action="store_true")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless a CUDA-enabled Torch build and GPU are available",
    )
    args = parser.parse_args()

    if "--checkpoint" in sys.argv and args.checkpoint is None:
        raise SystemExit(
            "--checkpoint needs a path. In PowerShell, set $CKPT first, then run "
            "preflight with --checkpoint $CKPT or pass a full checkpoint path."
        )

    base = load_config("configs/base.yaml")
    synthetic = load_config("configs/synthetic.yaml")
    longmem = load_config("configs/longmemeval.yaml")
    configured = set(synthetic["evaluation"]["agents"])
    if configured != EXPECTED_AGENTS:
        raise SystemExit(
            f"Synthetic publication conditions mismatch: {sorted(configured)}"
        )
    if set(longmem["evaluation"]["agents"]) != EXPECTED_AGENTS:
        raise SystemExit("LongMemEval publication conditions do not include all publication agents")
    dataset = Path(longmem["dataset"]["input"])
    if args.require_longmemeval:
        examples = load_longmemeval(dataset)
        if not examples:
            raise SystemExit("LongMemEval dataset is empty")
        print(f"LongMemEval: {len(examples)} examples at {dataset}")
    elif not dataset.exists():
        print(f"LongMemEval: not downloaded yet ({dataset})")
    if args.checkpoint:
        if not args.checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("config", {}).get("initialization_only"):
            raise SystemExit("Checkpoint is initialization-only, not trained")
        required = {"cell", "reader", "adapter"}
        if set(payload.get("modules", {})) != required:
            raise SystemExit("Checkpoint does not contain cell, reader, and adapter")
        checkpoint_liquid = payload.get("config", {}).get("liquid", {})
        for key in ("state_dim", "slots", "prefix_tokens", "lexical_bytes"):
            if checkpoint_liquid.get(key) != base["liquid"].get(key):
                raise SystemExit(
                    f"Checkpoint liquid.{key}={checkpoint_liquid.get(key)!r} does not "
                    f"match config value {base['liquid'].get(key)!r}; retrain it."
                )
        print(
            f"Checkpoint: epoch={payload.get('epoch')} "
            f"validation={payload.get('validation_metrics')}"
        )
        print(
            "Checkpoint architecture: "
            f"prefix_tokens={checkpoint_liquid.get('prefix_tokens')}, "
            f"lexical_bytes={checkpoint_liquid.get('lexical_bytes')}"
        )
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA preflight failed: this environment has a CPU-only or unusable "
            "Torch installation. Reinstall requirements-cu128.txt."
        )
    try:
        importlib.import_module("torchaudio")
        print("Torchaudio binary: PASS")
    except (ImportError, OSError) as error:
        raise SystemExit(f"Torchaudio binary check failed: {error}") from error
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            print(
                f"GPU {index}: {properties.name}, "
                f"{properties.total_memory / 2**30:.1f} GiB"
            )
    print(f"Answer model: {base['answer_llm']['model']}")
    print("Publication conditions: " + ", ".join(sorted(configured)))
    print("Preflight: PASS")


if __name__ == "__main__":
    main()
