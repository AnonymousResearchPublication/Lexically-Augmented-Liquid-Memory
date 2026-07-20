from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = config.pop("defaults", None)
    return _merge(load_config(path.parent / parent), config) if parent else config
