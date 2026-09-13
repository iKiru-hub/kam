"""Read and validate small JSON experiment configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("root_seeds", "data", "autoencoder", "memory"):
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    if not config["root_seeds"]:
        raise ValueError("root_seeds must not be empty")
    if config["data"]["dimension"] < 2:
        raise ValueError("dimension must be at least two")
    return config
