"""Simple immutable result folders used by the final simulations."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.preprint import PROTOCOL_VERSION


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_artifact(output: Path, config: dict[str, Any], arrays: dict[str, np.ndarray], rows: list[dict[str, Any]], report: dict[str, Any]) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty artifact: {output}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "arrays.npz", **arrays)
    _json(output / "config.json", config)
    if rows:
        with (output / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    digest = _digest(arrays)
    _json(output / "report.json", {**report, "protocol_version": PROTOCOL_VERSION, "scientific_digest": digest})
    root = Path(__file__).resolve().parents[3]
    _json(output / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "git_revision": git_revision(root), "scientific_digest": digest})
    return output


def load_arrays(artifact: Path) -> dict[str, np.ndarray]:
    with np.load(artifact / "arrays.npz", allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}
