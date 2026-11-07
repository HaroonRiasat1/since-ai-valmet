from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or repo_root() / "config.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_repo_path(cfg: dict[str, Any], rel: str) -> Path:
    return (repo_root() / rel).resolve()
