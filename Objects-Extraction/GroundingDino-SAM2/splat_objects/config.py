from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def slug(text: str) -> str:
    """Letters, digits and underscores only, so a name is safe in paths and archive keys."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def classes_from_prompts(
    prompts: list[str], defaults: dict[str, Any], overrides: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One class per prompt. Overrides are keyed by the prompt in lower case."""
    overrides = {k.strip().lower(): v for k, v in overrides.items()}
    unknown = set(overrides) - {p.strip().lower() for p in prompts}
    if unknown:
        raise ValueError(f"class_overrides for prompts that are not in the list: {sorted(unknown)}")
    classes, keys = [], set()
    for prompt in prompts:
        text = prompt.strip()
        key = slug(text.lower())
        if not key or key in keys:
            raise ValueError(f"prompt {prompt!r} gives an empty or repeated class key {key!r}")
        keys.add(key)
        classes.append({"key": key, "display": text[:1].upper() + text[1:], "prompt": text, **defaults, **overrides.get(text.lower(), {})})
    return classes


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = Path(path or repo_root() / "config.yaml").resolve()
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("extends", None)
    if base:
        cfg = _merge(load_config(p.parent / base), cfg)
    if "classes_file" in cfg:
        prompts = json.loads((p.parent / cfg["classes_file"]).read_text(encoding="utf-8"))
        cfg["classes"] = classes_from_prompts(prompts, cfg.get("class_defaults") or {}, cfg.get("class_overrides") or {})
    return cfg


def resolve_repo_path(cfg: dict[str, Any], rel: str) -> Path:
    return (repo_root() / rel).resolve()
