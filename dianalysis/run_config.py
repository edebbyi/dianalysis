"""Helpers for loading and merging optional TOML run configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover - final fallback
        import toml as _toml

        class _TomlCompat:
            @staticmethod
            def load(f: Any) -> Any:
                text = f.read()
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                return _toml.loads(text)

        tomllib = _TomlCompat()  # type: ignore[assignment]


def load_toml_config(path: Path | None) -> dict[str, Any]:
    """Load TOML config if present; otherwise return an empty dict."""
    if path is None:
        return {}
    if not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data if isinstance(data, dict) else {}


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dicts recursively, with override values taking precedence."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        cur = out.get(key)
        if isinstance(cur, dict) and isinstance(value, dict):
            out[key] = deep_merge_dicts(cur, value)
        else:
            out[key] = value
    return out


def resolve_profile_path(config_path: Path | None, profile_path: Path | None) -> Path | None:
    """Resolve profile path relative to the config file location when needed."""
    if profile_path is None:
        return None
    if profile_path.exists() or profile_path.is_absolute():
        return profile_path

    # First try profile path relative to the base config location.
    if config_path is not None:
        candidate = config_path.parent / profile_path
        if candidate.exists():
            return candidate

    # Then try a common shared profiles directory from repository root.
    candidate = Path("configs/profiles") / profile_path
    if candidate.exists():
        return candidate

    return profile_path


def load_runtime_config(config_path: Path | None, profile_path: Path | None = None) -> dict[str, Any]:
    """Load base config and optional profile overlay into one merged dictionary."""
    base_cfg = load_toml_config(config_path)
    resolved_profile = resolve_profile_path(config_path, profile_path)
    profile_cfg = load_toml_config(resolved_profile)
    return deep_merge_dicts(base_cfg, profile_cfg)


def cfg_get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return nested config value, falling back to default on missing keys."""
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def to_jsonable(value: Any) -> Any:
    """Convert values into JSON-safe equivalents."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2))
