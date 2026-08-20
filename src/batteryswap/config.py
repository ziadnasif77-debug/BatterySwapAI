"""Configuration loading with a hard guard against unresolved constants.

The no-invention rule (§0 of the project brief): any quantity not read from an
official competition file is stored as the literal string ``UNKNOWN`` and must
raise if a code path tries to consume it. This module is the single enforcement
point — all config access goes through :class:`GuardedConfig`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

UNKNOWN = "UNKNOWN"

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class UnknownValueError(RuntimeError):
    """Raised when code consumes a constant that has not been resolved from
    the official competition files."""

    def __init__(self, key_path: str):
        super().__init__(
            f"Config value '{key_path}' is UNKNOWN. It must be transcribed from the "
            f"official competition files (see configs/unknowns.yaml) before this code "
            f"path can run. Guessing it is forbidden by the no-invention rule."
        )
        self.key_path = key_path


class GuardedConfig:
    """Read-only nested mapping. Accessing a leaf equal to ``UNKNOWN`` raises
    :class:`UnknownValueError` identifying the full key path."""

    def __init__(self, data: dict[str, Any], _path: str = ""):
        self._data = data
        self._path = _path

    def __getitem__(self, key: str) -> Any:
        path = f"{self._path}.{key}" if self._path else key
        try:
            value = self._data[key]
        except KeyError:
            raise KeyError(f"Config key '{path}' does not exist") from None
        if isinstance(value, dict):
            return GuardedConfig(value, path)
        if isinstance(value, str) and value == UNKNOWN:
            raise UnknownValueError(path)
        return value

    def get_raw(self, key: str) -> Any:
        """Access without the UNKNOWN guard — for status reporting only."""
        return self._data[key]

    def is_unknown(self, key: str) -> bool:
        value = self._data.get(key)
        return isinstance(value, str) and value == UNKNOWN

    def keys(self):
        return self._data.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"GuardedConfig({self._path or '<root>'}, keys={list(self._data)})"


def load_config(name: str, config_dir: Path | None = None) -> GuardedConfig:
    """Load ``configs/<name>.yaml`` as a :class:`GuardedConfig`."""
    # PyYAML is absent from the competition runtime, and the planner never
    # reads a config there - so the import stays inside the function.
    import yaml

    directory = config_dir or CONFIG_DIR
    path = directory / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return GuardedConfig(data)


def unresolved_unknowns(config_dir: Path | None = None) -> dict[str, str]:
    """Return {name: blocks} for every unresolved entry in unknowns.yaml."""
    registry = load_config("unknowns", config_dir).get_raw("unknowns")
    return {
        name: entry.get("blocks", "")
        for name, entry in registry.items()
        if entry.get("status") != "resolved" and entry.get("value") == UNKNOWN
    }
