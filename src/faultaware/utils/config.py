"""Configuration loading (OmegaConf YAML).

The root ``configs/config.yaml`` names one file per group in ``defaults``; each group file
``configs/<group>/<name>.yaml`` is merged under the key ``<group>``. Dotted overrides
(``train.lr=0.01``) are applied last.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(overrides: list[str] | None = None, config_dir: Path | None = None) -> DictConfig:
    """Load and merge the project configuration.

    Args:
        overrides: Optional dotted overrides, e.g. ``["train.max_epochs=2"]``.
        config_dir: Directory containing ``config.yaml`` (defaults to ``configs/``).

    Returns:
        The merged, resolved configuration.

    Raises:
        FileNotFoundError: If the root or a group config file is missing.
    """
    cdir = config_dir or CONFIG_DIR
    root = OmegaConf.load(cdir / "config.yaml")
    assert isinstance(root, DictConfig)
    defaults: dict[str, Any] = OmegaConf.to_container(root.pop("defaults", {}))  # type: ignore[assignment]
    merged: Any = root
    for group, name in defaults.items():
        group_file = cdir / group / f"{name}.yaml"
        if not group_file.is_file():
            raise FileNotFoundError(group_file)
        merged = OmegaConf.merge(merged, {group: OmegaConf.load(group_file)})
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    assert isinstance(merged, DictConfig)
    OmegaConf.resolve(merged)
    return merged


def project_path(relative: str | Path) -> Path:
    """Resolve a config path relative to the project root.

    Args:
        relative: A path from the config (relative paths are anchored at the project root).

    Returns:
        Absolute path.
    """
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert a config to a plain dict (for logging / JSON)."""
    out = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(out, dict)
    return out  # type: ignore[return-value]
