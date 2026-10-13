from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rfdetr_baseline.yaml"


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML mapping without importing PyYAML until it is needed."""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "PyYAML is required to read configuration files. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return config


def resolve_project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve relative configuration paths against the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def get_required_path(
    config: Dict[str, Any],
    section: str,
    field: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    section_value = config.get(section)
    if not isinstance(section_value, dict):
        raise ValueError(f"Config missing required section: {section}")
    value = section_value.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Config missing required field: {section}.{field}")
    return resolve_project_path(value, project_root)