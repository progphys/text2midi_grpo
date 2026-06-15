from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def merge_config_files(paths: Iterable[Path | str]):
    cfg = OmegaConf.create()
    for path in paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    return cfg


def _resolve_path(project_root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def prepare_config(cfg, project_root: Path | None = None):
    project_root = project_root or _root()
    cfg.project_root = str(project_root)

    if "paths" in cfg:
        for key, value in cfg.paths.items():
            cfg.paths[key] = _resolve_path(project_root, value)

    return cfg


def train_config(model_name: str, experiment: str):
    root = _root()
    config_dir = root / "configs" / model_name
    cfg = merge_config_files(
        [
            config_dir / "model" / "paths.yaml",
            config_dir / "model" / "lora.yaml",
            config_dir / "metrics" / "base.yaml",
            config_dir / "reward" / "base.yaml",
            config_dir / "reward" / "observer.yaml",
            config_dir / "train" / "base.yaml",
            config_dir / "train" / "experiments" / f"{experiment}.yaml",
        ]
    )
    return prepare_config(cfg, root)


def inference_config(model_name: str):
    root = _root()
    config_dir = root / "configs" / model_name
    cfg = merge_config_files(
        [
            config_dir / "model" / "paths.yaml",
            config_dir / "metrics" / "base.yaml",
            config_dir / "reward" / "base.yaml",
            config_dir / "reward" / "observer.yaml",
            config_dir / "inference" / "base.yaml",
        ]
    )
    return prepare_config(cfg, root)


def evaluation_config(model_name: str, experiment: str | None = None):
    root = _root()
    config_dir = root / "configs" / model_name
    paths: list[Path] = [
        config_dir / "model" / "paths.yaml",
        config_dir / "metrics" / "base.yaml",
        config_dir / "reward" / "base.yaml",
        config_dir / "reward" / "observer.yaml",
        config_dir / "evaluation" / "base.yaml",
    ]
    if experiment:
        paths.extend(
            [
                config_dir / "model" / "lora.yaml",
                config_dir / "train" / "base.yaml",
                config_dir / "train" / "experiments" / f"{experiment}.yaml",
            ]
        )
    cfg = merge_config_files(paths)
    return prepare_config(cfg, root)
