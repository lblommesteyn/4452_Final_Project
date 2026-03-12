from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest_path: str = "data/synthetic/manifest.csv"
    image_size: int = 128
    batch_size: int = 16
    num_workers: int = 0
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    use_weighted_sampler: bool = True


@dataclass
class ModelConfig:
    name: str = "resnet18"
    image_size: int = 128
    pretrained: bool = False
    dropout: float = 0.1
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 6
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    positive_class_weight: float = 1.0
    focal_gamma: float = 0.0
    patience: int = 3
    device: str = "auto"
    mixed_precision: bool = False
    calibration: bool = True
    decision_threshold: float = 0.5


@dataclass
class ExperimentConfig:
    name: str = "lens_experiment"
    output_dir: str = "outputs/lens_experiment"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_section(section_cls: type, values: dict[str, Any] | None) -> Any:
    values = values or {}
    allowed = {key: values[key] for key in values if key in section_cls.__dataclass_fields__}
    return section_cls(**allowed)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    experiment_values = raw.get("experiment", {})
    return ExperimentConfig(
        name=experiment_values.get("name", "lens_experiment"),
        output_dir=experiment_values.get("output_dir", "outputs/lens_experiment"),
        seed=experiment_values.get("seed", 42),
        data=_load_section(DataConfig, raw.get("data")),
        model=_load_section(ModelConfig, raw.get("model")),
        training=_load_section(TrainingConfig, raw.get("training")),
    )

