from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from lensing.config import DataConfig


@dataclass
class ManifestRecord:
    image_path: Path
    label: int
    split: str
    source: str


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    ops: list = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(25),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(ops)


class LensManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, image_size: int, train: bool) -> None:
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.records = self._read_records(split)
        self.transform = build_transforms(image_size=image_size, train=train)

    def _read_records(self, split: str) -> list[ManifestRecord]:
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "label", "split", "source"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                missing = required.difference(reader.fieldnames or [])
                raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

            records: list[ManifestRecord] = []
            for row in reader:
                if row["split"] != split:
                    continue
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = (self.base_dir / image_path).resolve()
                records.append(
                    ManifestRecord(
                        image_path=image_path,
                        label=int(row["label"]),
                        split=row["split"],
                        source=row["source"],
                    )
                )
        if not records:
            raise ValueError(f"No manifest rows found for split '{split}' in {self.manifest_path}")
        return records

    @property
    def targets(self) -> list[int]:
        return [record.label for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        label = torch.tensor(record.label, dtype=torch.float32)
        return {
            "image": tensor,
            "label": label,
            "path": str(record.image_path),
            "source": record.source,
        }


def make_weighted_sampler(targets: list[int]) -> WeightedRandomSampler:
    positives = sum(targets)
    negatives = len(targets) - positives
    positive_weight = 0.0 if positives == 0 else len(targets) / (2.0 * positives)
    negative_weight = 0.0 if negatives == 0 else len(targets) / (2.0 * negatives)
    weights = [positive_weight if label == 1 else negative_weight for label in targets]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_dataloaders(config: DataConfig) -> dict[str, DataLoader]:
    train_dataset = LensManifestDataset(
        manifest_path=config.manifest_path,
        split=config.train_split,
        image_size=config.image_size,
        train=True,
    )
    val_dataset = LensManifestDataset(
        manifest_path=config.manifest_path,
        split=config.val_split,
        image_size=config.image_size,
        train=False,
    )
    test_dataset = LensManifestDataset(
        manifest_path=config.manifest_path,
        split=config.test_split,
        image_size=config.image_size,
        train=False,
    )

    sampler = make_weighted_sampler(train_dataset.targets) if config.use_weighted_sampler else None
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=config.num_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    }

