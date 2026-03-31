from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lensing.calibration import TemperatureScaler
from lensing.config import ExperimentConfig
from lensing.data.datasets import build_dataloaders
from lensing.metrics import stable_sigmoid, summarize_binary_metrics
from lensing.models.factory import create_model
from lensing.utils import ensure_dir, resolve_device, save_json, seed_everything
from lensing.robustness import apply_perturbation


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 0.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        modulation = (1.0 - p_t).pow(self.gamma)
        return torch.mean(modulation * bce)


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    return images, labels


def _collect_outputs(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    perturbation: str | None = None,
    perturbation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    logits_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    paths: list[str] = []
    sources: list[str] = []
    running_loss = 0.0
    count = 0

    if perturbation_kwargs is None:
        perturbation_kwargs = {}

    with torch.no_grad():
        for batch in loader:
            images, labels = _move_batch(batch, device)

            if perturbation is not None:
                images = apply_perturbation(images, perturbation, **perturbation_kwargs)

            logits = model(images).view(-1)
            loss = criterion(logits, labels)
            batch_size = labels.numel()
            running_loss += float(loss.item()) * batch_size
            count += batch_size

            logits_list.append(logits.detach().cpu().numpy())
            targets_list.append(labels.detach().cpu().numpy())
            paths.extend(batch["path"])
            sources.extend(batch["source"])

    logits_array = np.concatenate(logits_list) if logits_list else np.array([], dtype=np.float32)
    targets_array = np.concatenate(targets_list) if targets_list else np.array([], dtype=np.float32)
    loss_value = running_loss / count if count else None
    return {
        "logits": logits_array,
        "targets": targets_array,
        "paths": paths,
        "sources": sources,
        "loss": loss_value,
    }


def _write_predictions(
    path: str | Path,
    outputs: dict[str, Any],
    split: str,
    temperature: float | None = None,
) -> None:
    path = Path(path)
    probabilities = stable_sigmoid(outputs["logits"])
    if temperature is not None:
        calibrated_logits = outputs["logits"] / temperature
        calibrated_probabilities = stable_sigmoid(calibrated_logits)
    else:
        calibrated_probabilities = None

    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["path", "source", "split", "label", "logit", "probability"]
        if calibrated_probabilities is not None:
            fieldnames.append("calibrated_probability")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, image_path in enumerate(outputs["paths"]):
            row = {
                "path": image_path,
                "source": outputs["sources"][index],
                "split": split,
                "label": int(outputs["targets"][index]),
                "logit": f"{float(outputs['logits'][index]):.6f}",
                "probability": f"{float(probabilities[index]):.6f}",
            }
            if calibrated_probabilities is not None:
                row["calibrated_probability"] = f"{float(calibrated_probabilities[index]):.6f}"
            writer.writerow(row)


def _pick_score(metrics: dict[str, Any]) -> float:
    if metrics["average_precision"] is not None:
        return float(metrics["average_precision"])
    if metrics["f1"] is not None:
        return float(metrics["f1"])
    return 0.0


def run_training(config: ExperimentConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.training.device)
    output_dir = ensure_dir(config.output_dir)
    loaders = build_dataloaders(config.data)

    model = create_model(config.model).to(device)
    criterion = BinaryFocalLoss(
        gamma=config.training.focal_gamma,
        pos_weight=config.training.positive_class_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=[parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    best_state = None
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            images, labels = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        train_outputs = _collect_outputs(model, loaders["train"], device, criterion)
        val_outputs = _collect_outputs(model, loaders["val"], device, criterion)
        train_metrics = summarize_binary_metrics(
            train_outputs["logits"],
            train_outputs["targets"],
            threshold=config.training.decision_threshold,
            loss=train_outputs["loss"],
        )
        val_metrics = summarize_binary_metrics(
            val_outputs["logits"],
            val_outputs["targets"],
            threshold=config.training.decision_threshold,
            loss=val_outputs["loss"],
        )

        current_score = _pick_score(val_metrics)
        if best_metrics is None or current_score > _pick_score(best_metrics):
            best_metrics = deepcopy(val_metrics)
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        if epochs_without_improvement >= config.training.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    val_outputs = _collect_outputs(model, loaders["val"], device, criterion)
    test_outputs = _collect_outputs(model, loaders["test"], device, criterion)

    temperature_value = None
    calibrated_val_metrics = None
    calibrated_test_metrics = None
    if config.training.calibration:
        scaler = TemperatureScaler()
        temperature_value = scaler.fit(
            torch.tensor(val_outputs["logits"], dtype=torch.float32),
            torch.tensor(val_outputs["targets"], dtype=torch.float32),
        )
        calibrated_val_metrics = summarize_binary_metrics(
            torch.tensor(val_outputs["logits"], dtype=torch.float32).div(temperature_value).numpy(),
            val_outputs["targets"],
            threshold=config.training.decision_threshold,
            loss=val_outputs["loss"],
        )
        calibrated_test_metrics = summarize_binary_metrics(
            torch.tensor(test_outputs["logits"], dtype=torch.float32).div(temperature_value).numpy(),
            test_outputs["targets"],
            threshold=config.training.decision_threshold,
            loss=test_outputs["loss"],
        )

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "temperature": temperature_value,
        },
        checkpoint_path,
    )

    _write_predictions(output_dir / "val_predictions.csv", val_outputs, split="val", temperature=temperature_value)
    _write_predictions(output_dir / "test_predictions.csv", test_outputs, split="test", temperature=temperature_value)

    summary = {
        "experiment": config.name,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "history": history,
        "val_metrics": summarize_binary_metrics(
            val_outputs["logits"],
            val_outputs["targets"],
            threshold=config.training.decision_threshold,
            loss=val_outputs["loss"],
        ),
        "test_metrics": summarize_binary_metrics(
            test_outputs["logits"],
            test_outputs["targets"],
            threshold=config.training.decision_threshold,
            loss=test_outputs["loss"],
        ),
        "temperature": temperature_value,
        "val_metrics_calibrated": calibrated_val_metrics,
        "test_metrics_calibrated": calibrated_test_metrics,
    }
    save_json(output_dir / "metrics.json", summary)
    return summary


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    split: str = "test",
    perturbation: str | None = None,
    perturbation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(config.training.device)
    loaders = build_dataloaders(config.data)
    criterion = BinaryFocalLoss(
        gamma=config.training.focal_gamma,
        pos_weight=config.training.positive_class_weight,
    ).to(device)

    model = create_model(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    outputs = _collect_outputs(model, loaders[split], device, criterion, perturbation=perturbation, perturbation_kwargs=perturbation_kwargs)
    metrics = summarize_binary_metrics(
        outputs["logits"],
        outputs["targets"],
        threshold=config.training.decision_threshold,
        loss=outputs["loss"],
    )

    temperature = checkpoint.get("temperature")
    calibrated_metrics = None
    if temperature is not None:
        calibrated_metrics = summarize_binary_metrics(
            torch.tensor(outputs["logits"], dtype=torch.float32).div(float(temperature)).numpy(),
            outputs["targets"],
            threshold=config.training.decision_threshold,
            loss=outputs["loss"],
        )

    return {
        "split": split,
        "perturbation": perturbation or "clean",
        "perturbation_kwargs": perturbation_kwargs or {},
        "metrics": metrics,
        "metrics_calibrated": calibrated_metrics,
        "temperature": temperature,
    }
