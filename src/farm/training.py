"""Task-balanced training, prediction, and checkpoint utilities."""

from __future__ import annotations

import copy
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Trajectory, load_tokens
from .model import FARMReadout


@dataclass(frozen=True)
class TrainingConfig:
    input_width: int = 1024
    hidden_width: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    trajectory_batch: int = 4
    max_epochs: int = 100
    patience: int = 10
    seed: int = 0
    device: str = "auto"

    def validate(self) -> None:
        positive = {
            "input_width": self.input_width,
            "hidden_width": self.hidden_width,
            "learning_rate": self.learning_rate,
            "trajectory_batch": self.trajectory_batch,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")


@dataclass
class FitResult:
    model: FARMReadout
    selected_epochs: int
    best_validation_loss: float
    validation_history: list[float]


@dataclass(frozen=True)
class Prediction:
    trajectory_id: str
    task_id: str
    label: int
    risk: float
    frame_scores: np.ndarray


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {requested!r}, but CUDA is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_model(
    config: TrainingConfig, device: torch.device
) -> tuple[FARMReadout, torch.optim.Optimizer]:
    seed_everything(config.seed)
    model = FARMReadout(config.input_width, config.hidden_width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return model, optimizer


def task_weights(records: Iterable[Trajectory]) -> dict[str, float]:
    records = list(records)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.task_id] = counts.get(record.task_id, 0) + 1
    if not counts:
        raise ValueError("cannot compute task weights for an empty split")
    count = len(records)
    return {
        task_id: count / (len(counts) * task_count)
        for task_id, task_count in counts.items()
    }


def _autocast(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _forward(model: FARMReadout, tokens: torch.Tensor, device: torch.device):
    tokens = tokens.to(device=device, dtype=torch.float32, non_blocking=True)
    with _autocast(device):
        return model(tokens)


def train_epoch(
    model: FARMReadout,
    optimizer: torch.optim.Optimizer,
    records: list[Trajectory],
    weights: dict[str, float],
    rng: np.random.RandomState,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    order = np.asarray(records, dtype=object)
    rng.shuffle(order)
    model.train()
    for start in range(0, len(order), config.trajectory_batch):
        group = order[start : start + config.trajectory_batch].tolist()
        denominator = sum(weights[record.task_id] for record in group)
        optimizer.zero_grad(set_to_none=True)
        for record in group:
            tokens = load_tokens(record, config.input_width)
            logits = _forward(model, tokens, device)
            target = torch.full_like(logits, float(record.label))
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
            scaled = loss * weights[record.task_id] / (len(tokens) * denominator)
            scaled.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()


def validation_loss(
    model: FARMReadout,
    records: list[Trajectory],
    weights: dict[str, float],
    config: TrainingConfig,
    device: torch.device,
) -> float:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    with torch.inference_mode():
        for record in records:
            if record.task_id not in weights:
                raise ValueError(
                    f"validation task {record.task_id!r} is absent from inner_train"
                )
            logits = _forward(
                model, load_tokens(record, config.input_width), device
            )
            target = torch.full_like(logits, float(record.label))
            loss = float(
                F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
                .float()
                .cpu()
            )
            weight = weights[record.task_id]
            numerator += weight * loss
            denominator += weight
    return numerator / denominator


def fit_two_phase(
    inner_train: list[Trajectory],
    validation: list[Trajectory],
    config: TrainingConfig,
) -> FitResult:
    """Select epochs on validation, then retrain on the full outer train split."""
    config.validate()
    if not inner_train or not validation:
        raise ValueError("inner_train and validation must both be non-empty")
    device = resolve_device(config.device)

    phase_a, optimizer = initialize_model(config, device)
    weights = task_weights(inner_train)
    rng = np.random.RandomState(80_000 + config.seed)
    history: list[float] = []
    best_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    for epoch in range(config.max_epochs):
        train_epoch(phase_a, optimizer, inner_train, weights, rng, config, device)
        loss = validation_loss(phase_a, validation, weights, config, device)
        history.append(loss)
        if loss < best_loss - 1e-7:
            best_loss = loss
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break
    if best_epoch < 0:
        raise RuntimeError("epoch selection failed")
    selected_epochs = best_epoch + 1

    outer_train = [*inner_train, *validation]
    phase_b, optimizer = initialize_model(config, device)
    weights = task_weights(outer_train)
    rng = np.random.RandomState(80_000 + config.seed)
    for _ in range(selected_epochs):
        train_epoch(phase_b, optimizer, outer_train, weights, rng, config, device)

    return FitResult(
        model=phase_b,
        selected_epochs=selected_epochs,
        best_validation_loss=best_loss,
        validation_history=history,
    )


def adapt_fixed_epochs(
    model: FARMReadout,
    records: list[Trajectory],
    config: TrainingConfig,
    epochs: int,
) -> FARMReadout:
    """Update a source readout on target demonstrations for a fixed epoch count."""
    config.validate()
    if not records or epochs < 1:
        raise ValueError("adaptation records and a positive epoch count are required")
    device = resolve_device(config.device)
    seed_everything(config.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    weights = task_weights(records)
    rng = np.random.RandomState(80_000 + config.seed)
    for _ in range(epochs):
        train_epoch(model, optimizer, records, weights, rng, config, device)
    return model


def predict(
    model: FARMReadout,
    records: list[Trajectory],
    device_name: str = "auto",
) -> list[Prediction]:
    device = resolve_device(device_name)
    model = model.to(device)
    model.eval()
    predictions: list[Prediction] = []
    with torch.inference_mode():
        for record in records:
            logits = _forward(
                model, load_tokens(record, model.input_width), device
            )
            frame_scores = torch.sigmoid(logits).float().cpu().numpy()
            predictions.append(
                Prediction(
                    trajectory_id=record.trajectory_id,
                    task_id=record.task_id,
                    label=record.label,
                    risk=float(frame_scores.max()),
                    frame_scores=frame_scores,
                )
            )
    return predictions


def save_checkpoint(
    path: str | Path,
    model: FARMReadout,
    config: TrainingConfig,
    metadata: dict | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(
        {
            "format": "farm.readout.v1",
            "model": {
                "input_width": model.input_width,
                "hidden_width": model.hidden_width,
            },
            "training": asdict(config),
            "state_dict": state,
            "metadata": metadata or {},
        },
        destination,
    )


def load_checkpoint(path: str | Path, device_name: str = "auto") -> tuple[FARMReadout, dict]:
    device = resolve_device(device_name)
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "farm.readout.v1":
        raise ValueError("unsupported FARM checkpoint format")
    model_config = payload["model"]
    model = FARMReadout(
        input_width=int(model_config["input_width"]),
        hidden_width=int(model_config["hidden_width"]),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device), copy.deepcopy(payload)
