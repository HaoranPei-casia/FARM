"""Manifest and B1 feature loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    task_id: str
    label: int
    b1_path: Path
    split: str
    common_length: int | None = None


def load_manifest(path: str | Path) -> list[Trajectory]:
    """Load trajectory metadata and resolve feature paths by manifest location."""
    manifest_path = Path(path).expanduser().resolve()
    required = {"trajectory_id", "task_id", "label", "b1_path", "split"}
    records: list[Trajectory] = []

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
            except ValueError as exc:
                raise ValueError(f"line {line_number}: label must be 0 or 1") from exc
            if label not in (0, 1):
                raise ValueError(f"line {line_number}: label must be 0 or 1")

            raw_feature_path = Path(row["b1_path"]).expanduser()
            feature_path = (
                raw_feature_path
                if raw_feature_path.is_absolute()
                else manifest_path.parent / raw_feature_path
            ).resolve()
            raw_length = (row.get("common_length") or "").strip()
            common_length = int(raw_length) if raw_length else None
            if common_length is not None and common_length < 1:
                raise ValueError(f"line {line_number}: common_length must be positive")

            record = Trajectory(
                trajectory_id=row["trajectory_id"].strip(),
                task_id=row["task_id"].strip(),
                label=label,
                b1_path=feature_path,
                split=row["split"].strip(),
                common_length=common_length,
            )
            if not record.trajectory_id or not record.task_id or not record.split:
                raise ValueError(f"line {line_number}: identifiers and split are required")
            records.append(record)

    ids = [record.trajectory_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("trajectory_id values must be unique")
    if not records:
        raise ValueError("manifest has no trajectories")
    return records


def load_tokens(record: Trajectory, input_width: int | None = None) -> Tensor:
    """Load one trajectory as a float32 tensor with shape ``[T, N, D]``."""
    array = np.load(record.b1_path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3:
        raise ValueError(
            f"{record.b1_path}: expected [T, N, D], got shape {array.shape}"
        )
    if input_width is not None and array.shape[-1] != input_width:
        raise ValueError(
            f"{record.b1_path}: expected width {input_width}, got {array.shape[-1]}"
        )
    length = array.shape[0]
    if record.common_length is not None:
        if record.common_length > length:
            raise ValueError(
                f"{record.b1_path}: common_length {record.common_length} exceeds {length}"
            )
        length = record.common_length
    # Copy detaches the tensor from the read-only memory map.
    tokens = torch.from_numpy(np.array(array[:length], dtype=np.float32, copy=True))
    if not torch.isfinite(tokens).all():
        raise ValueError(f"{record.b1_path}: tokens contain NaN or infinity")
    return tokens


def select_split(records: list[Trajectory], split: str) -> list[Trajectory]:
    selected = [record for record in records if record.split == split]
    if not selected:
        raise ValueError(f"manifest has no trajectories in split {split!r}")
    return selected
