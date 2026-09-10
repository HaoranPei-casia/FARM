"""Create tiny synthetic manifests for the README smoke test."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


OUTPUT = Path(__file__).resolve().parent / "toy_data"
FIELDS = ["trajectory_id", "task_id", "label", "b1_path", "split"]


def feature(rng: np.random.Generator, label: int, target_shift: bool) -> np.ndarray:
    tokens = rng.normal(0.0, 0.45, size=(5, 6, 8)).astype(np.float32)
    direction = -1.0 if target_shift else 1.0
    tokens[..., 0] += direction * (1.5 if label else -1.5)
    return tokens


def write_dataset(name: str, splits: dict[str, int], target_shift: bool) -> None:
    directory = OUTPUT / name
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7 if name == "source" else 17)
    rows = []
    for task_index in range(2):
        task_id = f"{name}_task_{task_index}"
        for split, repetitions in splits.items():
            for repetition in range(repetitions):
                for label in (0, 1):
                    trajectory_id = f"{task_id}_{split}_{repetition}_{label}"
                    path = directory / f"{trajectory_id}.npy"
                    np.save(path, feature(rng, label, target_shift))
                    rows.append(
                        {
                            "trajectory_id": trajectory_id,
                            "task_id": task_id,
                            "label": label,
                            "b1_path": path.relative_to(OUTPUT),
                            "split": split,
                        }
                    )
    with (OUTPUT / f"{name}_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_dataset(
        "source",
        {"inner_train": 2, "validation": 1, "test": 1},
        target_shift=False,
    )
    write_dataset("target", {"adapt": 2, "test": 1}, target_shift=True)
    print(f"wrote toy data to {OUTPUT}")


if __name__ == "__main__":
    main()
