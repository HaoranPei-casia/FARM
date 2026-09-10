import numpy as np
import torch

from farm.data import Trajectory
from farm.training import (
    TrainingConfig,
    fit_two_phase,
    load_checkpoint,
    predict,
    save_checkpoint,
)


def _records(tmp_path):
    rng = np.random.default_rng(3)
    inner = []
    validation = []
    for task_id in ("a", "b"):
        for split, destination in (
            ("inner_train", inner),
            ("validation", validation),
        ):
            for label in (0, 1):
                trajectory_id = f"{task_id}_{split}_{label}"
                values = rng.normal(size=(3, 4, 6)).astype(np.float32)
                values[..., 0] += 1.0 if label else -1.0
                path = tmp_path / f"{trajectory_id}.npy"
                np.save(path, values)
                destination.append(
                    Trajectory(trajectory_id, task_id, label, path, split)
                )
    return inner, validation


def test_two_phase_training_and_checkpoint_roundtrip(tmp_path):
    inner, validation = _records(tmp_path)
    config = TrainingConfig(
        input_width=6,
        hidden_width=4,
        trajectory_batch=2,
        max_epochs=2,
        patience=2,
        seed=9,
        device="cpu",
    )
    result = fit_two_phase(inner, validation, config)
    assert 1 <= result.selected_epochs <= 2
    assert len(result.validation_history) >= result.selected_epochs

    checkpoint = tmp_path / "farm.pt"
    save_checkpoint(checkpoint, result.model, config)
    restored, payload = load_checkpoint(checkpoint, "cpu")
    assert payload["format"] == "farm.readout.v1"

    expected = predict(result.model, validation, "cpu")
    actual = predict(restored, validation, "cpu")
    assert [item.risk for item in actual] == [item.risk for item in expected]
    assert all(torch.isfinite(torch.from_numpy(item.frame_scores)).all() for item in actual)
