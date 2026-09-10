import csv

import numpy as np

from farm.data import load_manifest, load_tokens, select_split


def test_manifest_resolves_relative_features_and_common_length(tmp_path):
    features = np.zeros((5, 3, 4), dtype=np.float32)
    np.save(tmp_path / "tokens.npy", features)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trajectory_id",
                "task_id",
                "label",
                "b1_path",
                "split",
                "common_length",
            ]
        )
        writer.writerow(["t0", "task", 0, "tokens.npy", "test", 3])

    records = load_manifest(manifest)
    assert select_split(records, "test") == records
    assert load_tokens(records[0], input_width=4).shape == (3, 3, 4)
