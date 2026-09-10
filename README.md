# FARM

FARM is a lightweight trajectory-failure readout for frozen visual world-model
features. It turns a per-frame `B1` token tensor into a failure probability for
each timestep, then uses the maximum timestep probability as the trajectory
risk.

This repository is the minimal public implementation. It intentionally does not
include private robot data, model weights, experiment outputs, figures, or
third-party model source trees. Extract `B1` features with your own frozen
backbone and provide them as NumPy arrays.

The exact trajectory identifiers, rollout seeds, source sequence locators, and
fixed splits used by the main experiments are published in
[`DATA.md`](DATA.md). No trajectory content or extracted feature is included.

## Method

Each trajectory is stored as a `.npy` array with shape `[T, N, D]`:

- `T`: timesteps
- `N`: visual tokens per timestep
- `D`: token width (1024 in the main configuration)

FARM projects and normalizes the tokens, learns scalar attention over the token
dimension, pools the tokens, and applies a two-layer MLP readout. For the main
configuration (`D=1024`, hidden width 32), the readout has 33,985 trainable
parameters.

Training follows a two-phase protocol:

1. Select the epoch count on `inner_train` / `validation` trajectories.
2. Reinitialize from the same seed and train on their union for the selected
   number of epochs.

Losses are balanced by task, so each task contributes equal total weight even
when task sample counts differ. A trajectory label is applied to every frame.
Few-shot adaptation starts from a trained checkpoint and updates only this
readout for a fixed number of epochs.

## Installation

Python 3.10 or newer is required.

```bash
pip install -e ".[test]"
```

## Data manifest

The CSV manifest has five required columns:

```text
trajectory_id,task_id,label,b1_path,split
```

`label` is `0` for success and `1` for failure. Relative `b1_path` values are
resolved relative to the manifest. For source training, `split` is one of
`inner_train`, `validation`, or `test`. For adaptation, use `adapt` and `test`.
An optional `common_length` column truncates a trajectory to a fixed prefix.

## Quick start

Generate a small synthetic dataset, run source training, then adapt the readout:

```bash
python examples/make_toy_data.py
farm train --config configs/train.example.json
farm evaluate \
  --checkpoint outputs/farm_toy.pt \
  --manifest examples/toy_data/source_manifest.csv \
  --split test
farm adapt --config configs/adapt.example.json
pytest
```

The example is only a functional smoke test. It is not an experimental result.

## Checkpoint and evaluation outputs

`farm train` and `farm adapt` write a PyTorch checkpoint. `farm evaluate` prints
trajectory-level AUROC, average precision, and accuracy at threshold 0.5. Pass
`--output predictions.csv` to save per-trajectory risks. No experiment result is
tracked in this repository.

## Scope

This release contains the FARM readout, task-balanced training, fixed-epoch
few-shot adaptation, metrics, a command-line interface, and tests. Frozen
VLA/V-JEPA feature extraction and real-robot control are deliberately outside
the package boundary.

## License

MIT. See [LICENSE](LICENSE).
