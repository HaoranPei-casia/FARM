"""Command-line entry points for FARM."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .data import load_manifest, select_split
from .metrics import binary_metrics
from .training import (
    TrainingConfig,
    adapt_fixed_epochs,
    fit_two_phase,
    load_checkpoint,
    predict,
    save_checkpoint,
)


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _config(raw: dict[str, Any]) -> TrainingConfig:
    names = TrainingConfig.__dataclass_fields__.keys()
    return TrainingConfig(**{name: raw[name] for name in names if name in raw})


def _metrics(predictions) -> dict[str, float | int | None]:
    metrics = binary_metrics(
        [item.label for item in predictions],
        [item.risk for item in predictions],
    )
    return {
        key: None if isinstance(value, float) and math.isnan(value) else value
        for key, value in metrics.items()
    }


def _write_predictions(path: Path, predictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trajectory_id", "task_id", "label", "risk", "num_steps"],
        )
        writer.writeheader()
        for item in predictions:
            writer.writerow(
                {
                    "trajectory_id": item.trajectory_id,
                    "task_id": item.task_id,
                    "label": item.label,
                    "risk": f"{item.risk:.9g}",
                    "num_steps": len(item.frame_scores),
                }
            )


def command_train(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    raw = _json(config_path)
    base = config_path.parent
    config = _config(raw)
    records = load_manifest(_path(raw["manifest"], base))
    inner_train = select_split(records, "inner_train")
    validation = select_split(records, "validation")
    result = fit_two_phase(inner_train, validation, config)
    checkpoint = _path(raw["output_checkpoint"], base)
    save_checkpoint(
        checkpoint,
        result.model,
        config,
        {
            "selected_epochs": result.selected_epochs,
            "best_validation_loss": result.best_validation_loss,
        },
    )
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "selected_epochs": result.selected_epochs,
        "best_validation_loss": result.best_validation_loss,
    }
    test = [record for record in records if record.split == "test"]
    if test:
        report["test"] = _metrics(predict(result.model, test, config.device))
    print(json.dumps(report, indent=2, sort_keys=True))


def command_adapt(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    raw = _json(config_path)
    base = config_path.parent
    config = _config(raw)
    records = load_manifest(_path(raw["manifest"], base))
    adaptation = select_split(records, "adapt")
    model, source_payload = load_checkpoint(
        _path(raw["source_checkpoint"], base), config.device
    )
    if model.input_width != config.input_width or model.hidden_width != config.hidden_width:
        raise ValueError("adaptation configuration does not match source checkpoint")
    epochs = int(raw.get("adapt_epochs", 20))
    model = adapt_fixed_epochs(model, adaptation, config, epochs)
    checkpoint = _path(raw["output_checkpoint"], base)
    save_checkpoint(
        checkpoint,
        model,
        config,
        {
            "adapt_epochs": epochs,
            "source_format": source_payload["format"],
        },
    )
    report: dict[str, Any] = {"checkpoint": str(checkpoint), "adapt_epochs": epochs}
    test = [record for record in records if record.split == "test"]
    if test:
        report["test"] = _metrics(predict(model, test, config.device))
    print(json.dumps(report, indent=2, sort_keys=True))


def command_evaluate(args: argparse.Namespace) -> None:
    model, _ = load_checkpoint(args.checkpoint, args.device)
    records = load_manifest(args.manifest)
    if args.split:
        records = select_split(records, args.split)
    predictions = predict(model, records, args.device)
    if args.output:
        _write_predictions(Path(args.output).expanduser().resolve(), predictions)
    print(json.dumps(_metrics(predictions), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="farm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="run two-phase source training")
    train.add_argument("--config", required=True)
    train.set_defaults(func=command_train)

    adapt = subparsers.add_parser("adapt", help="run fixed-epoch few-shot adaptation")
    adapt.add_argument("--config", required=True)
    adapt.set_defaults(func=command_adapt)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a FARM checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--split")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
