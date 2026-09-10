import pytest

from farm.metrics import average_precision, binary_auroc, binary_metrics


def test_perfect_binary_ranking():
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.8, 0.2, 0.9]
    assert binary_auroc(labels, scores) == pytest.approx(1.0)
    assert average_precision(labels, scores) == pytest.approx(1.0)
    assert binary_metrics(labels, scores)["accuracy"] == pytest.approx(1.0)


def test_tied_ranking():
    assert binary_auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert average_precision([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
