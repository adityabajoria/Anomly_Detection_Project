import numpy as np
import pytest

from evaluation.metrics import (
    evaluate_predictions,
    best_f1_over_thresholds,
    point_adjust,
    evaluate_detector,
)


def test_point_adjust_credits_whole_segment():
    y_true = np.array([0, 1, 1, 1, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 0, 0])
    expected = np.array([0, 1, 1, 1, 0, 0, 0])
    assert np.array_equal(point_adjust(y_true, y_pred), expected)


def test_point_adjust_segment_at_array_end():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1])
    expected = np.array([0, 0, 1, 1])
    assert np.array_equal(point_adjust(y_true, y_pred), expected)


def test_point_adjust_no_detection_no_credit():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert np.array_equal(point_adjust(y_true, y_pred), y_pred)


def test_point_adjust_false_positives_untouched():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1])
    expected = np.array([1, 0, 1, 1, 1])
    assert np.array_equal(point_adjust(y_true, y_pred), expected)


def test_point_adjust_does_not_mutate_input():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    original = y_pred.copy()
    point_adjust(y_true, y_pred)
    assert np.array_equal(y_pred, original)


def test_evaluate_predictions_no_positive_predictions():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    result = evaluate_predictions(y_true, y_pred)
    assert result["precision"] == 0
    assert result["recall"] == 0
    assert result["f1"] == 0


def test_best_f1_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    result = best_f1_over_thresholds(y_true, scores)
    assert result["best_f1"] == pytest.approx(1.0)
    assert 0.3 < result["threshold"] <= 0.8


def test_evaluate_detector_returns_both_protocols():
    y_true = np.array([0, 0, 1, 1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.1, 0.2, 0.1, 0.3])
    result = evaluate_detector(y_true, scores)
    assert result["point_adjusted"]["f1"] >= result["honest"]["f1"]