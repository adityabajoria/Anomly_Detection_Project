import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, auc,
)

def evaluate_predictions(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def best_f1_over_thresholds(y_true, scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    best_idx = np.argmax(f1)
    return {
        "best_f1": f1[best_idx],
        "precision": precisions[best_idx],
        "recall": recalls[best_idx],
        "threshold": thresholds[min(best_idx, len(thresholds) - 1)],
        "pr_auc": auc(recalls, precisions),
    }

def point_adjust(y_true, y_pred):
    y_pred = y_pred.copy()
    in_seg = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_seg:
            in_seg, start = True, i
        if in_seg and (y_true[i] == 0 or i == len(y_true) - 1):
            end = i if y_true[i] == 0 else i + 1
            if y_pred[start:end].any():
                y_pred[start:end] = 1
            in_seg = False
    return y_pred

def find_segments(labels):
    segments = []
    start = None
    for i, val in enumerate(labels):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def detection_delay(labels, predictions):
    labels = np.asarray(labels).ravel()
    predictions = np.asarray(predictions).ravel()
    segments = find_segments(labels)

    delays = []
    for start, end in segments:
        seg_preds = predictions[start:end + 1]
        flagged = np.where(seg_preds == 1)[0]
        if len(flagged) > 0:
            delays.append(int(flagged[0]))

    return {
        "mean_delay": float(np.mean(delays)) if delays else None,
        "segments_detected": len(delays),
        "segments_total": len(segments),
    }

def evaluate_detector(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    sweep = best_f1_over_thresholds(y_true, scores)
    y_pred = (scores >= sweep['threshold']).astype(int)

    honest = evaluate_predictions(y_true, y_pred)
    adjusted = evaluate_predictions(y_true, point_adjust(y_true, y_pred))
    delay = detection_delay(y_true, y_pred)

    return {
        "pr_auc": sweep["pr_auc"],
        "threshold": sweep["threshold"],
        "honest": honest,
        "point_adjusted": adjusted,
        "detection_delay": delay
    }



