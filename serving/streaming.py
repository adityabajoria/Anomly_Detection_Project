import json
import time
from collections import deque

import numpy as np


def _sse(data):
    return f"data: {json.dumps(data)}\n\n"


def _find_segments(labels):
    segs, start = [], None
    for i, v in enumerate(labels):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            segs.append((start, i - 1)); start = None
    if start is not None:
        segs.append((start, len(labels) - 1))
    return segs


def _threshold(scores, labels):
    # oracle best-F1 threshold — same idea as evaluation, computed on the fly
    from sklearn.metrics import precision_recall_curve
    p, r, th = precision_recall_curve(labels, scores)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    best = int(np.argmax(f1))
    return float(th[min(best, len(th) - 1)])


def stream_scores(detector, test, labels, delay=0.02):
    """Yield SSE frames: one meta frame, then one per timestep, then done."""

    # Precompute the full score vector once to derive a stable threshold.
    full_scores = detector.score(test)

    if len(full_scores) < len(test):
        pad = np.zeros(len(test) - len(full_scores))
        full_scores = np.concatenate([pad, full_scores])

    threshold = _threshold(full_scores, labels)
    segments = _find_segments(labels)

    yield _sse({
        "meta": True,
        "threshold": threshold,
        "n_segments": len(segments),
        "n_timesteps": len(test)
    })

    is_lstm = detector.name == "lstm_autoencoder"

    if is_lstm:
        window_size = detector.window_size
        buffer = deque(maxlen=window_size)

    for t in range(len(test)):
        row = test[t]

        if is_lstm:
            buffer.append(row)

            if len(buffer) < window_size:
                score = None
            else:
                window = np.array(buffer, dtype=float)
                score = float(detector.score(window)[-1])

        else:
            score = float(detector.score_one(row))

        yield _sse({
            "t": t,
            "score": score,
            "label": int(labels[t])
        })

        if delay:
            time.sleep(delay)

    yield _sse({"done": True})