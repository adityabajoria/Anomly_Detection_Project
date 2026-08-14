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


def _score_step(detector, row, is_lstm, buffer, window_size):
    """Score a single timestep. Returns None during LSTM warm-up."""
    if is_lstm:
        buffer.append(row)
        if len(buffer) < window_size:
            return None
        window = np.array(buffer, dtype=float)
        return float(np.asarray(detector.score(window)).ravel()[-1])
    # BUGFIX: detectors expose .score(X) (batch API, same one /api/score uses),
    # not .score_one(row). Calling the nonexistent method raised AttributeError
    # on the first frame, killing the SSE stream immediately.
    if hasattr(detector, "score_one"):
        return float(detector.score_one(row))
    return float(np.asarray(detector.score(np.asarray(row, dtype=float)[None, :])).ravel()[0])


def stream_scores(detector, test, labels, threshold, delay=0.02,
                  threshold_source="score_file"):
    """Yield SSE frames: one meta frame, then one per timestep, then done.

    Any exception mid-stream is surfaced to the client as an {"error": ...}
    frame instead of silently severing the connection.
    """
    labels = np.asarray(labels).ravel()
    segments = _find_segments(labels)

    yield _sse({"meta": True, "threshold": float(threshold),
                "threshold_source": threshold_source,
                "n_segments": len(segments), "n_timesteps": len(test)})

    is_lstm = getattr(detector, "name", "") == "lstm_autoencoder"
    window_size = getattr(detector, "window_size", 1) if is_lstm else 1
    buffer = deque(maxlen=window_size) if is_lstm else None

    for t in range(len(test)):
        try:
            score = _score_step(detector, test[t], is_lstm, buffer, window_size)
        except Exception as e:  # noqa: BLE001
            yield _sse({"error": f"scoring failed at t={t}: {e}"})
            return

        yield _sse({"t": t, "score": score, "label": int(labels[t])})
        if delay:
            time.sleep(delay)

    yield _sse({"done": True})