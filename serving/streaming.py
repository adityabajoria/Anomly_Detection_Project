import json
import time
from collections import deque
from pathlib import Path

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


def stream_scores(detector, test, labels, threshold, delay=0.02):
    """Yield SSE frames: one meta frame, then one per timestep, then done.

    threshold: the decision threshold from the offline experiment (passed in by
    the route, read from the saved score file), so 'flagged' here means exactly
    what it means in the benchmark.
    """
    segments = _find_segments(labels)

    yield _sse({"meta": True, "threshold": float(threshold),
                "n_segments": len(segments), "n_timesteps": len(test)})

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

        yield _sse({"t": t, "score": score, "label": int(labels[t])})
        if delay:
            time.sleep(delay)

    yield _sse({"done": True})