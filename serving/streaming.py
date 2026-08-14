import json
import time
from collections import deque
from .session import StreamSession

import numpy as np


def _sse(data):
    return f"data: {json.dumps(data)}\n\n"

def _score_step(detector, row, detector_name, session: StreamSession):
    """
    Scores a single point using the active detector.
    Returns:
        - score: float
        - latency_ms: float
        - warmup: bool
    """
    is_lstm = detector_name == "lstm_autoencoder"
    if is_lstm:
        window_size = detector.window_size
        buffer = session.get_buffer(
            detector_name,
            maxlen=window_size
        )
        buffer.append(row)

        if len(buffer) < window_size:
            return None, None, True

        X = np.asarray(buffer, dtype=float)
    else:
        X = np.asarray(row, dtype=float)[None, :]

    start = time.perf_counter()
    scores = detector.score(X)
    end = time.perf_counter()

    latency_ms = (
        end - start
    ) * 1000.0

    score = float(np.asarray(scores).ravel()[-1])

    return score, latency_ms, False

def stream_session(
    session: StreamSession,
    test,
    registry,
    threshold_resolver,
    delay=0.02,
):
    """
    Stream telemetry from the session's current timestep.

    The telemetry cursor belongs to StreamSession, not to a detector.

    Changing session.active_detector therefore changes the model
    used for future events without resetting current_timestep.
    """

    test = np.asarray(
        test,
        dtype=float,
    )

    n_timesteps = len(test)

    yield _sse({
        "meta": True,
        "session_id": session.session_id,
        "machine_id": session.machine_id,
        "active_detector": session.active_detector,
        "current_timestep": session.current_timestep,
        "n_timesteps": n_timesteps,
    })

    while (
        session.current_timestep
        < n_timesteps
        and not session.stopped
    ):

        t = session.current_timestep

        detector_name = (
            session.active_detector
        )

        detector = registry.get(
            (
                session.machine_id,
                detector_name,
            )
        )

        if detector is None:
            yield _sse({
                "error": (
                    f"No detector '{detector_name}' "
                    f"loaded for machine "
                    f"'{session.machine_id}'"
                )
            })
            return

        try:
            model_threshold, threshold_source = (
                threshold_resolver(
                    session.machine_id,
                    detector_name,
                )
            )

            override = (
                session.get_threshold_override(
                    detector_name
                )
            )

            if override is not None:
                active_threshold = override
                threshold_source = "manual_override"
            else:
                active_threshold = model_threshold

        except Exception as e:
            yield _sse({
                "error": (
                    f"threshold resolution failed "
                    f"at t={t}: {e}"
                )
            })
            return

        row = test[t]

        try:
            (
                score,
                latency_ms,
                warmup,
            ) = _score_step(
                detector,
                row,
                detector_name,
                session,
            )

        except Exception as e:
            yield _sse({
                "error": (
                    f"scoring failed at t={t} "
                    f"with detector "
                    f"'{detector_name}': {e}"
                )
            })
            return

        is_alert = (
            score is not None
            and score >= active_threshold
        )

        yield _sse({
            "t": t,
            "detector": detector_name,
            "score": score,
            "threshold": float(
                active_threshold
            ),
            "threshold_source": (
                threshold_source
            ),
            "is_alert": is_alert,
            "latency_ms": latency_ms,
            "warmup": warmup,
        })

        session.advance()

        if delay:
            time.sleep(delay)

    yield _sse({
        "done": True,
        "session_id": session.session_id,
        "current_timestep": session.current_timestep,
    })
