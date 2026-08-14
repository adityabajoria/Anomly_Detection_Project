"""Sentinel — unified FastAPI application.

Run from anywhere:
    uvicorn serving.service:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data
from .persistence import load_detector
from .session import StreamSession
from .streaming import stream_session


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SCORES_DIR = REPO_ROOT / "results" / "scores"
STATIC_DIR = REPO_ROOT / "serving" / "static"


# ============================================================
# Registries
# ============================================================

# Loaded ML models:
#
# {
#     ("machine-1-1", "pca"): <PCADetector>,
#     ("machine-1-1", "lstm_autoencoder"): <LSTMAutoencoderDetector>,
# }
REGISTRY: dict = {}


# Persistent telemetry sessions:
#
# {
#     "abc123": StreamSession(...)
# }
SESSIONS: dict[str, StreamSession] = {}


# ============================================================
# Model Discovery / Startup
# ============================================================

def _discover_and_load():
    REGISTRY.clear()

    if not ARTIFACTS_DIR.exists():
        print(
            f"[startup] WARNING: no artifacts dir "
            f"at {ARTIFACTS_DIR}"
        )
        return

    for machine_dir in sorted(
        ARTIFACTS_DIR.iterdir()
    ):

        if not machine_dir.is_dir():
            continue

        for det_dir in sorted(
            machine_dir.iterdir()
        ):

            if not det_dir.is_dir():
                continue

            try:
                REGISTRY[
                    (
                        machine_dir.name,
                        det_dir.name,
                    )
                ] = load_detector(
                    det_dir.name,
                    det_dir,
                )

            except Exception as e:
                print(
                    "[startup] failed to load "
                    f"{machine_dir.name}/"
                    f"{det_dir.name}: {e}"
                )


@asynccontextmanager
async def lifespan(app: FastAPI):

    _discover_and_load()

    machine_count = len(
        {
            machine_id
            for machine_id, _
            in REGISTRY
        }
    )

    print(
        f"[startup] loaded {len(REGISTRY)} "
        f"detectors across {machine_count} machines"
    )

    yield

    SESSIONS.clear()
    REGISTRY.clear()


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Sentinel",
    version="3.0.0",
    lifespan=lifespan,
)


# ============================================================
# Health / Discovery
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "detectors_loaded": len(REGISTRY),
        "machines": len(
            {
                machine_id
                for machine_id, _
                in REGISTRY
            }
        ),
        "active_sessions": len(SESSIONS),
    }


@app.get("/api/machines")
def machines():

    return {
        "machines": data.list_machines()
    }


@app.get(
    "/api/detectors/{machine_id}"
)
def detectors(machine_id: str):

    dets = sorted(
        detector
        for machine, detector
        in REGISTRY
        if machine == machine_id
    )

    if not dets:
        raise HTTPException(
            404,
            (
                f"No detectors loaded for "
                f"machine '{machine_id}'"
            ),
        )

    return {
        "machine_id": machine_id,
        "detectors": dets,
    }


@app.get(
    "/api/results/{machine_id}"
)
def results(machine_id: str):

    try:
        return data.machine_results(
            machine_id
        )

    except FileNotFoundError:
        raise HTTPException(
            404,
            f"No results for machine '{machine_id}'",
        )


@app.get("/api/inflation")
def inflation():

    return {
        "rows": data.inflation_table()
    }


# ============================================================
# One-Off Scoring API
# ============================================================

class ScoreRequest(BaseModel):
    machine_id: str
    detector: str

    window: list[list[float]] = Field(
        ...,
        description=(
            "(n_timesteps, n_features)"
        ),
    )


@app.post("/api/score")
def score(req: ScoreRequest):

    det = REGISTRY.get(
        (
            req.machine_id,
            req.detector,
        )
    )

    if det is None:
        raise HTTPException(
            404,
            (
                f"No '{req.detector}' for "
                f"machine '{req.machine_id}'"
            ),
        )

    X = np.asarray(
        req.window,
        dtype=float,
    )

    if X.ndim != 2:
        raise HTTPException(
            400,
            (
                "window must be 2-D "
                "(n_timesteps, n_features)"
            ),
        )

    min_window = (
        det.window_size
        if req.detector
        == "lstm_autoencoder"
        else 1
    )

    if len(X) < min_window:
        raise HTTPException(
            400,
            (
                f"'{req.detector}' needs >= "
                f"{min_window} timesteps, "
                f"got {len(X)}"
            ),
        )

    try:
        scores = det.score(X)

    except Exception as e:
        raise HTTPException(
            400,
            f"scoring failed: {e}",
        ) from e

    return {
        "machine_id": req.machine_id,
        "detector": req.detector,
        "scores": [
            float(score)
            for score
            in np.asarray(scores).ravel()
        ],
    }


# ============================================================
# Threshold Resolution
# ============================================================

def _resolve_threshold(
    machine_id: str,
    detector: str,
) -> tuple[float, str]:

    """
    Resolve the calibrated threshold for a detector.

    Resolution order:
      1. threshold saved in scores NPZ
      2. threshold stored in results JSON
      3. P99 of saved offline scores
    """

    score_path = (
        SCORES_DIR /
        f"{machine_id}_{detector}.npz"
    )


    fallback = None


    if score_path.exists():

        npz = np.load(
            score_path
        )

        if "threshold" in npz.files:

            return (
                float(
                    npz["threshold"]
                ),
                "score_file",
            )


        if "scores" in npz.files:

            fallback = float(
                np.quantile(
                    npz["scores"],
                    0.99,
                )
            )


    try:

        res = data.machine_results(
            machine_id
        ).get(
            detector,
            {},
        )


        if "threshold" in res:

            return (
                float(
                    res["threshold"]
                ),
                "results_json",
            )


    except FileNotFoundError:
        pass


    if fallback is not None:

        return (
            fallback,
            "p99_of_offline_scores",
        )


    raise HTTPException(
        404,
        (
            f"No threshold available for "
            f"{machine_id}/{detector}: "
            f"expected a 'threshold' key in "
            f"{score_path.name} or in "
            f"results/{machine_id}.json"
        ),
    )


# ============================================================
# Session Request Models
# ============================================================

class CreateSessionRequest(BaseModel):
    machine_id: str
    active_detector: str


class SwitchDetectorRequest(BaseModel):
    detector: str


class ThresholdRequest(BaseModel):
    detector: str
    threshold: float


# ============================================================
# Session Helpers
# ============================================================

def _get_session(
    session_id: str,
) -> StreamSession:

    session = SESSIONS.get(
        session_id
    )

    if session is None:
        raise HTTPException(
            404,
            f"Unknown session '{session_id}'",
        )

    return session


# ============================================================
# Session Creation
# ============================================================

@app.post("/api/sessions")
def create_session(
    req: CreateSessionRequest,
):

    detector = REGISTRY.get(
        (
            req.machine_id,
            req.active_detector,
        )
    )


    if detector is None:
        raise HTTPException(
            404,
            (
                f"No '{req.active_detector}' "
                f"detector loaded for machine "
                f"'{req.machine_id}'"
            ),
        )


    session = StreamSession(
        machine_id=req.machine_id,
        active_detector=req.active_detector,
    )


    SESSIONS[
        session.session_id
    ] = session


    return {
        "session_id": session.session_id,
        "machine_id": session.machine_id,
        "active_detector": (
            session.active_detector
        ),
        "current_timestep": (
            session.current_timestep
        ),
    }


# ============================================================
# Session Status
# ============================================================

@app.get(
    "/api/sessions/{session_id}"
)
def session_status(
    session_id: str,
):

    session = _get_session(
        session_id
    )


    return {
        "session_id": (
            session.session_id
        ),

        "machine_id": (
            session.machine_id
        ),

        "active_detector": (
            session.active_detector
        ),

        "current_timestep": (
            session.current_timestep
        ),

        "switch_history": (
            session.switch_history
        ),

        "threshold_overrides": (
            session.threshold_overrides
        ),
    }


# ============================================================
# Hot-Swap Active Model
# ============================================================

@app.post(
    "/api/sessions/{session_id}/model"
)
def switch_detector(
    session_id: str,
    req: SwitchDetectorRequest,
):

    session = _get_session(
        session_id
    )


    detector = REGISTRY.get(
        (
            session.machine_id,
            req.detector,
        )
    )


    if detector is None:
        raise HTTPException(
            404,
            (
                f"No '{req.detector}' detector "
                f"loaded for machine "
                f"'{session.machine_id}'"
            ),
        )


    previous = (
        session.active_detector
    )


    session.switch_detector(
        req.detector
    )


    return {
        "session_id": (
            session.session_id
        ),

        "previous_detector": (
            previous
        ),

        "active_detector": (
            session.active_detector
        ),

        "current_timestep": (
            session.current_timestep
        ),
    }


# ============================================================
# Threshold Overrides
# ============================================================

@app.post(
    "/api/sessions/{session_id}/threshold"
)
def set_session_threshold(
    session_id: str,
    req: ThresholdRequest,
):

    session = _get_session(
        session_id
    )


    detector = REGISTRY.get(
        (
            session.machine_id,
            req.detector,
        )
    )


    if detector is None:
        raise HTTPException(
            404,
            (
                f"No '{req.detector}' detector "
                f"loaded for machine "
                f"'{session.machine_id}'"
            ),
        )


    session.set_threshold(
        req.detector,
        req.threshold,
    )


    return {
        "session_id": (
            session.session_id
        ),

        "detector": (
            req.detector
        ),

        "threshold": (
            req.threshold
        ),

        "source": (
            "manual_override"
        ),
    }


@app.delete(
    "/api/sessions/{session_id}/threshold/{detector}"
)
def clear_session_threshold(
    session_id: str,
    detector: str,
):

    session = _get_session(
        session_id
    )


    loaded_detector = REGISTRY.get(
        (
            session.machine_id,
            detector,
        )
    )


    if loaded_detector is None:
        raise HTTPException(
            404,
            (
                f"No '{detector}' detector "
                f"loaded for machine "
                f"'{session.machine_id}'"
            ),
        )


    session.clear_threshold(
        detector
    )


    model_threshold, source = (
        _resolve_threshold(
            session.machine_id,
            detector,
        )
    )


    return {
        "session_id": (
            session.session_id
        ),

        "detector": (
            detector
        ),

        "threshold": (
            model_threshold
        ),

        "source": (
            source
        ),
    }


# ============================================================
# Persistent Session Stream
# ============================================================

@app.get(
    "/api/sessions/{session_id}/stream"
)
def session_stream(
    session_id: str,
    delay: float = 0.02,
):

    session = _get_session(
        session_id
    )


    test_path = (
        SCORES_DIR /
        f"{session.machine_id}_test.npz"
    )


    if not test_path.exists():

        raise HTTPException(
            404,
            (
                f"No test set on disk for "
                f"'{session.machine_id}' "
                f"(expected {test_path})"
            ),
        )


    npz = np.load(
        test_path
    )


    return StreamingResponse(
        stream_session(
            session=session,
            test=npz["test"],
            registry=REGISTRY,
            threshold_resolver=(
                _resolve_threshold
            ),
            delay=max(
                0.0,
                min(
                    delay,
                    0.2,
                ),
            ),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no",
        },
    )


# ============================================================
# Legacy Stream Compatibility
# ============================================================

@app.get(
    "/api/stream/{machine_id}/{detector}"
)
def stream(
    machine_id: str,
    detector: str,
    delay: float = 0.02,
):

    det = REGISTRY.get(
        (
            machine_id,
            detector,
        )
    )


    if det is None:

        raise HTTPException(
            404,
            (
                f"No '{detector}' for "
                f"machine '{machine_id}'"
            ),
        )


    test_path = (
        SCORES_DIR /
        f"{machine_id}_test.npz"
    )


    if not test_path.exists():

        raise HTTPException(
            404,
            (
                f"No test set on disk for "
                f"'{machine_id}' "
                f"(expected {test_path})"
            ),
        )


    npz = np.load(
        test_path
    )


    temporary_session = (
        StreamSession(
            machine_id=machine_id,
            active_detector=detector,
        )
    )


    return StreamingResponse(
        stream_session(
            session=temporary_session,
            test=npz["test"],
            registry=REGISTRY,
            threshold_resolver=(
                _resolve_threshold
            ),
            delay=max(
                0.0,
                min(
                    delay,
                    0.2,
                ),
            ),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no",
        },
    )


# ============================================================
# Static Frontend
# ============================================================

@app.get("/")
def index():

    return FileResponse(
        STATIC_DIR /
        "index.html"
    )


if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(
                STATIC_DIR
            )
        ),
        name="static",
    )