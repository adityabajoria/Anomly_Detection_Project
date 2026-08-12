"""Sentinel — unified FastAPI application (package version).

Lives at serving/service.py. Run from repo root:
    uvicorn serving.service:app --reload
"""
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data
from .persistence import load_detector
from .streaming import stream_scores

ARTIFACTS_DIR = Path("artifacts")
SCORES_DIR = Path("results/scores")
STATIC_DIR = Path("serving/static")

REGISTRY: dict = {}


def _discover_and_load():
    REGISTRY.clear()
    if not ARTIFACTS_DIR.exists():
        return
    for machine_dir in ARTIFACTS_DIR.iterdir():
        if not machine_dir.is_dir():
            continue
        for det_dir in machine_dir.iterdir():
            if not det_dir.is_dir():
                continue
            try:
                REGISTRY[(machine_dir.name, det_dir.name)] = load_detector(det_dir.name, det_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[startup] failed to load {machine_dir.name}/{det_dir.name}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _discover_and_load()
    print(f"[startup] loaded {len(REGISTRY)} detectors across "
          f"{len({k[0] for k in REGISTRY})} machines")
    yield
    REGISTRY.clear()


app = FastAPI(title="Sentinel", version="2.0.0", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"status": "ok", "detectors_loaded": len(REGISTRY)}


@app.get("/api/machines")
def machines():
    return {"machines": data.list_machines()}


@app.get("/api/results/{machine_id}")
def results(machine_id: str):
    try:
        return data.machine_results(machine_id)
    except FileNotFoundError:
        raise HTTPException(404, f"No results for machine '{machine_id}'")


@app.get("/api/inflation")
def inflation():
    return {"rows": data.inflation_table()}


class ScoreRequest(BaseModel):
    machine_id: str
    detector: str
    window: list[list[float]] = Field(..., description="(n_timesteps, n_features)")


@app.post("/api/score")
def score(req: ScoreRequest):
    det = REGISTRY.get((req.machine_id, req.detector))
    if det is None:
        raise HTTPException(404, f"No '{req.detector}' for machine '{req.machine_id}'")
    X = np.asarray(req.window, dtype=float)
    if X.ndim != 2:
        raise HTTPException(400, "window must be 2-D (n_timesteps, n_features)")
    min_w = det.window_size if req.detector == "lstm_autoencoder" else 1
    if len(X) < min_w:
        raise HTTPException(400, f"'{req.detector}' needs >= {min_w} timesteps, got {len(X)}")
    try:
        scores = det.score(X)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"scoring failed: {e}") from e
    return {"machine_id": req.machine_id, "detector": req.detector,
            "scores": [float(s) for s in np.asarray(scores).ravel()]}


@app.get("/api/stream/{machine_id}/{detector}")
def stream(machine_id: str, detector: str, delay: float = 0.02):
    det = REGISTRY.get((machine_id, detector))
    if det is None:
        raise HTTPException(404, f"No '{detector}' for machine '{machine_id}'")
    test_path = SCORES_DIR / f"{machine_id}_test.npz"
    if not test_path.exists():
        raise HTTPException(404, f"No test set on disk for '{machine_id}'")
    npz = np.load(test_path)
    # use the threshold the offline experiment already computed for this detector,
    # so live detections match the benchmark numbers
    score_path = SCORES_DIR / f"{machine_id}_{detector}.npz"
    threshold = float(np.load(score_path)["threshold"]) if score_path.exists() else 0.0
    return StreamingResponse(
        stream_scores(det, npz["test"], npz["labels"], threshold, delay=max(0.0, min(delay, 0.2))),
        media_type="text/event-stream",
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")