"""Sentinel — unified FastAPI application (package version).

Run from anywhere (paths are anchored to the repo root, not the cwd):
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

# BUGFIX: anchor every path to the repo root instead of the process cwd.
# With the old relative paths, launching uvicorn from any directory other
# than the repo root made artifacts/results silently "disappear".
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SCORES_DIR = REPO_ROOT / "results" / "scores"
STATIC_DIR = REPO_ROOT / "serving" / "static"

REGISTRY: dict = {}


def _discover_and_load():
    REGISTRY.clear()
    if not ARTIFACTS_DIR.exists():
        print(f"[startup] WARNING: no artifacts dir at {ARTIFACTS_DIR}")
        return
    for machine_dir in sorted(ARTIFACTS_DIR.iterdir()):
        if not machine_dir.is_dir():
            continue
        for det_dir in sorted(machine_dir.iterdir()):
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


app = FastAPI(title="Sentinel", version="2.1.0", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"status": "ok", "detectors_loaded": len(REGISTRY),
            "machines": len({k[0] for k in REGISTRY})}


@app.get("/api/machines")
def machines():
    return {"machines": data.list_machines()}


@app.get("/api/detectors/{machine_id}")
def detectors(machine_id: str):
    """Detectors actually loaded in the registry for this machine.

    The frontend previously inferred the detector list from the results
    JSON; if that file was missing/malformed the dropdown ended up empty
    with no explanation. This endpoint reports the loaded models directly.
    """
    dets = sorted(d for (m, d) in REGISTRY if m == machine_id)
    if not dets:
        raise HTTPException(404, f"No detectors loaded for machine '{machine_id}'")
    return {"machine_id": machine_id, "detectors": dets}


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


def _resolve_threshold(machine_id: str, detector: str) -> tuple[float, str]:
    """Find the decision threshold, with graceful fallbacks.

    BUGFIX: the old code did `np.load(path)["threshold"]` — a KeyError
    (score file saved without the threshold key) became an opaque 500
    before the stream even started, and a missing file silently fell back
    to 0.0, which flags every single timestep.

    Resolution order:
      1. "threshold" key in results/scores/{machine}_{detector}.npz
      2. "threshold" recorded in results/{machine}.json for this detector
      3. 99th percentile of the saved offline scores (labelled as such)
    """
    score_path = SCORES_DIR / f"{machine_id}_{detector}.npz"
    if score_path.exists():
        npz = np.load(score_path)
        if "threshold" in npz.files:
            return float(npz["threshold"]), "score_file"
        if "scores" in npz.files:
            fallback = float(np.quantile(npz["scores"], 0.99))
        else:
            fallback = None
    else:
        fallback = None

    try:
        res = data.machine_results(machine_id).get(detector, {})
        if "threshold" in res:
            return float(res["threshold"]), "results_json"
    except FileNotFoundError:
        pass

    if fallback is not None:
        return fallback, "p99_of_offline_scores"
    raise HTTPException(
        404,
        f"No threshold available for {machine_id}/{detector}: expected a "
        f"'threshold' key in {score_path.name} or in results/{machine_id}.json",
    )


@app.get("/api/stream/{machine_id}/{detector}")
def stream(machine_id: str, detector: str, delay: float = 0.02):
    det = REGISTRY.get((machine_id, detector))
    if det is None:
        raise HTTPException(404, f"No '{detector}' for machine '{machine_id}'")
    test_path = SCORES_DIR / f"{machine_id}_test.npz"
    if not test_path.exists():
        raise HTTPException(404, f"No test set on disk for '{machine_id}' "
                                 f"(expected {test_path})")
    npz = np.load(test_path)
    threshold, source = _resolve_threshold(machine_id, detector)
    return StreamingResponse(
        stream_scores(det, npz["test"], npz["labels"], threshold,
                      delay=max(0.0, min(delay, 0.2)), threshold_source=source),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")