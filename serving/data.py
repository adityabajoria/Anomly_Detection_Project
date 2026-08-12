import json
import re
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")

MODEL_ORDER = ["random", "zscore", "pca", "iforest", "lstm_autoencoder"]
MODEL_LABELS = {
    "random": "Random (baseline)",
    "zscore": "Z-Score",
    "pca": "PCA",
    "iforest": "Isolation Forest",
    "lstm_autoencoder": "LSTM Autoencoder",
}


def _machine_sort_key(name: str):
    nums = re.findall(r"\d+", name)
    return tuple(int(n) for n in nums) if nums else (0,)


def list_machines() -> list[str]:
    return sorted(
        (p.stem for p in RESULTS_DIR.glob("machine-*.json")),
        key=_machine_sort_key,
    )


def machine_results(machine_id: str) -> dict:
    path = RESULTS_DIR / f"{machine_id}.json"
    if not path.exists():
        raise FileNotFoundError(machine_id)
    return json.loads(path.read_text())


def inflation_table() -> list[dict]:
    """Fleet-level per-detector honest vs point-adjusted F1 and the gap."""
    per_detector: dict = {}
    for p in RESULTS_DIR.glob("machine-*.json"):
        data = json.loads(p.read_text())
        for det, res in data.items():
            h = res["honest"]["f1"]
            a = res["point_adjusted"]["f1"]
            per_detector.setdefault(det, {"honest": [], "adjusted": []})
            per_detector[det]["honest"].append(h)
            per_detector[det]["adjusted"].append(a)

    rows = []
    for det, vals in per_detector.items():
        honest = np.array(vals["honest"])
        adjusted = np.array(vals["adjusted"])
        rows.append({
            "detector": det,
            "label": MODEL_LABELS.get(det, det),
            "honest_mean": float(honest.mean()),
            "adjusted_mean": float(adjusted.mean()),
            "inflation": float((adjusted - honest).mean()),
            "n": len(honest),
        })
    rows.sort(key=lambda r: MODEL_ORDER.index(r["detector"])
              if r["detector"] in MODEL_ORDER else len(MODEL_ORDER))
    return rows