import json
import sys
import time
from pathlib import Path

import numpy as np

from src.data_loader import get_machine_ids, load_machine
from models.pca_detector import PCADetector
from evaluation.metrics import evaluate_detector

RESULTS_DIR = Path("results")
SCORES_DIR = RESULTS_DIR / "scores"


def get_detectors():
    """Registry of detectors to run. Add new detectors here."""
    return [
        PCADetector(),
        # ZScoreDetector(),        <- next up
        # IForestDetector(),
        # LSTMAEDetector(),
    ]


def run_machine(machine_id: str) -> dict:
    train, test, labels = load_machine(machine_id)
    machine_results = {}

    for det in get_detectors():
        t0 = time.perf_counter()
        det.fit(train)
        fit_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = det.score(test)
        score_seconds = time.perf_counter() - t0

        result = evaluate_detector(labels, scores)
        result["fit_seconds"] = round(fit_seconds, 4)
        result["score_seconds"] = round(score_seconds, 4)
        result["throughput_pts_per_sec"] = round(len(test) / max(score_seconds, 1e-9))

        machine_results[det.name] = result

        SCORES_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            SCORES_DIR / f"{machine_id}_{det.name}.npz",
            scores=scores,
            labels=labels,
            threshold=result["threshold"],
        )

        h = result["honest"]["f1"]
        a = result["point_adjusted"]["f1"]
        print(f"  {det.name:12s}  honest F1={h:.3f}  adjusted F1={a:.3f}  "
              f"PR-AUC={result['pr_auc']:.3f}  ({result['throughput_pts_per_sec']:,} pts/s)")

    out_path = RESULTS_DIR / f"{machine_id}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(machine_results, f, indent=2)
    print(f"  saved -> {out_path}")

    return machine_results


def main():
    args = sys.argv[1:]
    if "--all" in args:
        machine_ids = get_machine_ids()
    elif args:
        machine_ids = args
    else:
        machine_ids = ["machine-1-1"]

    for mid in machine_ids:
        print(f"\n=== {mid} ===")
        run_machine(mid)


if __name__ == "__main__":
    main()