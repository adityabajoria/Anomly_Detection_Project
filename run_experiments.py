import hydra
import json
import time
import numpy as np
from pathlib import Path
from src.data_loader import get_machine_ids, load_machine
from models.pca_detector import PCADetector
from models.z_detector import ZScoreDetector
from evaluation.metrics import evaluate_detector
from omegaconf import DictConfig


def get_detectors():
    """Registry of detectors to run. Add new detectors here."""
    return [
        PCADetector(),
        ZScoreDetector(),
        # IForestDetector(), # next up
        # LSTMAEDetector(),
    ]


def run_machine(machine_id, cfg):
    results_dir = Path(cfg.results_dir)
    scores_dir = results_dir / "scores"
    train, test, labels = load_machine(machine_id, data_dir=cfg.data_dir)
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

        scores_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scores_dir / f"{machine_id}_{det.name}.npz",
            scores=scores,
            labels=labels,
            threshold=result["threshold"],
        )

        h = result["honest"]["f1"]
        a = result["point_adjusted"]["f1"]
        print(f"  {det.name:12s}  honest F1={h:.3f}  adjusted F1={a:.3f}  "
              f"PR-AUC={result['pr_auc']:.3f}  ({result['throughput_pts_per_sec']:,} pts/s)")

    out_path = results_dir / f"{machine_id}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(machine_results, f, indent=2)
    print(f"  saved -> {out_path}")

    return machine_results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.machines == "all":
        machine_ids = get_machine_ids(cfg.data_dir)
    else:
        machine_ids = list(cfg.machines)

    for mid in machine_ids:
        print(f"\n=== {mid} ===")
        run_machine(mid, cfg)


if __name__ == "__main__":
    main()