import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")


def main():
    # collect metric lists per detector across all machines
    per_detector = defaultdict(lambda: defaultdict(list))
    machine_files = sorted(RESULTS_DIR.glob("machine-*.json"))

    for path in machine_files:
        with open(path) as f:
            machine_results = json.load(f)
        for det_name, r in machine_results.items():
            d = per_detector[det_name]
            d["honest_f1"].append(r["honest"]["f1"])
            d["adjusted_f1"].append(r["point_adjusted"]["f1"])
            d["pr_auc"].append(r["pr_auc"])
            d["throughput"].append(r["throughput_pts_per_sec"])

    n_machines = len(machine_files)
    print(f"\nAggregated over {n_machines} machines\n")
    header = (f"{'detector':<12} {'honest F1':>16} {'adjusted F1':>16} "
              f"{'PR-AUC':>16} {'median pts/s':>14}")
    print(header)
    print("-" * len(header))

    summary = {}
    for det_name, d in per_detector.items():
        h, a, p = map(np.array, (d["honest_f1"], d["adjusted_f1"], d["pr_auc"]))
        row = {
            "n_machines": n_machines,
            "honest_f1_mean": float(h.mean()), "honest_f1_std": float(h.std()),
            "adjusted_f1_mean": float(a.mean()), "adjusted_f1_std": float(a.std()),
            "pr_auc_mean": float(p.mean()), "pr_auc_std": float(p.std()),
            "throughput_median": float(np.median(d["throughput"])),
        }
        summary[det_name] = row
        print(f"{det_name:<12} "
              f"{row['honest_f1_mean']:.3f} +/- {row['honest_f1_std']:.3f}   "
              f"{row['adjusted_f1_mean']:.3f} +/- {row['adjusted_f1_std']:.3f}   "
              f"{row['pr_auc_mean']:.3f} +/- {row['pr_auc_std']:.3f}   "
              f"{row['throughput_median']:>12,.0f}")

    # biggest honest-vs-adjusted gaps: where point-adjustment flatters most
    print("\nLargest point-adjustment inflation (per machine):")
    gaps = []
    for path in machine_files:
        with open(path) as f:
            machine_results = json.load(f)
        for det_name, r in machine_results.items():
            gaps.append((r["point_adjusted"]["f1"] - r["honest"]["f1"],
                         path.stem, det_name,
                         r["honest"]["f1"], r["point_adjusted"]["f1"]))
    for gap, machine, det, h, a in sorted(gaps, reverse=True)[:5]:
        print(f"  {machine:<16} {det:<10} honest={h:.3f}  adjusted={a:.3f}  (+{gap:.3f})")

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()