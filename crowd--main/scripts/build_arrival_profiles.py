"""
Build venue arrival profiles from real pedestrian sensor data.

Source: City of Melbourne Pedestrian Counting System (CC BY 4.0), via the
MONSTER benchmark mirror `monster-monash/Pedestrian` on HuggingFace.
    https://huggingface.co/datasets/monster-monash/Pedestrian
    https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour/information/

The MONSTER release is preprocessed for a *classification* task, so it carries
no calendar dates -- only (189621, 1, 24) hourly count vectors labelled by
sensor id. Two properties recovered by inspection make it usable anyway:

  1. Index 0 of each vector is midnight. The pooled mean profile troughs at
     04:00 and peaks at 08:00 / 13:00 / 17:00, which is a textbook urban day.

  2. Chronological order survives in the stored row order. Daily totals show
     lag-7 autocorrelation of +0.87 and lag-14 of +0.84, so `row_index % 7`
     recovers day-of-week. The two phases with ~4.5x lower volume are the
     weekend.

Three sensors are used as venue zone archetypes, chosen by profile shape:

    gate       sensor 12   65.8% of mass in commute peaks -> sharp bimodal surge
    concourse  sensor 40   31,994/day, flat               -> sustained all-day load
    food       sensor 67   44.6% of mass at midday        -> lunch-driven

Output: backend/data/arrival_profiles.json -- normalised 24-hour curves that
sum to 1.0, so they scale to any venue attendance figure.

Usage:
    python scripts/build_arrival_profiles.py [--x PX.npy --y Py.npy]

With no arguments the arrays are downloaded from HuggingFace (~20MB).
"""

import argparse
import json
import os
import sys

import numpy as np

# Sensor id -> venue zone archetype. See module docstring for selection basis.
ARCHETYPES = {
    "gate": 12,
    "concourse": 40,
    "food": 67,
}

WEEKDAY_PHASES = [1, 2, 3, 4, 5]
WEEKEND_PHASES = [0, 6]

REPO = "monster-monash/Pedestrian"
OUT_PATH = os.path.join("backend", "data", "arrival_profiles.json")


def load_arrays(x_path: str | None, y_path: str | None):
    """Load the count matrix and sensor labels, downloading if not supplied."""
    if x_path and y_path:
        return np.load(x_path), np.load(y_path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "huggingface-hub is not installed and no local arrays were given.\n"
            "  pip install huggingface-hub\n"
            "or pass --x Pedestrian_X.npy --y Pedestrian_y.npy"
        )

    print(f"Downloading {REPO} (~20MB)...")
    x = np.load(hf_hub_download(REPO, "Pedestrian_X.npy", repo_type="dataset"))
    y = np.load(hf_hub_download(REPO, "Pedestrian_y.npy", repo_type="dataset"))
    return x, y


def build_profile(counts: np.ndarray, phases: list[int]) -> tuple[list[float], float]:
    """
    Return a normalised 24-hour arrival curve plus the mean daily total.

    `counts` is (n_days, 24) for a single sensor in chronological order.
    `phases` selects weekday or weekend rows via row_index % 7.
    """
    rows = np.concatenate([counts[p::7] for p in phases])
    hourly_mean = rows.mean(axis=0)
    curve = hourly_mean / hourly_mean.sum()
    return [round(float(v), 6) for v in curve], float(rows.sum(axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", help="path to Pedestrian_X.npy")
    parser.add_argument("--y", help="path to Pedestrian_y.npy")
    parser.add_argument("--out", default=OUT_PATH, help="output json path")
    args = parser.parse_args()

    X, y = load_arrays(args.x, args.y)
    X = X[:, 0, :]  # (n, 1, 24) -> (n, 24)
    print(f"Loaded {X.shape[0]:,} daily profiles across {len(np.unique(y))} sensors.")

    archetypes = {}
    for name, sensor in ARCHETYPES.items():
        counts = X[y == sensor]
        weekday, weekday_total = build_profile(counts, WEEKDAY_PHASES)
        weekend, weekend_total = build_profile(counts, WEEKEND_PHASES)

        archetypes[name] = {
            "sensor_id": int(sensor),
            "days_observed": int(counts.shape[0]),
            "weekday": weekday,
            "weekend": weekend,
            "mean_daily_total_weekday": round(weekday_total, 1),
            "mean_daily_total_weekend": round(weekend_total, 1),
            "peak_hour_weekday": int(np.argmax(weekday)),
            "peak_share_weekday": round(max(weekday), 4),
        }
        print(
            f"  {name:10s} sensor {sensor:2d}  "
            f"{counts.shape[0]:5d} days  "
            f"peak {np.argmax(weekday):02d}:00 at {max(weekday) * 100:.1f}% of daily volume"
        )

    payload = {
        "source": "City of Melbourne Pedestrian Counting System",
        "license": "CC BY 4.0",
        "mirror": f"https://huggingface.co/datasets/{REPO}",
        "attribution": (
            "City of Melbourne (2022). Pedestrian counting system. "
            "https://data.melbourne.vic.gov.au/ -- CC BY 4.0"
        ),
        "note": (
            "Curves are normalised to sum to 1.0 across 24 hours. Multiply by "
            "expected attendance to get people per hour. Hour index 0 = midnight."
        ),
        "archetypes": archetypes,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
