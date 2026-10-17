"""Build the hourly modelling table from the files fetch_data.py saved.

Usage: uv run python scripts/build_features.py [--horizon 6] [--merge-gap 6h]
"""

import argparse

import pandas as pd

from triton.constants import PROCESSED_DIR, RAW_DIR
from triton.features import build_panel, merge_episodes, target_name


def main(horizon, merge_gap):
    sites = pd.read_csv(RAW_DIR / "sites_selected.csv.gz")
    events = pd.read_csv(RAW_DIR / "spills.csv.gz")
    for col in ("start", "end"):
        events[col] = pd.to_datetime(events[col], utc=True, format="ISO8601")
    events = events[events["site_id"].isin(sites["site_id"])]
    rain = pd.read_csv(RAW_DIR / "rain_hourly.csv.gz")
    rain["timestamp"] = pd.to_datetime(rain["timestamp"], utc=True, format="ISO8601")

    episodes = merge_episodes(events, merge_gap)
    print(f"{len(events)} spill events -> {len(episodes)} episodes (gap {merge_gap})")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(PROCESSED_DIR / "episodes.csv.gz", index=False)

    panel = build_panel(episodes, rain, horizon=horizon)
    path = PROCESSED_DIR / "model_table.csv.gz"
    panel.to_csv(path, index=False)

    target = target_name(horizon)
    at_risk = panel[panel["spilling_now"] == 0]
    print(f"Saved {path.name}: {len(panel)} rows, {panel['site_id'].nunique()} sites")
    print(f"  {panel['timestamp'].min()} to {panel['timestamp'].max()}")
    print(f"Class balance for {target}:")
    print(f"  all rows:         {panel[target].mean():.2%} positive")
    print(f"  not spilling now: {at_risk[target].mean():.2%} positive")
    by_year = panel.groupby(panel["timestamp"].dt.year)[target].agg(["size", "mean"])
    print(by_year.rename(columns={"size": "rows", "mean": "positive"}).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=6, help="hours ahead")
    parser.add_argument("--merge-gap", default="6h", help="join spills closer than")
    args = parser.parse_args()
    main(args.horizon, args.merge_gap)
