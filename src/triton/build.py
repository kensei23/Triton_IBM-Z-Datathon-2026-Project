"""Builds data/processed from data/interim: cleaned readings, labels and the
modelling table.

Usage: uv run python -m triton.build
"""

import json

import pandas as pd

from triton.cleaning import clean_gauges, clean_readings
from triton.config import INTERIM, PROCESSED, STATIONS
from triton.dataset import build_dataset
from triton.io import load, save_csv
from triton.labels import anomaly_lift, station_labels
from triton.spatial import match_outfalls, station_spills


def build_labels(readings, spills, upstream):
    frames = []
    for sid in STATIONS:
        wide = readings[readings["station_id"] == sid].pivot(
            index="timestamp", columns="measure", values="value"
        )
        if not {"dissolved-oxygen", "ammonium"}.issubset(wide.columns):
            print(f"  {sid}: skipped, needs dissolved oxygen and ammonium")
            continue
        events = station_spills(spills, upstream, sid)
        frames.append(station_labels(wide, events).assign(station_id=sid))
    cols = ["station_id", "timestamp", "spill_recent", "z_do", "z_nh4", "label"]
    return pd.concat(frames, ignore_index=True)[cols]


def summarise(labels, spills, upstream):
    rows = []
    for sid, group in labels.groupby("station_id"):
        rows.append(
            {
                "station_id": sid,
                "name": STATIONS[sid]["name"],
                "first": group["timestamp"].min().date(),
                "hours": len(group),
                "upstream_spills": len(station_spills(spills, upstream, sid)),
                "spill_recent": round(group["spill_recent"].mean(), 4),
                "label_hours": int(group["label"].sum()),
                "label_rate": round(group["label"].mean(), 4),
                "anomaly_lift": round(anomaly_lift(group), 2),
            }
        )
    return pd.DataFrame(rows)


def main():
    readings_raw = load("sensor_readings", INTERIM)
    readings_raw = readings_raw[readings_raw["station_id"].isin(STATIONS)]
    readings, quality = clean_readings(readings_raw)
    gauges = clean_gauges(load("gauge_readings", INTERIM))
    spills = load("spills", INTERIM)
    meta = json.loads((INTERIM / "stations.json").read_text())

    coords = pd.DataFrame(meta).T[["easting", "northing"]].astype(float)
    upstream = match_outfalls(spills, coords)
    labels = build_labels(readings, spills, upstream)
    events = {sid: station_spills(spills, upstream, sid) for sid in STATIONS}
    dataset = build_dataset(readings, gauges, labels, events)

    for name, df in {
        "dataset": dataset,
        "labels": labels,
        "spills": spills,
        "quality_report": quality,
    }.items():
        save_csv(df, PROCESSED, name)
    for sid in meta:
        meta[sid]["upstream_outfalls"] = upstream[sid]
    (PROCESSED / "stations.json").write_text(json.dumps(meta, indent=1))

    print(quality.to_string(index=False))
    print(summarise(labels, spills, upstream).to_string(index=False))
    print(
        f"dataset.csv: {len(dataset)} rows, {dataset['station_id'].nunique()} stations"
    )


if __name__ == "__main__":
    main()
