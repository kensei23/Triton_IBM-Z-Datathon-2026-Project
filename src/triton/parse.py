"""Parses data/raw into tidy tables in data/interim. Nothing is cleaned or dropped
here apart from records that cannot be read; cleaning happens in triton.build.

Usage: uv run python -m triton.parse
"""

import json

import pandas as pd

from triton.config import API_FROM, EA_EDM_FILES, INTERIM, RAW, SENSORS, STATIONS
from triton.io import save_csv
from triton.sources import ea, thames_water
from triton.spatial import pick_gauges

HYDROLOGY = RAW / "ea_hydrology"


def _readings(measure):
    df = ea.parse_readings(HYDROLOGY / "readings" / f"{measure}.csv")
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["dateTime"], utc=True),
            "value": pd.to_numeric(df["value"], errors="coerce"),
            "quality": df.get("quality"),
        }
    )


def sensor_readings():
    frames = []
    for sid in STATIONS:
        for prop in SENSORS:
            measure = ea.preferred_measure(
                HYDROLOGY / "measures" / f"{sid}_{prop}.json"
            )
            if measure is None:
                continue
            df = _readings(measure)
            frames.append(
                df.assign(station_id=sid, measure=prop, source_measure=measure)
            )
    cols = ["station_id", "timestamp", "measure", "value", "quality", "source_measure"]
    return pd.concat(frames, ignore_index=True)[cols]


def gauges(stations):
    coords = stations.loc[list(STATIONS), ["easting", "northing"]]
    rain_cands = ea.gauge_candidates(HYDROLOGY / "measures/rainfall.json", stations)
    flow_cands = ea.gauge_candidates(HYDROLOGY / "measures/waterFlow.json", stations)
    picks = {
        "rain": pick_gauges(rain_cands, coords),
        "flow": pick_gauges(flow_cands, coords, river_key="gauge_river"),
    }
    frames = [
        _readings(g["measure"]).assign(station_id=sid, kind=kind)
        for kind, by_station in picks.items()
        for sid, g in by_station.items()
    ]
    cols = ["station_id", "timestamp", "kind", "value", "quality"]
    return pd.concat(frames, ignore_index=True)[cols], picks


def spills():
    outfalls = thames_water.parse_outfalls(RAW / "thames_water/status")
    history = pd.concat(
        [
            ea.edm_events(ea.read_edm_file(RAW / "ea_edm" / name), outfalls, name)
            for name in EA_EDM_FILES
        ]
    )
    live = thames_water.alert_events(
        thames_water.read_pages(RAW / "thames_water/alerts")
    )
    combined = pd.concat(
        [
            history.loc[history["start"] < API_FROM, thames_water.SPILL_COLUMNS],
            live[live["start"] >= API_FROM],
        ]
    )
    return combined.sort_values(["outfall", "start"]).reset_index(drop=True), outfalls


def main():
    stations = ea.parse_stations(HYDROLOGY / "stations.json")
    print("Sensor readings")
    readings = sensor_readings()
    print("Rain and flow gauges")
    gauge_readings, picks = gauges(stations)
    print("Spills")
    spill_events, outfalls = spills()

    save_csv(readings, INTERIM, "sensor_readings")
    save_csv(gauge_readings, INTERIM, "gauge_readings")
    save_csv(spill_events, INTERIM, "spills")
    save_csv(outfalls, INTERIM, "outfalls")
    meta = {
        sid: {
            "name": cfg["name"],
            "easting": int(stations.at[sid, "easting"]),
            "northing": int(stations.at[sid, "northing"]),
            "rain_gauge": picks["rain"][sid],
            "flow_gauge": picks["flow"][sid],
        }
        for sid, cfg in STATIONS.items()
    }
    (INTERIM / "stations.json").write_text(json.dumps(meta, indent=1))
    by_year = spill_events.groupby([spill_events["start"].dt.year, "source"]).size()
    print(by_year.to_string())
    print(f"Interim tables written to {INTERIM}")


if __name__ == "__main__":
    main()
