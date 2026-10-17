"""Downloads every source into data/raw, unchanged.

Usage: uv run python -m triton.download [--force]

Files already present are skipped; --force downloads everything again.
Each file's URL, parameters and download time are recorded in data/raw/manifest.json.
"""

import argparse

import pandas as pd

from triton.config import API_FROM, EA_EDM_FILES, RAW, SENSORS, START, STATIONS
from triton.io import RawStore
from triton.sources import ea, thames_water
from triton.spatial import pick_gauges


def selected_gauges(store, stations):
    coords = stations.loc[list(STATIONS), ["easting", "northing"]]
    rain_path = ea.download_gauge_measures(store, "rainfall")
    flow_path = ea.download_gauge_measures(store, "waterFlow")
    rain = pick_gauges(ea.gauge_candidates(rain_path, stations), coords)
    flow = pick_gauges(
        ea.gauge_candidates(flow_path, stations), coords, river_key="gauge_river"
    )
    return rain, flow


def main(force=False):
    store = RawStore(RAW, force=force)
    end = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print("EA stations and sensors")
    ea.download_stations(store)
    stations = ea.parse_stations(RAW / "ea_hydrology/stations.json")
    for sid in STATIONS:
        for prop in SENSORS:
            measure = ea.preferred_measure(
                ea.download_station_measures(store, sid, prop)
            )
            if measure:
                ea.download_readings(store, measure, START, end)
                print(f"  {sid} {prop}: {measure}")

    print("EA rain and flow gauges")
    rain, flow = selected_gauges(store, stations)
    for picks in (rain, flow):
        for g in picks.values():
            ea.download_readings(store, g["measure"], START, end)

    print("EA spill history")
    for name in EA_EDM_FILES:
        ea.download_edm_file(store, name)

    print("Thames Water outfalls and alerts")
    thames_water.download_pages(store, "status")
    since = (API_FROM - pd.Timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
    thames_water.download_pages(store, "alerts", since=since)
    print(f"Done: {len(store.manifest)} files listed in {store.manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="download everything again"
    )
    main(parser.parse_args().force)
