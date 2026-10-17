"""Thames Water open data: outfall list (status) and live spill alerts."""

import time

import pandas as pd
import requests

from triton.config import THAMES_WATER
from triton.io import read_json

SPILL_COLUMNS = [
    "outfall",
    "permitNumber",
    "river",
    "easting",
    "northing",
    "start",
    "end",
    "source",
]
RENAME = {
    "locationName": "outfall",
    "receivingWaterCourse": "river",
    "x": "easting",
    "y": "northing",
}


def _page(endpoint, offset, limit, retries=6):
    url = f"{THAMES_WATER}/{endpoint}"
    params = {"limit": limit, "offset": offset}
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 10 * (attempt + 1))))
            continue
        r.raise_for_status()
        return r, url, params
    raise RuntimeError(f"Thames Water API still rate limiting at offset {offset}")


def download_pages(store, endpoint, since=None, limit=1000, pause=1.5):
    """Saves every page as returned. Alerts are newest first, so paging stops
    once a page reaches back past `since`."""
    folder = f"thames_water/{endpoint}"
    if store.has(folder):
        return
    offset = 0
    while True:
        r, url, params = _page(endpoint, offset, limit)
        items = r.json()["items"]
        if not items:
            break
        store.save(f"{folder}/page_{offset:06d}.json", r.content, url, params)
        print(f"  {endpoint}: {offset + len(items)} records")
        oldest = min(i.get("datetime", "9999") for i in items)
        if len(items) < limit or (since is not None and oldest < since):
            break
        offset += limit
        time.sleep(pause)


def read_pages(folder):
    rows = []
    for path in sorted(folder.glob("page_*.json")):
        rows += read_json(path)["items"]
    return pd.DataFrame(rows)


def parse_outfalls(folder):
    cols = [
        "uniqueId",
        "locationName",
        "permitNumber",
        "receivingWaterCourse",
        "x",
        "y",
    ]
    return read_pages(folder)[cols].rename(columns=RENAME).drop_duplicates("uniqueId")


def alert_events(alerts):
    """Pairs each Start with the next Stop at the same outfall.

    Offline start/stop means the monitor was down, not a spill. Several outfalls
    share one permit, so events are paired by location name.
    """
    a = alerts[alerts["alertType"].isin(["Start", "Stop"])].drop_duplicates().copy()
    a["datetime"] = pd.to_datetime(a["datetime"], utc=True)
    a = a.sort_values(["locationName", "datetime"])
    a["end"] = a.groupby("locationName")["datetime"].shift(-1)
    a["next_type"] = a.groupby("locationName")["alertType"].shift(-1)
    events = a[(a["alertType"] == "Start") & (a["next_type"] == "Stop")]
    return (
        events.rename(columns={**RENAME, "datetime": "start"})
        .assign(source="Thames Water API")[SPILL_COLUMNS]
        .reset_index(drop=True)
    )
