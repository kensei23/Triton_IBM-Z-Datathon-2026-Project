"""Environment Agency sources: Hydrology API and EDM spill files."""

from io import BytesIO, StringIO

import pandas as pd
import requests

from triton.config import (
    EA_EDM,
    EA_EDM_DATASET,
    EA_HYDROLOGY,
    REMOVED_CORRECTION,
    SEARCH_AREA,
)
from triton.io import read_json


def _get(url, params, timeout=300):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r


def _station_id(item):
    s = item["station"]
    return (s["@id"] if isinstance(s, dict) else s).split("/")[-1]


def _as_text(value):
    return " / ".join(value) if isinstance(value, list) else (value or "")


def download_stations(store):
    url = f"{EA_HYDROLOGY}/id/stations.json"
    params = {**SEARCH_AREA, "_limit": 10000}
    if not store.has("ea_hydrology/stations.json"):
        store.save("ea_hydrology/stations.json", _get(url, params).content, url, params)


def download_station_measures(store, station_id, prop):
    relpath = f"ea_hydrology/measures/{station_id}_{prop}.json"
    url = f"{EA_HYDROLOGY}/id/measures.json"
    params = {"station": station_id, "observedProperty": prop}
    if not store.has(relpath):
        store.save(relpath, _get(url, params).content, url, params)
    return store.root / relpath


def download_gauge_measures(store, prop):
    relpath = f"ea_hydrology/measures/{prop}.json"
    url = f"{EA_HYDROLOGY}/id/measures.json"
    params = {"observedProperty": prop, "_limit": 20000}
    if not store.has(relpath):
        store.save(relpath, _get(url, params).content, url, params)
    return store.root / relpath


def download_readings(store, measure, start, end):
    relpath = f"ea_hydrology/readings/{measure}.csv"
    url = f"{EA_HYDROLOGY}/id/measures/{measure}/readings.csv"
    params = {"mineq-date": start, "max-date": end, "_limit": 200000}
    if not store.has(relpath):
        store.save(relpath, _get(url, params).content, url, params)


def download_edm_file(store, name):
    relpath = f"ea_edm/{name}"
    params = {"fileDataSetId": EA_EDM_DATASET, "fileName": name}
    if not store.has(relpath):
        store.save(relpath, _get(EA_EDM, params).content, EA_EDM, params)


def parse_stations(path):
    rows = [
        {
            "station_id": s["@id"].split("/")[-1],
            "label": _as_text(s.get("label")),
            "river": _as_text(s.get("riverName")),
            "easting": s.get("easting"),
            "northing": s.get("northing"),
        }
        for s in read_json(path)["items"]
    ]
    return pd.DataFrame(rows).drop_duplicates("station_id").set_index("station_id")


def preferred_measure(path):
    # Dissolved oxygen is published in % and mg/L; mg/L is used for every station.
    items = read_json(path)["items"]
    if not items:
        return None
    notations = [i["notation"] for i in items]
    return next((n for n in notations if n.endswith("mgL")), notations[0])


def parse_gauge_measures(path):
    rows = [
        {
            "measure": i["notation"],
            "period": i.get("period"),
            "station_id": _station_id(i),
        }
        for i in read_json(path)["items"]
    ]
    m = pd.DataFrame(rows)
    return m[pd.to_numeric(m["period"], errors="coerce") == 900].drop(columns="period")


def gauge_candidates(path, stations):
    return parse_gauge_measures(path).join(stations, on="station_id", how="inner")


def parse_readings(path):
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return pd.DataFrame(columns=["dateTime", "value", "quality"])
    return pd.read_csv(StringIO(text))


def read_edm_file(path):
    # Some files published as .csv are Excel workbooks, so the content decides.
    content = path.read_bytes()
    if content[:2] == b"PK":
        return pd.read_excel(BytesIO(content))
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(BytesIO(content), encoding="latin-1")


def edm_events(df, outfalls, name):
    """Paired spill events from an EA EDM file, located via the shared TWL ID."""
    start = next(
        c for c in df.columns if c.startswith("Discharge St") and "Stop" not in c
    )
    stop = next(c for c in df.columns if c.startswith("Discharge Stop"))
    keep = pd.Series(True, index=df.index)
    if "Data Correction" in df:
        keep = ~df["Data Correction"].astype(str).str.contains(
            REMOVED_CORRECTION, case=False, na=False
        )
    events = pd.DataFrame(
        {
            "uniqueId": df.loc[keep, "Unique ID"],
            "start": pd.to_datetime(df.loc[keep, start], utc=True, errors="coerce"),
            "end": pd.to_datetime(df.loc[keep, stop], utc=True, errors="coerce"),
        }
    ).dropna()
    matched = events["uniqueId"].isin(outfalls["uniqueId"]).mean()
    print(
        f"  {name}: {int((~keep).sum())} removed by corrections, "
        f"{len(events)} events, {matched:.1%} matched to a location"
    )
    return events.merge(outfalls, on="uniqueId").assign(source="EA EDM")
