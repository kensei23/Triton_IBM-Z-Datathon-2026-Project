"""Thames Water storm overflow spill events, 2024 onwards.

History comes from the Environment Agency's yearly EDM files (complete, quality
checked start/stop pairs), the current year from the Thames Water open data API
v2 (no key needed). Download and pairing logic adapted from Oscar's pipeline.py.
"""

import time
from io import BytesIO

import pandas as pd
import requests

TW_API = "https://api.thameswater.co.uk/opendata/v2/discharge"
EA_EDM = "https://environment.data.gov.uk/api/file/download"
EA_EDM_ID = "c95ecc47-fa70-4a24-b8cc-098232987526"
EA_FILES = {
    2024: "Thames 2024 Detailed EDM Data.csv",
    2025: "Thames Water 2025 EDM Detailed Data.xlsx",
}
# EA files cover history; the API covers everything from here on
API_FROM = pd.Timestamp("2026-01-01", tz="UTC")
REMOVED_CORRECTION = r"remov|delet|invalid|not a (?:discharge|spill)|false"

SPILL_COLUMNS = ["site_id", "start", "end", "source"]
SITE_COLUMNS = ["site_id", "name", "permit", "river", "easting", "northing"]


def tw_page(endpoint, offset, limit=1000, retries=6):
    for attempt in range(retries):
        r = requests.get(
            f"{TW_API}/{endpoint}",
            params={"limit": limit, "offset": offset},
            timeout=120,
        )
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 10 * (attempt + 1))))
            continue
        r.raise_for_status()
        return r.json()["items"]
    raise RuntimeError(f"Thames Water API still rate limiting at offset {offset}")


def download_sites(limit=1000, pause=1.5):
    """Every overflow Thames Water monitors, with its grid reference."""
    rows, offset = [], 0
    while True:
        items = tw_page("status", offset, limit)
        rows += items
        if len(items) < limit:
            break
        offset += limit
        time.sleep(pause)
    return pd.DataFrame(rows)


def download_alerts(since, limit=1000, max_pages=400, pause=1.5):
    """Start/stop alerts, newest first, paged back until `since`."""
    rows, offset = [], 0
    for _ in range(max_pages):
        items = tw_page("alerts", offset, limit)
        if not items:
            break
        rows += items
        offset += limit
        oldest = min(i["datetime"] for i in items)
        print(f"  {len(rows)} alerts, oldest {oldest}")
        if pd.Timestamp(oldest, tz="UTC") < since:
            break
        time.sleep(pause)
    return pd.DataFrame(rows)


def download_ea_file(name):
    r = requests.get(
        EA_EDM, params={"fileDataSetId": EA_EDM_ID, "fileName": name}, timeout=300
    )
    r.raise_for_status()
    # files labelled .csv are sometimes really Excel, so the content decides
    if r.content[:2] == b"PK":
        return pd.read_excel(BytesIO(r.content))
    return pd.read_csv(BytesIO(r.content), encoding="cp1252")


def tidy_sites(raw):
    sites = raw.rename(
        columns={
            "uniqueId": "site_id",
            "locationName": "name",
            "permitNumber": "permit",
            "receivingWaterCourse": "river",
            "x": "easting",
            "y": "northing",
        }
    )
    sites = sites.dropna(subset=["site_id", "easting", "northing"])
    return sites.drop_duplicates("site_id")[SITE_COLUMNS].reset_index(drop=True)


def alerts_to_events(alerts, sites):
    """Pair each Start with the next Stop at the same overflow.

    Offline start/stop means the monitor was down, not a spill, so it's dropped.
    Several overflows share a permit, so pairing is by location name, then the
    name is mapped to the overflow's unique ID.
    """
    a = alerts[alerts["alertType"].isin(["Start", "Stop"])].drop_duplicates().copy()
    a["datetime"] = pd.to_datetime(a["datetime"], utc=True)
    a = a.sort_values(["locationName", "datetime"])
    a["end"] = a.groupby("locationName")["datetime"].shift(-1)
    next_type = a.groupby("locationName")["alertType"].shift(-1)
    ev = a[(a["alertType"] == "Start") & (next_type == "Stop")]
    ids = sites.drop_duplicates("name").set_index("name")["site_id"]
    ev = ev.assign(site_id=ev["locationName"].map(ids), source="TW API")
    ev = ev.rename(columns={"datetime": "start"}).dropna(subset=["site_id"])
    return ev[SPILL_COLUMNS].reset_index(drop=True)


def ea_to_events(df):
    """Start/stop pairs from an EA EDM file, minus events a correction removed."""
    start = next(
        c for c in df.columns if c.startswith("Discharge St") and "Stop" not in c
    )
    stop = next(c for c in df.columns if c.startswith("Discharge Stop"))
    keep = pd.Series(True, index=df.index)
    if "Data Correction" in df:
        corr = df["Data Correction"].astype(str)
        keep = ~corr.str.contains(REMOVED_CORRECTION, case=False, na=False)
    ev = pd.DataFrame(
        {
            "site_id": df.loc[keep, "Unique ID"],
            "start": pd.to_datetime(df.loc[keep, start], utc=True, errors="coerce"),
            "end": pd.to_datetime(df.loc[keep, stop], utc=True, errors="coerce"),
        }
    ).dropna()
    ev = ev[ev["end"] >= ev["start"]]
    return ev.assign(source="EA EDM")[SPILL_COLUMNS].reset_index(drop=True)


def combine_events(ea_events, api_events, api_from=API_FROM):
    """EA history before `api_from`, API events from then on."""
    history = ea_events[ea_events["start"] < api_from]
    live = api_events[api_events["start"] >= api_from]
    spills = pd.concat([history, live], ignore_index=True)
    spills = spills.drop_duplicates(["site_id", "start"])
    return spills.sort_values(["site_id", "start"]).reset_index(drop=True)


def top_sites(spills, n, min_events_per_year=10):
    """The n overflows with the most spills that report in every year.

    Requiring events in every year keeps sites whose monitor was missing for a
    whole year out, so the test period (the most recent months) has data.
    """
    per_year = spills.groupby(["site_id", spills["start"].dt.year]).size()
    per_year = per_year.unstack(fill_value=0)
    steady = per_year[(per_year >= min_events_per_year).all(axis=1)]
    return steady.sum(axis=1).sort_values(ascending=False).head(n).index.tolist()
