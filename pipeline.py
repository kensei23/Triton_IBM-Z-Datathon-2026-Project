"""Builds the datasets for the Triton river pollution early warning system.

Usage: python pipeline.py [--refresh]

Raw downloads are cached in data/raw. Use --refresh to download them again.
Every output that other code should read is written to data/processed as CSV;
read them with load() so date columns come back as UTC timestamps.
"""
import argparse
import json
import time
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

EA = "https://environment.data.gov.uk/hydrology"
EA_EDM = "https://environment.data.gov.uk/api/file/download"
EA_EDM_ID = "c95ecc47-fa70-4a24-b8cc-098232987526"
TW = "https://api.thameswater.co.uk/opendata/v2/discharge"
RAW, PROCESSED = Path("data/raw"), Path("data/processed")

START = "2024-01-01"
END = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
# EA yearly spill files cover history; the Thames Water API covers the current year.
EA_SPILL_FILES = ["Thames 2024 Detailed EDM Data.csv", "Thames Water 2025 EDM Detailed Data.xlsx"]
API_FROM = pd.Timestamp("2026-01-01", tz="UTC")
# Tunnel outfalls spilled into the river until the Tideway Tunnel was fully connected.
TIDEWAY_DATE = pd.Timestamp("2025-02-14", tz="UTC")
REMOVED_CORRECTION = r"remov|delet|invalid|not a (?:discharge|spill)|false"

PROPS = ["dissolved-oxygen", "ammonium", "conductivity", "turbidity", "temperature", "ph"]
USABLE_QUALITY = ["Good", "Unchecked", "Estimated"]
VALID_RANGE = {"dissolved-oxygen": (0, 20), "ammonium": (0, 50), "ph": (4, 10),
               "temperature": (-1, 35), "conductivity": (1, 5000), "turbidity": (0, 1000)}
STUCK_HOURS = 6
MAX_GAP_HOURS = 3
OUTLIER_Z = 5
DATE_COLUMNS = {"timestamp", "start", "end"}

LEE = r"\b(?:Lee|Lea|Pymmes|Salmon|Ching)\b"

# direction: where upstream outfalls lie. The Lee flows south and the tidal Thames
# flows both ways. Brent and Roding stations were tested and dropped: their sensors
# showed no clear response to upstream spills (anomaly lift 0.9 to 1.7).
STATIONS = {
    "E01336A": {"name": "Lee Springfield Park", "outfall_river": LEE, "direction": "north",
                "km": 15, "gauge_river": r"\b(?:Lee|Lea)\b"},
    "E26020A": {"name": "Lee Lea Bridge Weir", "outfall_river": LEE, "direction": "north",
                "km": 15, "gauge_river": r"\b(?:Lee|Lea)\b"},
    "HAMME2": {"name": "Thames Hammersmith", "outfall_river": r"\bThames\b",
               "direction": "both", "km": 8, "gauge_river": r"\bThames\b"},
    "BREPON": {"name": "Thames Brentford", "outfall_river": r"\b(?:Thames|Brent)\b",
               "direction": "both", "km": 8, "gauge_river": r"\bThames\b"},
}


def get_json(url, **params):
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()["items"]


def get_readings(measure, start=START, end=END):
    r = requests.get(f"{EA}/id/measures/{measure}/readings.csv",
                     params={"mineq-date": start, "max-date": end, "_limit": 200000}, timeout=300)
    if not r.ok or not r.text.strip():
        print(f"  no data: {measure}")
        return pd.DataFrame()
    return pd.read_csv(StringIO(r.text))


def as_text(value):
    return " / ".join(value) if isinstance(value, list) else (value or "")


def station_id(item):
    s = item["station"]
    return (s["@id"] if isinstance(s, dict) else s).split("/")[-1]


def km_from(df, x, y):
    return np.hypot(df["easting"] - x, df["northing"] - y) / 1000


def cached(path, fetch, refresh):
    if path.exists() and not refresh:
        print(f"  using cached {path}")
        return pd.read_csv(path)
    df = fetch()
    df.to_csv(path, index=False)
    return df


def load(name, folder=PROCESSED):
    # CSV does not store types, so date columns are parsed back to UTC timestamps here.
    df = pd.read_csv(folder / f"{name}.csv")
    for col in DATE_COLUMNS.intersection(df.columns):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def to_hourly(df, how="mean"):
    df = df[df["quality"].isin(USABLE_QUALITY)] if "quality" in df else df
    s = df.assign(timestamp=pd.to_datetime(df["dateTime"], utc=True)).set_index("timestamp")["value"]
    return s.resample("1h").sum(min_count=1) if how == "sum" else s.resample("1h").mean()


def ea_stations():
    items = get_json(f"{EA}/id/stations.json", lat=51.6, long=-0.8, dist=70, _limit=10000)
    return pd.DataFrame([{"station_id": s["@id"].split("/")[-1], "label": as_text(s.get("label")),
                          "river": as_text(s.get("riverName")), "easting": s.get("easting"),
                          "northing": s.get("northing")} for s in items]).set_index("station_id")


def preferred_measure(items):
    # Dissolved oxygen is published in % and mg/L; mg/L is used for every station.
    return next((i["notation"] for i in items if i["notation"].endswith("mgL")), items[0]["notation"])


def download_readings():
    frames = []
    for sid in STATIONS:
        for prop in PROPS:
            items = get_json(f"{EA}/id/measures.json", station=sid, observedProperty=prop)
            if not items:
                continue
            df = get_readings(preferred_measure(items))
            if not df.empty:
                frames.append(df.assign(station_id=sid, measure=prop))
                print(f"  {sid} {prop}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def gap_lengths(s):
    missing = s.isna()
    runs = (missing != missing.shift()).cumsum()
    return missing.groupby(runs).sum()[lambda x: x > 0]


def clean_readings(raw):
    # Faults (impossible values, frozen sensors) are removed. Large spikes are only
    # flagged, never removed, because a sudden spike is the pollution signal itself.
    parts, report = [], []
    for (sid, prop), g in raw.groupby(["station_id", "measure"]):
        hourly = to_hourly(g)
        lo, hi = VALID_RANGE[prop]
        out_of_range = hourly.notna() & ~hourly.between(lo, hi)
        run = (~hourly.diff().eq(0)).cumsum()
        stuck = hourly.notna() & (hourly.groupby(run).transform("size") >= STUCK_HOURS)
        s = hourly.mask(out_of_range | stuck)
        filled = s.interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
        gaps = gap_lengths(s)
        report.append({"station_id": sid, "measure": prop, "hours": len(hourly),
                       "missing_raw_pct": round(hourly.isna().mean() * 100, 2),
                       "out_of_range": int(out_of_range.sum()), "stuck": int(stuck.sum()),
                       "filled": int(filled.notna().sum() - s.notna().sum()),
                       "missing_final_pct": round(filled.isna().mean() * 100, 2),
                       "longest_gap_h": int(gaps.max()) if len(gaps) else 0,
                       "large_spikes": int((robust_z(filled).abs() > OUTLIER_Z).sum())})
        parts.append(filled.dropna().rename("value").reset_index().assign(station_id=sid, measure=prop))
    readings = pd.concat(parts, ignore_index=True)[["station_id", "timestamp", "measure", "value"]]
    return readings, pd.DataFrame(report)


def tw_page(endpoint, offset, limit=1000, retries=6):
    for attempt in range(retries):
        r = requests.get(f"{TW}/{endpoint}", params={"limit": limit, "offset": offset}, timeout=120)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 10 * (attempt + 1))))
            continue
        r.raise_for_status()
        return r.json()["items"]
    raise RuntimeError(f"Thames Water API still rate limiting at offset {offset}")


def download_alerts(since=API_FROM - pd.Timedelta(days=7), limit=1000, max_pages=300, pause=1.5):
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


def download_outfalls(limit=1000):
    rows, offset = [], 0
    while True:
        items = tw_page("status", offset, limit)
        rows += items
        if len(items) < limit:
            break
        offset += limit
        time.sleep(1.5)
    return pd.DataFrame(rows)[["uniqueId", "locationName", "permitNumber", "receivingWaterCourse", "x", "y"]]


def download_ea_file(name):
    r = requests.get(EA_EDM, params={"fileDataSetId": EA_EDM_ID, "fileName": name}, timeout=300)
    r.raise_for_status()
    # Some files labelled .csv are really Excel files, so the content decides.
    if r.content[:2] == b"PK":
        return pd.read_excel(BytesIO(r.content))
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(BytesIO(r.content), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(BytesIO(r.content), encoding="latin-1")


SPILL_COLUMNS = ["outfall", "permitNumber", "river", "easting", "northing", "start", "end", "source"]


def to_events(alerts):
    # Offline start/stop means the monitor was down, not a spill. Several outfalls
    # share one permit, so events are paired by location name.
    a = alerts[alerts["alertType"].isin(["Start", "Stop"])].drop_duplicates().copy()
    a["datetime"] = pd.to_datetime(a["datetime"], utc=True)
    a = a.sort_values(["locationName", "datetime"])
    a["next_time"] = a.groupby("locationName")["datetime"].shift(-1)
    a["next_type"] = a.groupby("locationName")["alertType"].shift(-1)
    ev = a[(a["alertType"] == "Start") & (a["next_type"] == "Stop")]
    return ev.rename(columns={"locationName": "outfall", "receivingWaterCourse": "river",
                              "x": "easting", "y": "northing", "datetime": "start", "next_time": "end"}
                     ).assign(source="Thames Water API")[SPILL_COLUMNS]


def ea_events(df, outfalls, name):
    # The EA files hold paired events but no locations; locations come from the
    # Thames Water outfall list, joined on the shared TWL unique ID.
    start = next(c for c in df.columns if c.startswith("Discharge St") and "Stop" not in c)
    stop = next(c for c in df.columns if c.startswith("Discharge Stop"))
    keep = pd.Series(True, index=df.index)
    if "Data Correction" in df:
        keep = ~df["Data Correction"].astype(str).str.contains(REMOVED_CORRECTION, case=False, na=False)
        print(f"  {name}: {int((~keep).sum())} events removed by data corrections")
    ev = pd.DataFrame({"uniqueId": df.loc[keep, "Unique ID"],
                       "start": pd.to_datetime(df.loc[keep, start], utc=True, errors="coerce"),
                       "end": pd.to_datetime(df.loc[keep, stop], utc=True, errors="coerce")}).dropna()
    print(f"  {name}: {len(ev)} events, {ev['uniqueId'].isin(outfalls['uniqueId']).mean():.1%} matched to a location")
    ev = ev.merge(outfalls, on="uniqueId")
    return ev.rename(columns={"locationName": "outfall", "receivingWaterCourse": "river",
                              "x": "easting", "y": "northing"}).assign(source="EA EDM")[SPILL_COLUMNS]


def build_spills(ea_frames, alerts, outfalls):
    history = pd.concat([ea_events(df, outfalls, name) for name, df in ea_frames.items()])
    live = to_events(alerts)
    spills = pd.concat([history[history["start"] < API_FROM], live[live["start"] >= API_FROM]])
    return spills.sort_values(["outfall", "start"]).reset_index(drop=True)


def match_outfalls(spills, coords):
    outfalls = spills.drop_duplicates("outfall")
    river = outfalls["river"].fillna("")
    matched = {}
    for sid, cfg in STATIONS.items():
        x, y = coords.at[sid, "easting"], coords.at[sid, "northing"]
        mask = river.str.contains(cfg["outfall_river"], case=False, regex=True) & (km_from(outfalls, x, y) <= cfg["km"])
        if cfg["direction"] == "north":
            mask &= outfalls["northing"] > y
        elif cfg["direction"] == "south":
            mask &= outfalls["northing"] < y
        matched[sid] = sorted(outfalls.loc[mask, "outfall"])
    return matched


def station_spills(spills, upstream, sid):
    s = spills[spills["outfall"].isin(upstream[sid])]
    via_tunnel = s["river"].fillna("").str.contains("Tideway tunnel", case=False)
    return s[~(via_tunnel & (s["start"] >= TIDEWAY_DATE))]


def robust_z(s, window="7D", min_periods=24):
    base = s.rolling(window, min_periods=min_periods).median().shift(1)
    mad = (s - base).abs().rolling(window, min_periods=min_periods).median().shift(1)
    return (s - base) / (1.4826 * mad + 1e-6)


def spill_window(index, events, after="6h"):
    mask = pd.Series(False, index=index)
    for start, end in zip(events["start"], events["end"]):
        mask |= (index >= start) & (index <= end + pd.Timedelta(after))
    return mask


def build_labels(readings, spills, upstream, z_thr=2.5):
    frames = []
    for sid in STATIONS:
        wide = readings[readings["station_id"] == sid].pivot(index="timestamp", columns="measure", values="value")
        if not {"dissolved-oxygen", "ammonium"}.issubset(wide.columns):
            print(f"  {sid}: skipped, needs both dissolved oxygen and ammonium")
            continue
        z_do, z_nh4 = robust_z(wide["dissolved-oxygen"]), robust_z(wide["ammonium"])
        recent = spill_window(wide.index, station_spills(spills, upstream, sid))
        frames.append(pd.DataFrame({
            "station_id": sid, "timestamp": wide.index, "spill_recent": recent.values,
            "z_do": z_do.values, "z_nh4": z_nh4.values,
            "label": (recent & ((z_do < -z_thr) | (z_nh4 > z_thr))).values}))
    return pd.concat(frames, ignore_index=True)


def anomaly_lift(station_labels, z_thr=2.5):
    # How much more often the sensors look abnormal during spill windows than outside them.
    # Near 1 means the labels are mostly chance; well above 1 means spills really move the sensors.
    anomaly = (station_labels["z_do"] < -z_thr) | (station_labels["z_nh4"] > z_thr)
    inside = anomaly[station_labels["spill_recent"]].mean()
    outside = anomaly[~station_labels["spill_recent"]].mean()
    return inside / outside if outside > 0 else float("nan")


def gauge_candidates(prop, stations):
    items = get_json(f"{EA}/id/measures.json", observedProperty=prop, _limit=20000)
    m = pd.DataFrame([{"measure": i["notation"], "period": i.get("period"), "station_id": station_id(i)}
                      for i in items])
    m = m[pd.to_numeric(m["period"], errors="coerce") == 900]
    return m.join(stations, on="station_id", how="inner")


def pick_gauges(cands, coords, river_key=None):
    # Flow is taken from the station's own river when possible; rain from the nearest gauge.
    picks = {}
    for sid, cfg in STATIONS.items():
        c = cands.assign(km=km_from(cands, coords.at[sid, "easting"], coords.at[sid, "northing"])).sort_values("km")
        if river_key:
            same = c[c["river"].str.contains(cfg[river_key], case=False, regex=True)]
            c = same if not same.empty else c
        best = c.iloc[0]
        picks[sid] = {"measure": best["measure"], "label": best["label"], "km": round(float(best["km"]), 1)}
    return picks


def gauge_series(picks, how, name):
    frames = []
    for sid, g in picks.items():
        df = get_readings(g["measure"])
        if not df.empty:
            s = to_hourly(df, how)
            frames.append(s.mask(s < 0).rename(name).reset_index().assign(station_id=sid))
    return pd.concat(frames, ignore_index=True)[["station_id", "timestamp", name]]


def main(refresh=False):
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    stations = ea_stations()
    coords = stations.loc[list(STATIONS), ["easting", "northing"]]

    print("Sensor readings")
    raw = cached(RAW / "raw_readings.csv", download_readings, refresh)
    readings, quality = clean_readings(raw[raw["station_id"].isin(STATIONS)])
    print(quality.to_string(index=False))

    print("Sewage spills")
    outfalls = cached(RAW / "tw_outfalls.csv", download_outfalls, refresh)
    ea_frames = {name: cached(RAW / f"ea_{Path(name).stem.replace(' ', '_').lower()}.csv",
                              lambda n=name: download_ea_file(n), refresh) for name in EA_SPILL_FILES}
    alerts = cached(RAW / "tw_alerts.csv", download_alerts, refresh)
    spills = build_spills(ea_frames, alerts, outfalls)
    upstream = match_outfalls(spills, coords)
    labels = build_labels(readings, spills, upstream)

    print("Rain and flow gauges")
    rain_gauges = pick_gauges(gauge_candidates("rainfall", stations), coords)
    flow_gauges = pick_gauges(gauge_candidates("waterFlow", stations), coords, river_key="gauge_river")
    rain = gauge_series(rain_gauges, "sum", "mm")
    flow = gauge_series(flow_gauges, "mean", "flow_m3s")

    outputs = {"readings": readings, "spills": spills, "labels": labels,
               "rain": rain, "flow": flow, "quality_report": quality}
    for name, df in outputs.items():
        df.to_csv(PROCESSED / f"{name}.csv", index=False)

    meta = {sid: {"name": cfg["name"],
                  "easting": int(coords.at[sid, "easting"]), "northing": int(coords.at[sid, "northing"]),
                  "rain_gauge": rain_gauges[sid], "flow_gauge": flow_gauges[sid],
                  "upstream_outfalls": upstream[sid]}
            for sid, cfg in STATIONS.items()}
    (PROCESSED / "stations.json").write_text(json.dumps(meta, indent=1))
    (PROCESSED / "upstream.json").write_text(json.dumps(upstream, indent=1))

    print(spills.groupby([spills["start"].dt.year, "source"]).size().to_string())
    summary = labels.groupby("station_id").agg(first=("timestamp", "min"), hours=("timestamp", "size"),
                                               spill_recent=("spill_recent", "mean"),
                                               label_rate=("label", "mean"), label_hours=("label", "sum"))
    summary["upstream_spills"] = [len(station_spills(spills, upstream, s)) for s in summary.index]
    summary["anomaly_lift"] = [anomaly_lift(labels[labels["station_id"] == s]) for s in summary.index]
    print(summary.round({"spill_recent": 4, "label_rate": 4, "anomaly_lift": 2}).to_string())
    for sid, m in meta.items():
        print(f"{m['name']}: rain {m['rain_gauge']['label']} ({m['rain_gauge']['km']} km), "
              f"flow {m['flow_gauge']['label']} ({m['flow_gauge']['km']} km)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="download the raw data again")
    main(parser.parse_args().refresh)
