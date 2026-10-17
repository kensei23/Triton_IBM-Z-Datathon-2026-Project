"""Matching outfalls and gauges to stations. Coordinates are British National Grid
metres, so distance is plain Pythagoras."""

import numpy as np

from triton.config import STATIONS, TIDEWAY_DATE


def km_from(df, x, y):
    return np.hypot(df["easting"] - x, df["northing"] - y) / 1000


def match_outfalls(spills, coords, stations=STATIONS):
    outfalls = spills.drop_duplicates("outfall")
    river = outfalls["river"].fillna("")
    matched = {}
    for sid, cfg in stations.items():
        x, y = coords.at[sid, "easting"], coords.at[sid, "northing"]
        mask = river.str.contains(cfg["outfall_river"], case=False, regex=True)
        mask &= km_from(outfalls, x, y) <= cfg["km"]
        if cfg["direction"] == "north":
            mask &= outfalls["northing"] > y
        elif cfg["direction"] == "south":
            mask &= outfalls["northing"] < y
        matched[sid] = sorted(outfalls.loc[mask, "outfall"])
    return matched


def station_spills(spills, upstream, station_id):
    s = spills[spills["outfall"].isin(upstream[station_id])]
    via_tunnel = s["river"].fillna("").str.contains("Tideway tunnel", case=False)
    return s[~(via_tunnel & (s["start"] >= TIDEWAY_DATE))]


def pick_gauges(candidates, coords, river_key=None, stations=STATIONS):
    """Nearest gauge per station; with river_key, the nearest on the same river."""
    picks = {}
    for sid, cfg in stations.items():
        x, y = coords.at[sid, "easting"], coords.at[sid, "northing"]
        c = candidates.assign(km=km_from(candidates, x, y)).sort_values("km")
        if river_key:
            same = c[c["river"].str.contains(cfg[river_key], case=False, regex=True)]
            c = same if not same.empty else c
        best = c.iloc[0]
        picks[sid] = {
            "measure": best["measure"],
            "gauge": best["label"],
            "km": round(float(best["km"]), 1),
        }
    return picks
