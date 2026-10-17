import numpy as np
import pandas as pd

from triton.config import STATIONS
from triton.sources.ea import edm_events
from triton.sources.thames_water import alert_events
from triton.spatial import match_outfalls, station_spills

OUTFALLS = pd.DataFrame(
    [
        ["TWL1", "Deephams", "P1", "Salmon Brook", 535660, 193150],
        [
            "TWL2",
            "Brook Green CSO",
            "P2",
            "River Thames (via the Tideway tunnel)",
            523040,
            178150,
        ],
        ["TWL3", "Downstream Lee", "P3", "River Lee", 535800, 180000],
    ],
    columns=["uniqueId", "outfall", "permitNumber", "river", "easting", "northing"],
)
COORDS = pd.DataFrame(
    {
        "easting": [534890, 535636, 522681, 517933],
        "northing": [187514, 186539, 178245, 177017],
    },
    index=list(STATIONS),
)


def alert(name, kind, time):
    return {
        "locationName": name,
        "permitNumber": "P1",
        "receivingWaterCourse": "Salmon Brook",
        "x": 535660,
        "y": 193150,
        "alertType": kind,
        "datetime": time,
    }


def test_alerts_paired_and_offline_ignored():
    alerts = pd.DataFrame(
        [
            alert("Deephams", "Start", "2026-02-01T10:00:00"),
            alert("Deephams", "Offline start", "2026-02-01T10:30:00"),
            alert("Deephams", "Stop", "2026-02-01T12:00:00"),
            alert("Deephams", "Start", "2026-02-02T10:00:00"),
        ]
    )
    events = alert_events(alerts)
    assert len(events) == 1
    assert events.loc[0, "end"] == pd.Timestamp("2026-02-01 12:00", tz="UTC")


def test_edm_corrections_removed_and_locations_joined():
    edm = pd.DataFrame(
        {
            "Unique ID": ["TWL1", "TWL1", "TWL9"],
            "Discharge Strat (GMT)": ["2024-02-15T18:00:00.000Z"] * 3,
            "Discharge Stop (GMT)": ["2024-02-15T20:00:00.000Z"] * 3,
            "Data Correction": [np.nan, "Event removed - not a discharge", np.nan],
        }
    )
    events = edm_events(edm, OUTFALLS, "test")
    assert list(events["outfall"]) == ["Deephams"]


def test_lee_matches_only_upstream_outfalls():
    upstream = match_outfalls(OUTFALLS.assign(start=pd.NaT), COORDS)
    assert upstream["E01336A"] == ["Deephams"]


def test_tunnel_outfalls_only_count_before_tideway():
    spills = OUTFALLS.iloc[[1, 1]].assign(
        start=pd.to_datetime(["2024-06-01", "2025-06-01"], utc=True)
    )
    upstream = {"HAMME2": ["Brook Green CSO"]}
    kept = station_spills(spills, upstream, "HAMME2")
    assert list(kept["start"].dt.year) == [2024]
