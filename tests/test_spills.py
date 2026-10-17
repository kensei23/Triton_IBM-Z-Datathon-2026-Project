import pandas as pd

from triton.data.spills import (
    alerts_to_events,
    combine_events,
    ea_to_events,
    tidy_sites,
    top_sites,
)

SITES = tidy_sites(
    pd.DataFrame(
        {
            "uniqueId": ["TWL1", "TWL2"],
            "locationName": ["Alpha CSO", "Beta CSO"],
            "permitNumber": ["P1", "P1"],
            "receivingWaterCourse": ["River A", "River B"],
            "x": [530000, 531000],
            "y": [180000, 181000],
        }
    )
)


def alert(name, kind, when):
    return {"locationName": name, "alertType": kind, "datetime": when}


def test_alerts_pair_start_with_next_stop():
    alerts = pd.DataFrame(
        [
            alert("Alpha CSO", "Stop", "2026-02-01T03:00:00"),
            alert("Alpha CSO", "Start", "2026-02-01T01:00:00"),
            alert("Beta CSO", "Start", "2026-02-01T02:00:00"),
            alert("Beta CSO", "Stop", "2026-02-01T02:30:00"),
        ]
    )
    ev = alerts_to_events(alerts, SITES)
    assert list(ev["site_id"]) == ["TWL1", "TWL2"]
    assert ev.loc[0, "end"] - ev.loc[0, "start"] == pd.Timedelta("2h")


def test_offline_and_unpaired_alerts_are_dropped():
    alerts = pd.DataFrame(
        [
            alert("Alpha CSO", "Offline start", "2026-02-01T00:00:00"),
            alert("Alpha CSO", "Offline stop", "2026-02-01T00:30:00"),
            # still spilling, no stop yet
            alert("Alpha CSO", "Start", "2026-02-02T00:00:00"),
            alert("Unknown CSO", "Start", "2026-02-01T00:00:00"),
            alert("Unknown CSO", "Stop", "2026-02-01T01:00:00"),
        ]
    )
    assert alerts_to_events(alerts, SITES).empty


def test_ea_corrections_remove_events():
    df = pd.DataFrame(
        {
            "Unique ID": ["TWL1", "TWL1", "TWL1"],
            "Discharge Strat (GMT)": [
                "2025-01-01T00:00:00.000Z",
                "2025-01-02T00:00:00.000Z",
                "2025-01-03T00:00:00.000Z",
            ],
            "Discharge Stop (GMT)": [
                "2025-01-01T01:00:00.000Z",
                "2025-01-02T01:00:00.000Z",
                "2025-01-03T01:00:00.000Z",
            ],
            "Data Correction": [None, "Removed - not a spill", "Amended"],
        }
    )
    ev = ea_to_events(df)
    assert len(ev) == 2
    assert ev["start"].dt.tz is not None


def test_combine_switches_source_at_cutoff():
    def events(source, starts):
        s = pd.to_datetime(starts, utc=True)
        return pd.DataFrame(
            {
                "site_id": "TWL1",
                "start": s,
                "end": s + pd.Timedelta("1h"),
                "source": source,
            }
        )

    ea = events("EA EDM", ["2025-12-30", "2026-01-05"])
    api = events("TW API", ["2025-12-30", "2026-01-06"])
    out = combine_events(ea, api, pd.Timestamp("2026-01-01", tz="UTC"))
    assert list(out["source"]) == ["EA EDM", "TW API"]


def test_top_sites_needs_events_every_year():
    starts = (
        ["2024-06-01"] * 5
        + ["2025-06-01"] * 5  # busy site A
        + ["2024-06-01"] * 20  # busier site B but silent in 2025
    )
    df = pd.DataFrame(
        {
            "site_id": ["A"] * 10 + ["B"] * 20,
            "start": pd.to_datetime(starts, utc=True),
        }
    )
    assert top_sites(df, n=5, min_events_per_year=3) == ["A"]
