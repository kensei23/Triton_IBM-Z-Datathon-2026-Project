"""Replay a past storm hour by hour, as if it were happening live.

Every feature in the modelling table only uses data up to its hour (see the
leakage tests), so stepping through the table in time order is exactly what a
live system would have seen.

An alert is raised when a site escalates (MONITOR -> NOTIFY, or up to
INTERVENE), not every hour it stays high, as an operator would want.
"""

import pandas as pd

from triton import model as m
from triton.decision import decide

RANK = {"MONITOR": 0, "NOTIFY": 1, "INTERVENE": 2}


def pick_storm(episodes, start_from, hours=36, lead_h=12):
    """A window over the busiest 24 hours of spill starts after `start_from`.

    Starts `lead_h` hours before that day so the build-up is visible, and runs
    `hours` hours from there.
    """
    ep = episodes[episodes["start"] >= start_from]
    per_hour = ep.set_index("start").resample("1h").size()
    busy_day_end = per_hour.rolling(24).sum().idxmax()
    t0 = (busy_day_end - pd.Timedelta(hours=24 + lead_h)).floor("h")
    t0 = max(t0, pd.Timestamp(start_from).ceil("h"))
    return t0, t0 + pd.Timedelta(hours=hours)


def replay(panel, episodes, bundle, thresholds, t0, t1):
    """Yield one dict per hour: every site's probability and level, and events.

    Events are escalations (an alert to raise) and spill starts (what really
    happened in that hour), so the caller can print them as they happen.
    """
    rows = panel[(panel["timestamp"] >= t0) & (panel["timestamp"] <= t1)]
    starts = episodes[
        (episodes["start"] > t0) & (episodes["start"] <= t1 + pd.Timedelta(hours=1))
    ]
    last = {}
    for t, hour in rows.groupby("timestamp", sort=True):
        hour = hour.set_index("site_id")
        risk = hour[hour["spilling_now"] == 0]
        probs = (
            pd.Series(m.predict(bundle, risk), index=risk.index)
            if len(risk)
            else pd.Series(dtype=float)
        )
        levels = {}
        escalations = []
        for site in hour.index:
            if hour.at[site, "spilling_now"]:
                levels[site] = "SPILLING"
                last[site] = "SPILLING"
                continue
            level = decide(probs[site], thresholds)
            levels[site] = level
            before = last.get(site, "MONITOR")
            before_rank = RANK.get(before, 0)  # after a spill, start from MONITOR
            if RANK[level] > before_rank:
                escalations.append(
                    {"site_id": site, "level": level, "p": float(probs[site])}
                )
            last[site] = level
        new_spills = starts[
            (starts["start"] > t - pd.Timedelta(hours=1)) & (starts["start"] <= t)
        ]
        yield {
            "timestamp": t,
            "probs": probs,
            "levels": levels,
            "rows": hour,
            "escalations": sorted(escalations, key=lambda e: -e["p"]),
            "spill_starts": new_spills[["site_id", "start"]].to_dict("records"),
        }


def warning_lead(hours, site_id, start, horizon=6):
    """Hours between the first NOTIFY-or-higher hour and the spill start.

    Only the `horizon` hours before the start count, the window the model
    promises. None if the site was never warned in that window.
    """
    window = pd.Timedelta(hours=horizon)
    for h in hours:
        t = h["timestamp"]
        if start - window <= t < start and RANK.get(h["levels"].get(site_id), 0) >= 1:
            return (start - t) / pd.Timedelta(hours=1)
    return None


def score(hours, episodes, t0, t1, horizon=6):
    """How the replay did: spills warned in time, lead times, false alarms."""
    window = pd.Timedelta(hours=horizon)
    # spills in the first `horizon` hours had no full warning window
    ep = episodes[(episodes["start"] > t0 + window) & (episodes["start"] <= t1)]
    leads = [warning_lead(hours, e.site_id, e.start, horizon) for e in ep.itertuples()]

    alerts = [(h["timestamp"], e["site_id"]) for h in hours for e in h["escalations"]]
    false_alarms = 0
    for t, site in alerts:
        nxt = episodes[
            (episodes["site_id"] == site)
            & (episodes["start"] > t)
            & (episodes["start"] <= t + window)
        ]
        false_alarms += nxt.empty
    caught = [x for x in leads if x is not None]
    return {
        "spills": len(leads),
        "warned": len(caught),
        "median_lead_h": round(float(pd.Series(caught).median()), 1)
        if caught
        else None,
        "alerts": len(alerts),
        "false_alarms": int(false_alarms),
        "lead_times_h": [None if x is None else round(x, 2) for x in leads],
    }
