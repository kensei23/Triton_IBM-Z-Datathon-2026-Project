"""Hourly modelling table: one row per overflow per hour.

Row time t is the moment a decision is made. Every feature uses only data
known at t: Open-Meteo rain stamped t is the total for the hour ending at t,
and a spill counts once its start is at or before t. The target looks ahead:
does a new spill episode start in (t, t + horizon]?
"""

import numpy as np
import pandas as pd

HORIZON_H = 6
MERGE_GAP = "6h"
RAIN_WINDOWS_H = {"1h": 1, "3h": 3, "6h": 6, "24h": 24, "7d": 168, "30d": 720}
# the longest rain window needs this much history before rows are kept
WARMUP = pd.Timedelta(hours=720)
HOURS_SINCE_CAP = 24 * 90

FEATURES = [
    *(f"rain_{name}" for name in RAIN_WINDOWS_H),
    "rain_max_1h_24h",
    "spilling_now",
    "hours_since_spill",
    "spills_7d",
    "hour",
    "month",
    "site_code",
]
CATEGORICAL = ["site_code"]


def target_name(horizon=HORIZON_H):
    return f"spill_start_next_{horizon}h"


def merge_episodes(events, gap=MERGE_GAP):
    """Join spills at the same overflow that are less than `gap` apart.

    Monitors often flicker on and off through one storm; this turns those
    bursts into single episodes so a "new spill" means a genuinely new one.
    """
    gap = pd.Timedelta(gap)
    ev = events.sort_values(["site_id", "start"])
    # an event starts a new episode if it begins `gap` or more after every
    # earlier event at that site has ended
    prev_end = ev.groupby("site_id")["end"].transform(lambda s: s.cummax().shift())
    new = prev_end.isna() | (ev["start"] - prev_end >= gap)
    episode = new.cumsum()
    out = ev.groupby(episode).agg(
        site_id=("site_id", "first"), start=("start", "min"), end=("end", "max")
    )
    return out.reset_index(drop=True)


def _rain_features(rain):
    """Trailing rain sums and peak hourly intensity, all ending at t."""
    out = pd.DataFrame(index=rain.index)
    for name, hours in RAIN_WINDOWS_H.items():
        out[f"rain_{name}"] = rain.rolling(hours, min_periods=1).sum()
    out["rain_max_1h_24h"] = rain.rolling(24, min_periods=1).max()
    return out


def _spill_features(times, episodes, horizon):
    """Spill history features and the target for one site, via binary search."""
    # a sentinel episode in 1970 means every lookup has something to index;
    # it is far outside every window so it never changes a count
    epoch = np.array(["1970-01-01"], dtype="datetime64[ns]")
    starts = np.concatenate([epoch, episodes["start"].to_numpy("datetime64[ns]")])
    ends = np.concatenate([epoch, episodes["end"].to_numpy("datetime64[ns]")])
    t = times.to_numpy(dtype="datetime64[ns]")
    hour = np.timedelta64(1, "h")

    started = np.searchsorted(starts, t, side="right")  # starts <= t
    spilling = ends[started - 1] > t

    ended = np.searchsorted(ends, t, side="right") - 1  # last end <= t
    since = (t - ends[ended]) / hour
    since = np.where(spilling, 0.0, np.minimum(since, HOURS_SINCE_CAP))

    week_ago = np.searchsorted(starts, t - 168 * hour, side="right")
    ahead = np.searchsorted(starts, t + horizon * hour, side="right")
    return pd.DataFrame(
        {
            "spilling_now": spilling.astype(int),
            "hours_since_spill": since,
            "spills_7d": started - week_ago,
            target_name(horizon): (ahead > started).astype(int),
        },
        index=times,
    )


def build_panel(episodes, rain, horizon=HORIZON_H, warmup=WARMUP):
    """Hourly table for every site in `rain`, joined to its spill episodes.

    `rain` has site_id, timestamp (UTC, hourly) and rain_mm. Rows start after
    `warmup` so every rolling window is full, and stop `horizon` hours before
    the last rain hour so the target never looks past the data.
    """
    sites = sorted(rain["site_id"].unique())
    frames = []
    for code, site in enumerate(sites):
        r = rain[rain["site_id"] == site].set_index("timestamp")["rain_mm"]
        r = r.sort_index().asfreq("1h").fillna(0.0)
        ep = episodes[episodes["site_id"] == site].sort_values("start")
        part = pd.concat(
            [_rain_features(r), _spill_features(r.index, ep, horizon)], axis=1
        )
        first = r.index[0] + warmup
        last = r.index[-1] - pd.Timedelta(hours=horizon)
        part = part.loc[first:last]
        part.insert(0, "site_id", site)
        part["site_code"] = code
        frames.append(part)

    panel = pd.concat(frames).rename_axis("timestamp").reset_index()
    panel["hour"] = panel["timestamp"].dt.hour
    panel["month"] = panel["timestamp"].dt.month
    rain_cols = [c for c in panel if c.startswith("rain_")]
    panel[rain_cols] = panel[rain_cols].round(2)
    cols = ["timestamp", "site_id", *FEATURES, target_name(horizon)]
    return panel[cols]
