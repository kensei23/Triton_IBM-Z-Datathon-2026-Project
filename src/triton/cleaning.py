"""Cleaning sensor and gauge readings onto an hourly grid.

Faults (impossible values, frozen sensors) are removed. Large spikes are flagged
in the quality report but kept, because a sudden spike is the pollution signal.
"""

import pandas as pd

from triton.config import (
    FAULT_PERIODS,
    MAX_GAP_HOURS,
    OUTLIER_Z,
    STUCK_HOURS,
    USABLE_QUALITY,
    VALID_RANGE,
)
from triton.labels import robust_z


def usable(df):
    if "quality" not in df:
        return df
    return df[df["quality"].isin(USABLE_QUALITY) | df["quality"].isna()]


def to_hourly(df, how="mean"):
    s = usable(df).set_index("timestamp")["value"].sort_index()
    return (
        s.resample("1h").sum(min_count=1) if how == "sum" else s.resample("1h").mean()
    )


def stuck_mask(s, hours=STUCK_HOURS):
    """Flags whole runs of identical readings lasting at least `hours`."""
    run = (~s.diff().eq(0)).cumsum()
    times = s.index.to_series()
    step = times.diff().median()
    duration = times.groupby(run).transform("max") - times.groupby(run).transform("min")
    return s.notna() & (duration + step >= pd.Timedelta(hours=hours))


def fill_short_gaps(s, max_hours=MAX_GAP_HOURS):
    """Interpolates gaps of up to max_hours between readings. Longer gaps stay empty
    in full; interpolate(limit=...) alone would fill the start of a long gap."""
    missing = s.isna()
    run_length = missing.groupby((missing != missing.shift()).cumsum()).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return filled.where(~missing | (run_length <= max_hours))


def gap_lengths(s):
    missing = s.isna()
    runs = (missing != missing.shift()).cumsum()
    lengths = missing.groupby(runs).sum()
    return lengths[lengths > 0]


def in_periods(index, periods):
    mask = pd.Series(False, index=index)
    for start, end in periods:
        mask |= (index >= pd.Timestamp(start, tz="UTC")) & (
            index < pd.Timestamp(end, tz="UTC")
        )
    return mask


def clean_series(raw, prop, faults=()):
    """Removes faults from raw readings, then averages to hours and fills short
    gaps. Faults are checked before averaging so they cannot hide in a mean."""
    s = usable(raw).set_index("timestamp")["value"].sort_index()
    lo, hi = VALID_RANGE[prop]
    fault = in_periods(s.index, faults) & s.notna()
    out_of_range = ~fault & s.notna() & ~s.between(lo, hi)
    stuck = stuck_mask(s.mask(fault | out_of_range))
    hourly = s.resample("1h").mean()
    removed = s.mask(fault | out_of_range | stuck).resample("1h").mean()
    filled = fill_short_gaps(removed)
    gaps = gap_lengths(removed)
    report = {
        "hours": len(hourly),
        "missing_raw_pct": round(hourly.isna().mean() * 100, 2),
        "fault_period_readings": int(fault.sum()),
        "out_of_range_readings": int(out_of_range.sum()),
        "stuck_readings": int(stuck.sum()),
        "filled_hours": int(filled.notna().sum() - removed.notna().sum()),
        "missing_final_pct": round(filled.isna().mean() * 100, 2),
        "longest_gap_h": int(gaps.max()) if len(gaps) else 0,
        "large_spikes": int((robust_z(filled).abs() > OUTLIER_Z).sum()),
    }
    return filled, report


def clean_readings(readings):
    parts, report = [], []
    for (sid, prop), group in readings.groupby(["station_id", "measure"]):
        faults = FAULT_PERIODS.get((sid, prop), [])
        filled, stats = clean_series(group, prop, faults)
        report.append({"station_id": sid, "measure": prop, **stats})
        parts.append(
            filled.dropna()
            .rename("value")
            .reset_index()
            .assign(station_id=sid, measure=prop)
        )
    cols = ["station_id", "timestamp", "measure", "value"]
    return pd.concat(parts, ignore_index=True)[cols], pd.DataFrame(report)


def clean_gauges(gauge_readings):
    """Rain is summed per hour (an amount); flow is averaged (a rate)."""
    parts = []
    for (sid, kind), group in gauge_readings.groupby(["station_id", "kind"]):
        s = to_hourly(group, how="sum" if kind == "rain" else "mean")
        parts.append(
            s.mask(s < 0)
            .rename("value")
            .reset_index()
            .assign(station_id=sid, kind=kind)
        )
    return pd.concat(parts, ignore_index=True)[
        ["station_id", "timestamp", "kind", "value"]
    ]
