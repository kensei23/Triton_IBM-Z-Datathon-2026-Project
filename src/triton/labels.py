"""Weak labels: an hour is a pollution event when an upstream outfall spilled
recently (cause) and the sensors moved away from their normal level (effect)."""

import pandas as pd

from triton.config import BASELINE_MIN_HOURS, BASELINE_WINDOW, SPILL_AFTER, Z_THRESHOLD


def robust_z(s, window=BASELINE_WINDOW, min_periods=BASELINE_MIN_HOURS):
    """Distance from the median of the previous window, scaled by the median
    absolute deviation. shift(1) keeps the current reading out of its own baseline."""
    base = s.rolling(window, min_periods=min_periods).median().shift(1)
    mad = (s - base).abs().rolling(window, min_periods=min_periods).median().shift(1)
    return (s - base) / (1.4826 * mad + 1e-6)


def spill_window(index, events, after=SPILL_AFTER):
    mask = pd.Series(False, index=index)
    for start, end in zip(events["start"], events["end"], strict=True):
        mask |= (index >= start) & (index <= end + pd.Timedelta(after))
    return mask


def station_labels(wide, events, z_thr=Z_THRESHOLD, after=SPILL_AFTER):
    """wide: hourly readings for one station with dissolved-oxygen and ammonium."""
    z_do = robust_z(wide["dissolved-oxygen"])
    z_nh4 = robust_z(wide["ammonium"])
    recent = spill_window(wide.index, events, after)
    anomaly = (z_do < -z_thr) | (z_nh4 > z_thr)
    label = (recent & anomaly).astype("Int8")
    # No baseline yet means no label, not "no pollution".
    label[z_do.isna() & z_nh4.isna()] = pd.NA
    return pd.DataFrame(
        {
            "timestamp": wide.index,
            "spill_recent": recent.to_numpy(),
            "z_do": z_do.to_numpy(),
            "z_nh4": z_nh4.to_numpy(),
            "label": label.to_numpy(),
        }
    )


def anomaly_lift(labels, z_thr=Z_THRESHOLD):
    """How much more often sensors look abnormal inside spill windows than outside.
    Near 1 means the labels are mostly chance."""
    anomaly = (labels["z_do"] < -z_thr) | (labels["z_nh4"] > z_thr)
    inside = anomaly[labels["spill_recent"]].mean()
    outside = anomaly[~labels["spill_recent"]].mean()
    return inside / outside if outside > 0 else float("nan")
