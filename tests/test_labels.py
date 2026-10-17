import numpy as np
import pandas as pd

from triton.labels import anomaly_lift, robust_z, station_labels


def series(n=300, seed=0):
    index = pd.date_range("2026-02-01", periods=n, freq="1h", tz="UTC")
    return pd.Series(np.random.default_rng(seed).normal(5, 0.2, n), index=index)


def test_baseline_ignores_current_and_future_values():
    s = series()
    changed = s.copy()
    changed.iloc[200:] += 100
    z_before, z_after = robust_z(s), robust_z(changed)
    assert np.allclose(z_before.iloc[:200], z_after.iloc[:200], equal_nan=True)


def test_label_needs_both_spill_and_anomaly():
    wide = pd.DataFrame(
        {"dissolved-oxygen": series(seed=1), "ammonium": series(seed=2)}
    )
    wide.iloc[150, 1] = 50
    wide.iloc[250, 1] = 50
    events = pd.DataFrame(
        {
            "start": [pd.Timestamp("2026-02-07 04:00", tz="UTC")],
            "end": [pd.Timestamp("2026-02-07 05:00", tz="UTC")],
        }
    )
    labels = station_labels(wide, events).set_index("timestamp")
    assert labels["label"].iloc[150] == 1
    assert labels["label"].iloc[250] == 0


def test_no_baseline_means_no_label():
    wide = pd.DataFrame(
        {"dissolved-oxygen": series(seed=1), "ammonium": series(seed=2)}
    )
    events = pd.DataFrame(columns=["start", "end"])
    labels = station_labels(wide, events)
    assert labels["label"].iloc[:24].isna().all()
    assert labels["label"].iloc[48:].notna().all()


def test_anomaly_lift_detects_real_effect():
    rng = np.random.default_rng(0)
    n = 20000
    labels = pd.DataFrame(
        {
            "spill_recent": rng.random(n) < 0.1,
            "z_do": rng.normal(0, 1, n),
            "z_nh4": rng.normal(0, 1, n),
        }
    )
    assert 0.7 < anomaly_lift(labels) < 1.4
    labels.loc[labels["spill_recent"], "z_nh4"] += 2
    assert anomaly_lift(labels) > 5
