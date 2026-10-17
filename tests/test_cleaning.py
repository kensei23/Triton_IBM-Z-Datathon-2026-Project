import numpy as np
import pandas as pd

from triton.cleaning import clean_gauges, clean_series, to_hourly


def raw(values, freq="1h"):
    times = pd.date_range("2026-02-01", periods=len(values), freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": times, "value": values, "quality": "Good"})


def noisy(n=240, seed=1):
    return list(np.random.default_rng(seed).normal(1.7, 0.1, n))


def test_impossible_values_removed():
    values = noisy()
    values[10] = -3
    filled, report = clean_series(raw(values), "ammonium")
    assert report["out_of_range_readings"] == 1
    assert filled.iloc[10] > 0


def test_whole_frozen_run_removed():
    values = noisy()
    values[50:65] = [1.5] * 15
    _, report = clean_series(raw(values), "ammonium")
    assert report["stuck_readings"] == 15


def test_impossible_value_removed_before_hourly_mean():
    values = noisy(n=960)
    values[400] = -5
    filled, report = clean_series(raw(values, freq="15min"), "dissolved-oxygen")
    assert report["out_of_range_readings"] == 1
    assert filled.iloc[100] > 1.5


def test_frozen_run_detected_at_15_minute_resolution():
    values = noisy(n=960)
    values[200:250] = [1.5] * 50
    _, report = clean_series(raw(values, freq="15min"), "ammonium")
    assert report["stuck_readings"] == 50


def test_short_repeat_not_flagged_as_frozen():
    values = noisy()
    values[50:58] = [1.5] * 8
    _, report = clean_series(raw(values), "ammonium")
    assert report["stuck_readings"] == 0


def test_only_short_gaps_filled():
    values = noisy()
    values[100:105] = [np.nan] * 5
    values[150:152] = [np.nan] * 2
    filled, report = clean_series(raw(values), "ammonium")
    assert filled.iloc[150:152].notna().all()
    assert filled.iloc[100:105].isna().all()
    assert report["longest_gap_h"] == 5


def test_genuine_spike_kept():
    values = noisy()
    values[180] = 8
    filled, report = clean_series(raw(values), "ammonium")
    assert filled.iloc[180] == 8
    assert report["large_spikes"] >= 1


def test_suspect_readings_dropped_before_resampling():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-02-01 00:10", "2026-02-01 00:40"], utc=True
            ),
            "value": [1.0, 99.0],
            "quality": ["Good", "Suspect"],
        }
    )
    assert to_hourly(df).iloc[0] == 1.0


def test_rain_summed_and_flow_averaged():
    times = pd.to_datetime(["2026-02-01 00:00", "2026-02-01 00:30"], utc=True)
    gauges = pd.DataFrame(
        {
            "station_id": "S1",
            "timestamp": list(times) * 2,
            "kind": ["rain", "rain", "flow", "flow"],
            "value": [1.0, 2.0, 4.0, 6.0],
            "quality": "Good",
        }
    )
    result = clean_gauges(gauges).set_index("kind")["value"]
    assert result["rain"] == 3.0
    assert result["flow"] == 5.0


def test_fault_period_removed_even_when_in_range():
    values = noisy()
    filled, report = clean_series(
        raw(values), "ammonium", faults=[("2026-02-03", "2026-02-04")]
    )
    assert report["fault_period_readings"] == 24
    assert filled["2026-02-03"].isna().all()
    assert filled["2026-02-02"].notna().all()
