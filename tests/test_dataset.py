import pandas as pd
import pytest

from triton.dataset import (
    COLUMNS,
    build_dataset,
    check_dataset,
    future_target,
    spill_features,
)


def inputs():
    times = pd.date_range("2026-02-01", periods=5, freq="1h", tz="UTC")
    readings = pd.DataFrame(
        {
            "station_id": "S1",
            "timestamp": list(times[[0, 1, 4]]) * 2,
            "measure": ["dissolved-oxygen"] * 3 + ["ammonium"] * 3,
            "value": [5.0, 5.1, 5.2, 1.6, 1.7, 1.8],
        }
    )
    gauges = pd.DataFrame(
        {
            "station_id": "S1",
            "timestamp": times[:2],
            "kind": "rain",
            "value": [0.0, 2.0],
        }
    )
    labels = pd.DataFrame(
        {"station_id": "S1", "timestamp": times, "label": [pd.NA, 0, 0, 1, 0]}
    )
    events = {
        "S1": pd.DataFrame(
            {
                "start": [pd.Timestamp("2026-02-01 01:30", tz="UTC")],
                "end": [pd.Timestamp("2026-02-01 02:10", tz="UTC")],
            }
        )
    }
    return readings, gauges, labels, events


def test_one_row_per_station_hour_on_a_regular_grid():
    dataset = build_dataset(*inputs())
    assert list(dataset.columns) == COLUMNS
    assert len(dataset) == 5
    assert dataset["dissolved_oxygen_mg_l"].isna().sum() == 2
    assert dataset["turbidity_ntu"].isna().all()
    assert pd.isna(dataset.loc[0, "label"])


def test_duplicate_rows_rejected():
    dataset = build_dataset(*inputs())
    with pytest.raises(ValueError, match="duplicate"):
        check_dataset(pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True))


def test_spill_features_use_only_past_and_present():
    times = pd.date_range("2026-02-01", periods=6, freq="1h", tz="UTC")
    events = pd.DataFrame(
        {
            "start": [pd.Timestamp("2026-02-01 01:30", tz="UTC")],
            "end": [pd.Timestamp("2026-02-01 02:10", tz="UTC")],
        }
    )
    f = spill_features(times, events)
    assert list(f["upstream_spill_active"]) == [0, 1, 1, 0, 0, 0]
    assert pd.isna(f["hours_since_spill"].iloc[0])
    assert list(f["hours_since_spill"].iloc[1:]) == [0, 1, 2, 3, 4]


def test_target_looks_ahead_only():
    label = pd.Series([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype="Int8")
    target = future_target(label, hours=2)
    assert list(target.iloc[:5]) == [0, 1, 1, 0, 0]
    assert pd.isna(target.iloc[-1])
