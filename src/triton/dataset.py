"""The modelling table: one row per station and hour, one column per variable."""

import pandas as pd

from triton.config import SENSORS, TARGET_HOURS

GAUGE_COLUMNS = {"rain": "rain_mm", "flow": "flow_m3s"}
SPILL_COLUMNS = ["upstream_spill_active", "hours_since_spill"]
TARGET = f"target_{TARGET_HOURS}h"
COLUMNS = [
    "station_id",
    "timestamp",
    *SENSORS.values(),
    *GAUGE_COLUMNS.values(),
    *SPILL_COLUMNS,
    "label",
    TARGET,
]


def hourly_grid(timestamps):
    return pd.date_range(
        timestamps.min(), timestamps.max(), freq="1h", name="timestamp"
    )


def spill_features(grid, events):
    """What the live spill feed shows at each hour: whether an upstream outfall is
    spilling, and hours since the latest spill started (empty before the first)."""
    starts = events["start"].dt.floor("1h")
    ends = events["end"].dt.floor("1h")
    active = pd.Series(0, index=grid, dtype="Int8")
    for start, end in zip(starts, ends, strict=True):
        active[(grid >= start) & (grid <= end)] = 1
    unique_starts = starts.sort_values().drop_duplicates()
    last_start = pd.Series(unique_starts.to_numpy(), index=unique_starts).reindex(
        grid, method="ffill"
    )
    hours = (grid.to_series() - last_start) / pd.Timedelta("1h")
    return pd.DataFrame(
        {"upstream_spill_active": active, "hours_since_spill": hours}, index=grid
    )


def future_target(label, hours=TARGET_HOURS):
    """1 if any of the next `hours` hours is labelled; empty if none has a label."""
    future = pd.concat([label.shift(-k) for k in range(1, hours + 1)], axis=1)
    return future.max(axis=1).where(future.notna().any(axis=1)).astype("Int8")


def station_table(station_id, readings, gauges, labels, events):
    """Regular hourly grid for one station: missing hours stay as empty cells so
    lags and rolling features line up."""
    sensors = readings.pivot(index="timestamp", columns="measure", values="value")
    sensors = sensors.reindex(columns=list(SENSORS)).rename(columns=SENSORS)
    grid = hourly_grid(sensors.index)
    context = gauges.pivot(index="timestamp", columns="kind", values="value")
    context = context.reindex(columns=list(GAUGE_COLUMNS)).rename(columns=GAUGE_COLUMNS)
    label = labels.set_index("timestamp")["label"].reindex(grid).astype("Int8")
    table = (
        sensors.reindex(grid)
        .join(context.reindex(grid))
        .join(spill_features(grid, events))
        .assign(label=label, **{TARGET: future_target(label)})
    )
    return table.reset_index().assign(station_id=station_id)[COLUMNS]


def build_dataset(readings, gauges, labels, events):
    """events: spill events per station, {station_id: DataFrame with start, end}."""
    tables = [
        station_table(
            sid,
            readings[readings["station_id"] == sid],
            gauges[gauges["station_id"] == sid],
            labels[labels["station_id"] == sid],
            events[sid],
        )
        for sid in readings["station_id"].unique()
    ]
    dataset = pd.concat(tables, ignore_index=True)
    dataset = dataset.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    check_dataset(dataset)
    return dataset


def check_dataset(dataset):
    steps = dataset.groupby("station_id")["timestamp"].diff().dropna()
    checks = {
        "unexpected columns": list(dataset.columns) == COLUMNS,
        "duplicate station-hours": not dataset.duplicated(
            ["station_id", "timestamp"]
        ).any(),
        "timestamps not UTC": str(dataset["timestamp"].dt.tz) == "UTC",
        "grid not regular hourly": (steps == pd.Timedelta("1h")).all(),
        "labels not 0 or 1": all(
            dataset[c].dropna().isin([0, 1]).all()
            for c in ["label", TARGET, "upstream_spill_active"]
        ),
    }
    failed = [problem for problem, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"dataset checks failed: {', '.join(failed)}")
