import numpy as np
import pandas as pd
import pytest

from triton import model as m
from triton.features import target_name

T0 = pd.Timestamp("2025-06-01", tz="UTC")
TARGET = target_name(6)


def synthetic_panel(n_hours=400, sites=("A", "B"), seed=0):
    """Rain drives the target, so a model should beat chance easily."""
    rng = np.random.default_rng(seed)
    frames = []
    for code, site in enumerate(sites):
        rain = rng.exponential(1.0, n_hours) * (rng.random(n_hours) < 0.3)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(T0, periods=n_hours, freq="1h"),
                    "site_id": site,
                    **{f: rng.random(n_hours) for f in m.MODEL_FEATURES},
                    "rain_24h": rain * 10,
                    "spilling_now": (rng.random(n_hours) < 0.1).astype(int),
                    "site_code": code,
                    TARGET: (rain > 1.5).astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_time_split_is_by_time_and_drops_spilling_rows():
    panel = synthetic_panel()
    cut = T0 + pd.Timedelta(hours=300)
    train, test = m.time_split(panel, cut)
    assert train["timestamp"].max() < cut <= test["timestamp"].min()
    assert train["spilling_now"].eq(0).all() and test["spilling_now"].eq(0).all()
    assert train["timestamp"].is_monotonic_increasing


def test_model_learns_and_is_a_probability():
    panel = synthetic_panel()
    train, test = m.time_split(panel, T0 + pd.Timedelta(hours=300))
    model = m.make_model(max_iter=30).fit(train[m.MODEL_FEATURES], train[TARGET])
    p = model.predict_proba(test[m.MODEL_FEATURES])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert m.scores(test[TARGET], p)["roc_auc"] > 0.9


def test_rain_baseline_rises_with_rain():
    X = pd.DataFrame({"rain_24h": [0, 0, 5, 5, 20, 20]})
    y = np.array([0, 0, 0, 1, 1, 1])
    p = m.RainBaseline().fit(X, y).predict_proba(X)[:, 1]
    assert np.all(np.diff(p) >= 0)


def test_pick_threshold_meets_its_rule():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    hi = m.pick_threshold(y, p, min_precision=1.0)
    assert hi == pytest.approx(0.6)
    lo = m.pick_threshold(y, p, min_recall=1.0)
    assert lo == pytest.approx(0.4)


def test_threshold_table_counts():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.9, 0.8, 0.1, 0.05])
    row = m.threshold_table(y, p, hours_per_day=1, thresholds=[0.5])[0]
    assert row["precision"] == 0.5 and row["recall"] == 0.5
    assert row["alert_hours_per_day"] == 2


def test_lead_time_is_first_crossing_inside_the_window():
    times = pd.date_range(T0, periods=12, freq="1h")
    rows = pd.DataFrame({"site_id": "A", "timestamp": times})
    p = np.zeros(12)
    p[[1, 6, 7]] = 0.9  # hour 1 is outside the 6h window before the spill
    episodes = pd.DataFrame(
        {"site_id": ["A"], "start": [T0 + pd.Timedelta(hours=10, minutes=30)]}
    )
    leads = m.lead_times(rows, p, episodes, threshold=0.5, horizon=6)
    assert leads.tolist() == [4.5]
    missed = m.lead_times(rows, np.zeros(12), episodes, threshold=0.5, horizon=6)
    assert np.isnan(missed).all()
    assert m.lead_summary(missed)["caught"] == 0
