"""Spill prediction model, baseline and evaluation.

The model only scores hours when an overflow is not already spilling: that is
when an early warning is useful, and while it spills a "new" spill can't start.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from triton.features import CATEGORICAL, FEATURES

MODEL_FEATURES = [f for f in FEATURES if f != "spilling_now"]
TEST_FROM = pd.Timestamp("2026-01-01", tz="UTC")
REPORT_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def at_risk(panel):
    return panel[panel["spilling_now"] == 0]


def time_split(panel, test_from=TEST_FROM):
    """Everything before `test_from` trains, everything after tests. Never random."""
    # sorted by time so the calibration folds are blocks of time, not of sites
    rows = at_risk(panel).sort_values(["timestamp", "site_id"])
    return rows[rows["timestamp"] < test_from], rows[rows["timestamp"] >= test_from]


def make_model(max_iter=300, seed=0):
    booster = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        categorical_features=CATEGORICAL,
        early_stopping=False,
        random_state=seed,
    )
    # 3 folds in time order (no shuffle), each calibrated on data it didn't see
    return CalibratedClassifierCV(booster, method="isotonic", cv=3)


class RainBaseline:
    """Rain-only rule: spill chance as a rising function of the last 24h of rain."""

    feature = "rain_24h"

    def fit(self, X, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.iso.fit(X[self.feature], y)
        return self

    def predict_proba(self, X):
        p = self.iso.predict(X[self.feature])
        return np.column_stack([1 - p, p])


def scores(y, p):
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
    }


def threshold_table(y, p, hours_per_day, thresholds=REPORT_THRESHOLDS):
    """Precision, recall and alerts per day at each threshold."""
    y = np.asarray(y)
    rows = []
    for t in thresholds:
        hit = p >= t
        tp = int((hit & (y == 1)).sum())
        rows.append(
            {
                "threshold": t,
                "precision": round(tp / hit.sum(), 3) if hit.any() else None,
                "recall": round(tp / max(y.sum(), 1), 3),
                "alert_hours_per_day": round(hit.sum() / hours_per_day, 2),
            }
        )
    return rows


def pick_threshold(y, p, min_precision=None, min_recall=None):
    """Lowest threshold reaching a precision, or highest reaching a recall."""
    precision, recall, cuts = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    if min_precision is not None:
        ok = np.flatnonzero(precision >= min_precision)
        return float(cuts[ok[0]]) if len(ok) else None
    ok = np.flatnonzero(recall >= min_recall)
    return float(cuts[ok[-1]]) if len(ok) else None


def lead_times(rows, p, episodes, threshold, horizon):
    """For each spill episode in `rows`' period, how early the threshold was crossed.

    Looks at the `horizon` hours before each start (the window the model
    promises), at that site's not-spilling hours. Returns hours of warning
    per episode, NaN where the threshold was never crossed in time.
    """
    scored = rows.assign(p=p)[["site_id", "timestamp", "p"]]
    t0, t1 = rows["timestamp"].min(), rows["timestamp"].max()
    ep = episodes[(episodes["start"] > t0) & (episodes["start"] <= t1)]
    window = pd.Timedelta(hours=horizon)
    by_site = {s: g.set_index("timestamp")["p"] for s, g in scored.groupby("site_id")}
    leads = []
    for e in ep.itertuples():
        site = by_site.get(e.site_id)
        if site is None:
            continue
        before = site[(site.index >= e.start - window) & (site.index < e.start)]
        crossed = before.index[before.to_numpy() >= threshold]
        lead = (
            (e.start - crossed[0]) / pd.Timedelta(hours=1) if len(crossed) else np.nan
        )
        leads.append(lead)
    return np.array(leads, dtype=float)


def lead_summary(leads):
    caught = leads[~np.isnan(leads)]
    return {
        "episodes": int(len(leads)),
        "caught": int(len(caught)),
        "caught_pct": round(100 * len(caught) / max(len(leads), 1), 1),
        "median_lead_h": round(float(np.median(caught)), 2) if len(caught) else None,
        "mean_lead_h": round(float(np.mean(caught)), 2) if len(caught) else None,
    }
