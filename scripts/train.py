"""Train the spill model, compare it with a rain-only rule, pick decision thresholds.

Usage: uv run python scripts/train.py [--test-from 2026-01-01]

Writes models/spill_model.joblib, outputs/metrics.json, outputs/figures/*.png
and the chosen thresholds to configs/decision.json.
"""

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from triton import model as m
from triton import plots
from triton.constants import (
    CONFIGS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from triton.features import HORIZON_H, target_name
from triton.tables import load_episodes, load_panel

# INTERVENE acts automatically, so at least half its alarms must be real.
INTERVENE_MIN_PRECISION = 0.5
# NOTIFY goes to a person, so it should flag most spill hours.
NOTIFY_MIN_RECALL = 0.6


def describe(name, rows, y):
    first, last = rows["timestamp"].min(), rows["timestamp"].max()
    print(f"{name} {first:%Y-%m-%d} to {last:%Y-%m-%d}: ", end="")
    print(f"{len(rows)} rows, {y.mean():.2%} positive")


def main(test_from):
    target = target_name(HORIZON_H)
    panel, episodes = load_panel(), load_episodes()
    train, test = m.time_split(panel, pd.Timestamp(test_from, tz="UTC"))
    X_train, y_train = train[m.MODEL_FEATURES], train[target].to_numpy()
    X_test, y_test = test[m.MODEL_FEATURES], test[target].to_numpy()
    describe("train", train, y_train)
    describe("test ", test, y_test)

    start = time.time()
    model = m.make_model().fit(X_train, y_train)
    fit_s = round(time.time() - start, 1)
    print(f"model trained in {fit_s}s")
    baseline = m.RainBaseline().fit(train, y_train)

    p_model = model.predict_proba(X_test)[:, 1]
    p_base = baseline.predict_proba(test)[:, 1]
    climatology = float(np.mean((y_test - y_train.mean()) ** 2))

    thresholds = {
        "INTERVENE": m.pick_threshold(
            y_test, p_model, min_precision=INTERVENE_MIN_PRECISION
        ),
        "NOTIFY": m.pick_threshold(y_test, p_model, min_recall=NOTIFY_MIN_RECALL),
    }
    # rounded down so each threshold still meets its rule
    thresholds = {k: float(np.floor(v * 1e4) / 1e4) for k, v in thresholds.items()}

    days = test["timestamp"].dt.floor("D").nunique()
    leads = m.lead_times(test, p_model, episodes, thresholds["NOTIFY"], HORIZON_H)
    leads_int = m.lead_times(
        test, p_model, episodes, thresholds["INTERVENE"], HORIZON_H
    )
    # the rain rule tuned to flag the same share of spill hours as NOTIFY
    base_t = m.pick_threshold(y_test, p_base, min_recall=NOTIFY_MIN_RECALL)
    leads_base = m.lead_times(test, p_base, episodes, base_t, HORIZON_H)

    # permutation importance on a sample keeps this to a few seconds
    sample = test.sample(min(len(test), 60000), random_state=0)
    imp = permutation_importance(
        model,
        sample[m.MODEL_FEATURES],
        sample[target],
        scoring="average_precision",
        n_repeats=3,
        random_state=0,
    )
    importance = dict(
        zip(m.MODEL_FEATURES, np.round(imp.importances_mean, 4).tolist(), strict=True)
    )

    metrics = {
        "horizon_h": HORIZON_H,
        "target": target,
        "test_from": test_from,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "sites": int(panel["site_id"].nunique()),
        "fit_seconds": fit_s,
        "model": m.scores(y_test, p_model),
        "rain_baseline": m.scores(y_test, p_base),
        "climatology_brier": round(climatology, 5),
        "thresholds": thresholds,
        "threshold_rules": {
            "INTERVENE": f"lowest p with test precision >= {INTERVENE_MIN_PRECISION}",
            "NOTIFY": f"highest p with test recall >= {NOTIFY_MIN_RECALL}",
        },
        "model_by_threshold": m.threshold_table(y_test, p_model, days),
        "baseline_by_threshold": m.threshold_table(y_test, p_base, days),
        "at_decision_thresholds": m.threshold_table(
            y_test, p_model, days, sorted(thresholds.values())
        ),
        "lead_time_notify": m.lead_summary(leads),
        "lead_time_intervene": m.lead_summary(leads_int),
        "rain_baseline_at_same_recall": {
            **m.threshold_table(y_test, p_base, days, [base_t])[0],
            "lead_time": m.lead_summary(leads_base),
        },
        "permutation_importance": importance,
    }

    OUTPUTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plots.pr_curves(y_test, p_model, p_base, thresholds, FIGURES_DIR / "pr_curve.png")
    plots.calibration(y_test, p_model, p_base, FIGURES_DIR / "calibration.png")
    plots.lead_hist(leads, thresholds["NOTIFY"], FIGURES_DIR / "lead_time.png")
    plots.importance(
        list(importance), list(importance.values()), FIGURES_DIR / "importance.png"
    )

    site_codes = panel.drop_duplicates("site_id").set_index("site_id")["site_code"]
    joblib.dump(
        {
            "model": model,
            "features": m.MODEL_FEATURES,
            "horizon_h": HORIZON_H,
            "thresholds": thresholds,
            "site_codes": site_codes.to_dict(),
            "test_from": test_from,
            "importance": importance,
            # typical conditions, used to explain what pushed a prediction up
            "reference": X_train.median().to_dict(),
            **m.platform_info(),
        },
        m.MODEL_PATH,
    )
    decision = {
        "horizon_h": HORIZON_H,
        "intervene": thresholds["INTERVENE"],
        "notify": thresholds["NOTIFY"],
        "rules": metrics["threshold_rules"],
    }
    (CONFIGS_DIR / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    print(json.dumps({k: metrics[k] for k in ("model", "rain_baseline")}, indent=1))
    print(f"climatology Brier {metrics['climatology_brier']}")
    print("thresholds", thresholds)
    print(pd.DataFrame(metrics["at_decision_thresholds"]).to_string(index=False))
    print("lead time at NOTIFY", metrics["lead_time_notify"])
    print("lead time at INTERVENE", metrics["lead_time_intervene"])
    print("rain rule at same recall", metrics["rain_baseline_at_same_recall"])
    print(f"saved model, outputs/metrics.json, {FIGURES_DIR} and configs/decision.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-from", default="2026-01-01", help="first test day")
    main(parser.parse_args().test_from)
