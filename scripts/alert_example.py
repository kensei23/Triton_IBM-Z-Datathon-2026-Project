"""Explain one prediction and write its alert, to check the decision and LLM step.

Usage: uv run python scripts/alert_example.py [--no-llm]
       uv run python scripts/alert_example.py --site TWL00452 --time "2026-02-03 08:00"

Without --site/--time it picks the test-period hour with the highest spill
probability. Granite runs locally through llama-cpp-python; if the model file
or library isn't available, the template alert is used and the reason printed.
"""

import argparse

import pandas as pd

from triton import alerts
from triton import model as m
from triton.constants import PROCESSED_DIR, RAW_DIR
from triton.decision import decide, load_thresholds


def load_tables():
    panel = pd.read_csv(PROCESSED_DIR / "model_table.csv.gz")
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    episodes = pd.read_csv(PROCESSED_DIR / "episodes.csv.gz")
    for col in ("start", "end"):
        episodes[col] = pd.to_datetime(episodes[col], utc=True, format="ISO8601")
    sites = pd.read_csv(RAW_DIR / "sites_selected.csv.gz").set_index("site_id")
    return panel, episodes, sites


def main(site_id, when, use_llm):
    bundle = m.load_bundle()
    thresholds = load_thresholds()
    panel, episodes, sites = load_tables()

    if site_id and when:
        t = pd.Timestamp(when, tz="UTC")
        row = panel[(panel["site_id"] == site_id) & (panel["timestamp"] == t)].iloc[0]
        p = m.predict(bundle, row.to_frame().T)[0]
    else:
        test = m.at_risk(panel[panel["timestamp"] >= bundle["test_from"]])
        probs = m.predict(bundle, test)
        row, p = test.iloc[probs.argmax()], probs.max()

    decision = decide(p, thresholds)
    site = {"site_id": row["site_id"], **sites.loc[row["site_id"]].to_dict()}
    summary = alerts.build_summary(
        site,
        row["timestamp"],
        p,
        decision,
        alerts.drivers(bundle, row),
        alerts.similar_event(panel, episodes, row["site_id"], row["timestamp"]),
    )
    print("summary:", summary)
    print("facts:  ", alerts.facts_line(summary))

    llm = None
    if use_llm:
        llm, why = alerts.load_granite()
        print("granite:", why if llm else f"not available ({why})")
    result = alerts.generate_alert(summary, llm)
    print(f"\n[{decision}] {summary['time']}\n{result['text']}")
    print(f"(source: {result['source']}, {result['seconds']}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", help="site id, e.g. TWL00452")
    parser.add_argument("--time", help="UTC hour, e.g. '2026-02-03 08:00'")
    parser.add_argument("--no-llm", action="store_true", help="template only")
    args = parser.parse_args()
    main(args.site, args.time, not args.no_llm)
