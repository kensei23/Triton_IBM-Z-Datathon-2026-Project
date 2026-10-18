"""Turn a spill probability into one of three responses.

Thresholds come from configs/decision.json, written by scripts/train.py from
the test-set precision and recall.
"""

import json

from triton.constants import CONFIGS_DIR

DECISION_FILE = CONFIGS_DIR / "decision.json"

ACTIONS = {
    "INTERVENE": "Start emptying the storm tank to treatment and alert the operator",
    "NOTIFY": "Send the operator a plain-English alert",
    "MONITOR": "Log only",
}


def load_thresholds(path=DECISION_FILE):
    return json.loads(path.read_text())


def decide(p, thresholds=None):
    """INTERVENE, NOTIFY or MONITOR for a spill probability p."""
    t = thresholds or load_thresholds()
    if t["notify"] > t["intervene"]:
        raise ValueError("notify threshold must not be above intervene")
    if p >= t["intervene"]:
        return "INTERVENE"
    if p >= t["notify"]:
        return "NOTIFY"
    return "MONITOR"
