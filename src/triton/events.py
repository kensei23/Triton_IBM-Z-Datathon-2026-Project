import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

REQUIRED = [
    "station_id",
    "timestamp",
    "pollution_probability",
    "top_features",
    "eta_minutes_to_intake",
    "intake_id",
    "upstream_spills",
]

# feature names go straight into the explanation sentence so keep them plain
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_number(value):
    # bool is a subclass of int so it has to be ruled out separately
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value):
    return isinstance(value, str) and bool(value.strip())


def _parse_time(value):
    """Parse an ISO 8601 string, None if it's bad or has no timezone."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    # explain.py subtracts times so naive and aware ones can't be mixed
    if parsed.tzinfo is None:
        return None
    return parsed


def _check_features(features):
    if not isinstance(features, list) or not features:
        return ["top_features must be a non-empty list"]

    errors = []
    for i, feat in enumerate(features):
        label = f"top_features[{i}]"
        if not isinstance(feat, dict):
            errors.append(f"{label} must be an object")
            continue
        name = feat.get("name")
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            errors.append(f"{label}.name must be lowercase with underscores")
        if not _is_number(feat.get("shap")):
            errors.append(f"{label}.shap must be a number")
    return errors


def _check_spills(spills, event_time):
    if not isinstance(spills, list):
        return ["upstream_spills must be a list (empty is fine)"]

    errors = []
    for i, spill in enumerate(spills):
        label = f"upstream_spills[{i}]"
        if not isinstance(spill, dict):
            errors.append(f"{label} must be an object")
            continue
        if not _is_text(spill.get("site")):
            errors.append(f"{label}.site must be a non-empty string")
        start = _parse_time(spill.get("start"))
        if start is None:
            errors.append(f"{label}.start must be ISO 8601 with a timezone")
        elif event_time is not None and start > event_time:
            errors.append(f"{label}.start is after the event timestamp")
        duration = spill.get("duration_min")
        if not _is_number(duration) or duration < 0:
            errors.append(f"{label}.duration_min must be a number, 0 or more")
    return errors


def validate_event(event):
    """Return a list of problems with the event, empty list means it's valid."""
    if not isinstance(event, dict):
        return ["event must be a json object"]

    missing = [key for key in REQUIRED if key not in event]
    if missing:
        # no point checking values when whole fields are gone
        return [f"missing fields: {missing}"]

    errors = []
    for key in ("station_id", "intake_id"):
        if not _is_text(event[key]):
            errors.append(f"{key} must be a non-empty string")

    event_time = _parse_time(event["timestamp"])
    if event_time is None:
        errors.append("timestamp must be ISO 8601 with a timezone, e.g. ...Z")

    prob = event["pollution_probability"]
    if not _is_number(prob) or not 0 <= prob <= 1:
        errors.append("pollution_probability must be a number from 0 to 1")

    eta = event["eta_minutes_to_intake"]
    if not _is_int(eta) or eta < 0:
        errors.append("eta_minutes_to_intake must be a whole number, 0 or more")

    errors += _check_features(event["top_features"])
    errors += _check_spills(event["upstream_spills"], event_time)
    return errors


def load_event(path):
    """Read an event json and raise if it doesn't match the schema."""
    try:
        event = json.loads(Path(path).read_text())
    except json.JSONDecodeError as err:
        raise ValueError(f"{path} is not valid json: {err}") from err

    errors = validate_event(event)
    if errors:
        raise ValueError("invalid event: " + "; ".join(errors))
    return event


if __name__ == "__main__":
    # quick check from the terminal: python -m triton.events some_event.json
    problems = validate_event(json.loads(Path(sys.argv[1]).read_text()))
    print("\n".join(problems) or "OK")
    sys.exit(1 if problems else 0)
