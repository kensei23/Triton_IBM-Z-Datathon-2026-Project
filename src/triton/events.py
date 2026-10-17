import json
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


def load_event(path):
    """Read an event json and fail early if the schema is missing fields."""
    event = json.loads(Path(path).read_text())
    missing = [key for key in REQUIRED if key not in event]
    if missing:
        raise ValueError(f"event is missing fields: {missing}")
    return event
