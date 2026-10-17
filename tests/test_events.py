import copy
import json
from pathlib import Path

import pytest

from triton.events import load_event, validate_event

ROOT = Path(__file__).resolve().parent.parent
MOCK_PATH = ROOT / "src/triton/agent/mock_event.json"
MOCK = json.loads(MOCK_PATH.read_text())

GOOD_SPILL = {
    "site": "Demo CSO",
    "start": "2026-10-17T12:30:00Z",
    "duration_min": 40,
}


def make(**changes):
    data = copy.deepcopy(MOCK)
    data.update(changes)
    return data


# (field, bad value, text that should appear in the error)
BAD_VALUES = [
    ("pollution_probability", 91, "pollution_probability"),
    ("pollution_probability", -0.1, "pollution_probability"),
    ("pollution_probability", "0.9", "pollution_probability"),
    ("pollution_probability", True, "pollution_probability"),
    ("pollution_probability", float("nan"), "pollution_probability"),
    ("eta_minutes_to_intake", 95.5, "eta_minutes_to_intake"),
    ("eta_minutes_to_intake", -1, "eta_minutes_to_intake"),
    ("eta_minutes_to_intake", "95", "eta_minutes_to_intake"),
    ("eta_minutes_to_intake", True, "eta_minutes_to_intake"),
    ("timestamp", "2026-10-17T14:15:00", "timestamp"),
    ("timestamp", "yesterday", "timestamp"),
    ("timestamp", 123, "timestamp"),
    ("station_id", "", "station_id"),
    ("intake_id", None, "intake_id"),
    ("top_features", [], "top_features"),
    ("top_features", "ammonium", "top_features"),
    ("top_features", [{"name": "Dissolved Oxygen", "shap": -0.4}], "name"),
    ("top_features", [{"name": "ph"}], "shap"),
    ("upstream_spills", "none", "upstream_spills"),
    ("upstream_spills", [{**GOOD_SPILL, "start": "bad"}], "start"),
    ("upstream_spills", [{**GOOD_SPILL, "start": "2026-10-17T15:00:00Z"}], "after"),
    ("upstream_spills", [{**GOOD_SPILL, "duration_min": -5}], "duration_min"),
    ("upstream_spills", [{**GOOD_SPILL, "site": ""}], "site"),
]


def test_mock_event_is_valid():
    assert validate_event(MOCK) == []


def test_empty_spill_list_is_valid():
    assert validate_event(make(upstream_spills=[])) == []


@pytest.mark.parametrize(("field", "value", "fragment"), BAD_VALUES)
def test_bad_value_is_reported(field, value, fragment):
    errors = validate_event(make(**{field: value}))
    assert errors
    assert any(fragment in err for err in errors)


def test_missing_field_is_reported():
    data = make()
    del data["eta_minutes_to_intake"]
    errors = validate_event(data)
    assert len(errors) == 1
    assert "eta_minutes_to_intake" in errors[0]


@pytest.mark.parametrize("not_an_event", [None, [], "event", 5])
def test_non_object_is_rejected(not_an_event):
    assert validate_event(not_an_event)


def test_all_problems_come_back_together():
    errors = validate_event(make(pollution_probability=91, eta_minutes_to_intake=-5))
    assert len(errors) == 2


def test_load_event_reads_good_file():
    assert load_event(MOCK_PATH) == MOCK


def test_load_event_rejects_bad_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(make(pollution_probability=91)))
    with pytest.raises(ValueError, match="pollution_probability"):
        load_event(bad)


def test_load_event_rejects_broken_json(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="not valid json"):
        load_event(broken)
