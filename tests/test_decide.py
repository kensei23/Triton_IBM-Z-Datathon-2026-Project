import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triton.agent.decide import decide  # noqa: E402

MOCK = json.loads((ROOT / "src/triton/agent/mock_event.json").read_text())


def make(prob, eta):
    return {**MOCK, "pollution_probability": prob, "eta_minutes_to_intake": eta}


def test_mock_event_closes_intake():
    result = decide(MOCK)
    assert result["close_intake"] is True
    assert result["alert_ea"] is True
    assert result["severity"] == "high"
    assert result["probability_pct"] == 91
    assert result["eta_minutes"] == 95


def test_confident_but_far_away_only_alerts():
    result = decide(make(0.9, 200))
    assert result["close_intake"] is False
    assert result["alert_ea"] is True
    assert result["severity"] == "medium"


def test_eta_boundary():
    assert decide(make(0.8, 179))["close_intake"] is True
    assert decide(make(0.8, 180))["close_intake"] is False


def test_low_probability_does_nothing():
    result = decide(make(0.3, 60))
    assert result["close_intake"] is False
    assert result["alert_ea"] is False
    assert result["severity"] == "low"
