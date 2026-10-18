import numpy as np
import pandas as pd

from triton import alerts
from triton.features import target_name

T0 = pd.Timestamp("2026-02-01 08:00", tz="UTC")
TARGET = target_name(6)
SITE = {"site_id": "TWL1", "name": "Windsor ", "river": "River Thames"}


class RainModel:
    """Stand-in model: probability rises with 24h rain only."""

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X["rain_24h"].to_numpy() - 10) / 3))
        return np.column_stack([1 - p, p])


BUNDLE = {
    "model": RainModel(),
    "features": ["rain_24h", "rain_7d", "hour"],
    "reference": {"rain_24h": 0.0, "rain_7d": 5.0, "hour": 12},
}


def summary(decision="NOTIFY", similar=True):
    return alerts.build_summary(
        SITE,
        T0,
        0.34,
        decision,
        [{"text": "18 mm of rain in the last 24 hours"}],
        {"date": "3 Feb 2025", "duration_h": 5, "rain_7d": 40.0} if similar else None,
    )


class FakeGranite:
    def __init__(self, text):
        self.text = text

    def write(self, s):
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def fake_llm(text):
    return FakeGranite(text)


def test_drivers_finds_the_feature_that_matters():
    row = pd.Series({"rain_24h": 18.0, "rain_7d": 40.0, "hour": 8})
    found = alerts.drivers(BUNDLE, row)
    assert [d["feature"] for d in found] == ["rain_24h"]
    assert found[0]["text"] == "18 mm of rain in the last 24 hours"


def test_similar_event_is_past_and_closest():
    times = [T0 - pd.Timedelta(days=d) for d in (60, 30, 2)]
    panel = pd.DataFrame(
        {
            "site_id": "TWL1",
            "timestamp": [*times, T0],
            "rain_24h": [2.0, 18.0, 18.0, 18.0],
            "rain_7d": [5.0, 40.0, 40.0, 40.0],
            "rain_30d": [20.0, 90.0, 90.0, 90.0],
            TARGET: [1, 1, 1, 0],
        }
    )
    episodes = pd.DataFrame(
        {
            "site_id": "TWL1",
            "start": [t + pd.Timedelta(hours=3) for t in times],
            "end": [t + pd.Timedelta(hours=8) for t in times],
        }
    )
    sim = alerts.similar_event(panel, episodes, "TWL1", T0)
    # the 2-day-old match is too recent, so the 30-day-old one wins
    assert sim["date"] == f"{(times[1] + pd.Timedelta(hours=3)).day} Jan 2026"
    assert sim["duration_h"] == 5


def test_summary_and_facts():
    s = summary()
    assert s["site_name"] == "Windsor"
    assert s["probability_pct"] == 34
    facts = alerts.facts_line(s)
    assert "Windsor on the River Thames" in facts
    assert "spilled for 5 hours" in facts


def test_prompt_ends_with_headline():
    prompt = alerts.build_prompt(summary())
    assert prompt.endswith("has a raised risk of spilling within 6 hours (34% chance).")


def test_granite_text_used_when_numbers_check_out():
    tail = " Rain of 18 mm in 24 hours is like 3 Feb 2025. Extra words"
    out = alerts.generate_alert(summary(), fake_llm(tail))
    assert out["source"] == "granite"
    assert out["text"].endswith("like 3 Feb 2025.")


def test_invented_numbers_fall_back_to_template():
    out = alerts.generate_alert(summary(), fake_llm(" It will spill for 12 hours."))
    assert out["source"] == "template"
    assert "Main reasons: 18 mm of rain in the last 24 hours." in out["text"]


def test_llm_errors_fall_back_to_template():
    broken = fake_llm(RuntimeError("model crashed"))
    assert alerts.generate_alert(summary(), broken)["source"] == "template"


def test_chat_prompt_has_rules_and_facts():
    msgs = alerts.chat_messages(summary(), "Use only the facts given.")
    assert msgs[0] == {"role": "system", "content": "Use only the facts given."}
    assert "Windsor on the River Thames" in msgs[1]["content"]


def test_alert_config_is_valid():
    cfg = alerts.alert_settings()
    assert cfg["temperature"] == 0
    assert cfg["chat_model"].endswith(".gguf")
    assert "only the facts" in cfg["system"]


def test_intervene_action_comes_from_code():
    out = alerts.generate_alert(summary("INTERVENE"), None)
    assert out["text"].endswith("storm tank emptying to treatment has started.")
    assert "storm tank" not in alerts.generate_alert(summary("NOTIFY"))["text"]


def test_template_without_similar_case():
    out = alerts.generate_alert(summary(similar=False))
    assert "Conditions are like" not in out["text"]


def test_facts_check_catches_wrong_window_and_time_words():
    s = summary()
    assert alerts.facts_ok(" Heavy rain of 18 mm in the last 24 hours.", s)
    # right number, wrong window
    assert not alerts.facts_ok(" Heavy rain of 18 mm in the last 7 days.", s)
    # a time span the facts never mention
    assert not alerts.facts_ok(" The ground is wet from 18 mm this week.", s)


def test_monitor_is_a_log_line_without_the_model():
    out = alerts.generate_alert(summary("MONITOR"), fake_llm(RuntimeError("unused")))
    assert out["source"] == "log"
    assert "low risk" in out["text"]


def test_facts_check_rejects_echoed_chance_and_comparisons():
    s = summary()
    assert not alerts.facts_ok(" The risk is raised due to the 34% chance.", s)
    assert not alerts.facts_ok(
        " Rain of 18 mm in the last 24 hours is higher than 5.", s
    )
