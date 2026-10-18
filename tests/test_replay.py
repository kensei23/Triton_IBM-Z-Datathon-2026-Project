import numpy as np
import pandas as pd

from triton import replay

T0 = pd.Timestamp("2026-01-10", tz="UTC")
THRESHOLDS = {"notify": 0.05, "intervene": 0.12}


class FixedModel:
    """Probability is the row's `p` column, so tests choose it directly."""

    def predict_proba(self, X):
        return np.column_stack([1 - X["p"], X["p"]])


BUNDLE = {"model": FixedModel(), "features": ["p"]}


def hour(h):
    return T0 + pd.Timedelta(hours=h)


def panel(p_by_hour, spilling=()):
    return pd.DataFrame(
        {
            "timestamp": [hour(h) for h in range(len(p_by_hour))],
            "site_id": "A",
            "p": p_by_hour,
            "rain_1h": 0.0,
            "spilling_now": [int(h in spilling) for h in range(len(p_by_hour))],
        }
    )


def episodes(*starts):
    start = pd.to_datetime([hour(h) for h in starts], utc=True)
    return pd.DataFrame(
        {"site_id": "A", "start": start, "end": start + pd.Timedelta(hours=1)}
    )


def run(p, ep, spilling=()):
    return list(
        replay.replay(panel(p, spilling), ep, BUNDLE, THRESHOLDS, hour(0), hour(len(p)))
    )


def test_alerts_only_on_escalation():
    hours = run([0.01, 0.06, 0.07, 0.2, 0.2, 0.01, 0.06], episodes())
    raised = [(i, e["level"]) for i, h in enumerate(hours) for e in h["escalations"]]
    # NOTIFY once, INTERVENE once, then NOTIFY again after dropping back
    assert raised == [(1, "NOTIFY"), (3, "INTERVENE"), (6, "NOTIFY")]


def test_spilling_sites_are_not_scored():
    hours = run([0.2, 0.2, 0.2], episodes(), spilling={1})
    assert hours[1]["levels"]["A"] == "SPILLING"
    assert hours[1]["escalations"] == []
    # after the spill the site starts from MONITOR again, so it re-alerts
    assert [e["level"] for e in hours[2]["escalations"]] == ["INTERVENE"]


def test_spill_start_reported_in_its_hour():
    ep = episodes(3)
    ep["start"] = ep["start"] - pd.Timedelta(minutes=20)  # 02:40
    hours = run([0.01] * 6, ep)
    assert [len(h["spill_starts"]) for h in hours] == [0, 0, 0, 1, 0, 0]


def test_warning_lead_and_score():
    p = [0.01] * 8 + [0.06, 0.06, 0.06] + [0.01] * 10
    ep = episodes(11, 18)  # first warned 3 h ahead, second not at all
    hours = run(p, ep)
    assert replay.warning_lead(hours, "A", hour(11)) == 3
    assert replay.warning_lead(hours, "A", hour(18)) is None
    result = replay.score(hours, ep, hour(0), hour(len(p)))
    assert result["spills"] == 2 and result["warned"] == 1
    assert result["median_lead_h"] == 3
    assert result["alerts"] == 1 and result["false_alarms"] == 0


def test_false_alarm_counted():
    hours = run([0.01, 0.2] + [0.01] * 10, episodes())
    assert replay.score(hours, episodes(), hour(0), hour(12))["false_alarms"] == 1


def test_pick_storm_covers_the_busiest_day():
    starts = [5, 300, 301, 302, 303, 310, 600]
    t0, t1 = replay.pick_storm(episodes(*starts), hour(0), hours=36, lead_h=12)
    assert t1 - t0 == pd.Timedelta(hours=36)
    assert all(t0 <= hour(h) <= t1 for h in (300, 303, 310))
