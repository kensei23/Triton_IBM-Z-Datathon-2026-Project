import pytest

from triton.decision import ACTIONS, decide, load_thresholds

T = {"intervene": 0.12, "notify": 0.05}


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0, "MONITOR"),
        (0.049, "MONITOR"),
        (0.05, "NOTIFY"),
        (0.119, "NOTIFY"),
        (0.12, "INTERVENE"),
        (0.9, "INTERVENE"),
    ],
)
def test_decide_bands(p, expected):
    assert decide(p, T) == expected


def test_thresholds_in_wrong_order_are_rejected():
    with pytest.raises(ValueError):
        decide(0.5, {"intervene": 0.05, "notify": 0.2})


def test_committed_config_is_usable():
    t = load_thresholds()
    assert 0 < t["notify"] <= t["intervene"] < 1
    assert decide(t["intervene"], t) == "INTERVENE"


def test_every_decision_has_an_action():
    assert set(ACTIONS) == {"INTERVENE", "NOTIFY", "MONITOR"}
