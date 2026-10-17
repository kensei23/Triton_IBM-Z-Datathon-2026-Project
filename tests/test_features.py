import numpy as np
import pandas as pd
import pytest

from triton.features import FEATURES, build_panel, merge_episodes, target_name

T0 = pd.Timestamp("2025-01-01", tz="UTC")
TARGET = target_name(6)


def hours(n):
    return pd.Timedelta(hours=n)


def episodes(*pairs, site="S1"):
    return pd.DataFrame(
        {
            "site_id": site,
            "start": [T0 + hours(a) for a, _ in pairs],
            "end": [T0 + hours(b) for _, b in pairs],
        }
    )


def rain_series(values, site="S1"):
    return pd.DataFrame(
        {
            "site_id": site,
            "timestamp": pd.date_range(T0, periods=len(values), freq="1h"),
            "rain_mm": values,
        }
    )


def panel(ep, rain, horizon=6):
    out = build_panel(ep, rain, horizon=horizon, warmup=pd.Timedelta(0))
    return out.set_index("timestamp")


def test_merge_joins_flicker_but_not_separate_storms():
    ev = episodes((0, 1), (2, 3), (3.5, 4), (20, 21))
    merged = merge_episodes(ev, gap="6h")
    assert len(merged) == 2
    assert merged.loc[0, "end"] == T0 + hours(4)


def test_merge_keeps_sites_apart():
    ev = pd.concat([episodes((0, 1), site="A"), episodes((2, 3), site="B")])
    assert len(merge_episodes(ev, gap="6h")) == 2


def test_target_marks_hours_before_a_new_spill():
    p = panel(episodes((30, 32)), rain_series(np.zeros(60)))
    flagged = p.index[p[TARGET] == 1]
    # starts at hour 30: decision hours 24..29 see it in their next 6 hours
    assert list(flagged) == [T0 + hours(h) for h in range(24, 30)]


def test_spill_state_features():
    p = panel(episodes((10, 12), (14, 15)), rain_series(np.zeros(40)))
    assert p.loc[T0 + hours(9), "spilling_now"] == 0
    assert p.loc[T0 + hours(10), "spilling_now"] == 1
    assert p.loc[T0 + hours(12), "spilling_now"] == 0
    assert p.loc[T0 + hours(13), "hours_since_spill"] == 1
    assert p.loc[T0 + hours(20), "spills_7d"] == 2


def test_rain_windows_end_at_t():
    rain = np.zeros(48)
    rain[10] = 5.0
    p = panel(episodes(), rain_series(rain))
    assert p.loc[T0 + hours(9), "rain_1h"] == 0
    assert p.loc[T0 + hours(10), "rain_1h"] == 5
    assert p.loc[T0 + hours(12), "rain_3h"] == 5
    assert p.loc[T0 + hours(13), "rain_3h"] == 0
    assert p.loc[T0 + hours(30), "rain_24h"] == 5


def test_panel_stops_before_target_runs_out():
    p = panel(episodes(), rain_series(np.zeros(48)), horizon=6)
    assert p.index.max() == T0 + hours(47 - 6)


@pytest.mark.parametrize("cut", [50, 100, 150])
def test_features_never_use_the_future(cut):
    """Rewriting everything after `cut` must not change any feature up to it."""
    rng = np.random.default_rng(cut)
    n = 240
    rain = rng.exponential(1.0, n) * (rng.random(n) < 0.3)
    ep = episodes((20, 25), (60, 62), (110, 130), (170, 171))
    base = panel(ep, rain_series(rain))

    future_rain = rain.copy()
    future_rain[cut + 1 :] = rng.exponential(5.0, n - cut - 1)
    upto = T0 + hours(cut)
    # spills that started after the cut vanish, a spill still running at the
    # cut ends at a different (still future) time, and a new one appears
    future_ep = ep[ep["start"] <= upto].copy()
    running = future_ep["end"] > upto
    future_ep.loc[running, "end"] = upto + hours(1)
    future_ep = pd.concat([future_ep, episodes((cut + 8, cut + 9))])
    changed = panel(future_ep, rain_series(future_rain))

    pd.testing.assert_frame_equal(
        base.loc[:upto, FEATURES], changed.loc[:upto, FEATURES]
    )
    # the targets do change, which shows the rewrite really touched the future
    assert not base[TARGET].equals(changed[TARGET])
