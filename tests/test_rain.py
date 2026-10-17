import pytest

from triton.data.rain import bng_to_osgb36, bng_to_wgs84, parse_archive


def test_os_worked_example():
    # Ordnance Survey guide, annex C: 52 39 27.2531 N, 1 43 4.5177 E
    lat, lon = bng_to_osgb36(651409.903, 313177.270)
    assert float(lat) == pytest.approx(52 + 39 / 60 + 27.2531 / 3600, abs=1e-6)
    assert float(lon) == pytest.approx(1 + 43 / 60 + 4.5177 / 3600, abs=1e-6)


def test_big_ben_in_wgs84():
    lat, lon = bng_to_wgs84(530268, 179640)
    assert float(lat) == pytest.approx(51.5007, abs=1e-3)
    assert float(lon) == pytest.approx(-0.1246, abs=1e-3)


def test_works_on_arrays():
    lat, lon = bng_to_wgs84([530268, 538874], [179640, 177344])
    assert lat.shape == (2,)


def test_parse_archive():
    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "precipitation": [0.0, None],
        }
    }
    df = parse_archive(payload, "TWL1")
    assert list(df.columns) == ["site_id", "timestamp", "rain_mm"]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["rain_mm"].isna().sum() == 1
