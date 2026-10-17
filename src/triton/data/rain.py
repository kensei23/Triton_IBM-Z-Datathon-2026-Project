"""Hourly rainfall at each overflow from the Open-Meteo historical archive.

Thames Water gives overflow locations as British National Grid easting and
northing; Open-Meteo wants WGS84 latitude and longitude. pyproj has no s390x
wheel, so the conversion is done here with the Ordnance Survey formulas
("A guide to coordinate systems in Great Britain"), good to a few metres.
"""

import time

import numpy as np
import pandas as pd
import requests

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Airy 1830 ellipsoid and the National Grid projection
AIRY_A, AIRY_B = 6377563.396, 6356256.909
F0 = 0.9996012717
LAT0, LON0 = np.radians(49), np.radians(-2)
N0, E0 = -100000, 400000
# GRS80 / WGS84 ellipsoid
WGS_A, WGS_B = 6378137.0, 6356752.3141
# OSGB36 -> WGS84 Helmert transform (metres, ppm, arc seconds)
HELMERT = {"tx": 446.448, "ty": -125.157, "tz": 542.060, "s": -20.4894}
HELMERT_R = {"rx": 0.1502, "ry": 0.2470, "rz": 0.8421}


def _meridional_arc(lat, n):
    d, s = lat - LAT0, lat + LAT0
    return (
        AIRY_B
        * F0
        * (
            (1 + n + 1.25 * n**2 + 1.25 * n**3) * d
            - (3 * n + 3 * n**2 + 21 / 8 * n**3) * np.sin(d) * np.cos(s)
            + (15 / 8 * n**2 + 15 / 8 * n**3) * np.sin(2 * d) * np.cos(2 * s)
            - 35 / 24 * n**3 * np.sin(3 * d) * np.cos(3 * s)
        )
    )


def bng_to_osgb36(easting, northing):
    """National Grid easting/northing to OSGB36 latitude/longitude in degrees."""
    east = np.asarray(easting, dtype=float)
    north = np.asarray(northing, dtype=float)
    a, b = AIRY_A, AIRY_B
    e2 = 1 - b**2 / a**2
    n = (a - b) / (a + b)

    lat = (north - N0) / (a * F0) + LAT0
    for _ in range(20):
        m = _meridional_arc(lat, n)
        if np.all(np.abs(north - N0 - m) < 1e-5):
            break
        lat = lat + (north - N0 - m) / (a * F0)

    sin2 = np.sin(lat) ** 2
    nu = a * F0 / np.sqrt(1 - e2 * sin2)
    rho = a * F0 * (1 - e2) / (1 - e2 * sin2) ** 1.5
    eta2 = nu / rho - 1
    t, sec = np.tan(lat), 1 / np.cos(lat)
    de = east - E0

    vii = t / (2 * rho * nu)
    viii = t / (24 * rho * nu**3) * (5 + 3 * t**2 + eta2 - 9 * t**2 * eta2)
    ix = t / (720 * rho * nu**5) * (61 + 90 * t**2 + 45 * t**4)
    x = sec / nu
    xi = sec / (6 * nu**3) * (nu / rho + 2 * t**2)
    xii = sec / (120 * nu**5) * (5 + 28 * t**2 + 24 * t**4)
    xiia = sec / (5040 * nu**7) * (61 + 662 * t**2 + 1320 * t**4 + 720 * t**6)

    lat_out = lat - vii * de**2 + viii * de**4 - ix * de**6
    lon_out = LON0 + x * de - xi * de**3 + xii * de**5 - xiia * de**7
    return np.degrees(lat_out), np.degrees(lon_out)


def osgb36_to_wgs84(lat, lon):
    """Shift OSGB36 coordinates onto WGS84 with a Helmert transform."""
    lat, lon = np.radians(lat), np.radians(lon)
    e2 = 1 - AIRY_B**2 / AIRY_A**2
    nu = AIRY_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = nu * np.cos(lat) * np.cos(lon)
    y = nu * np.cos(lat) * np.sin(lon)
    z = (1 - e2) * nu * np.sin(lat)

    s = 1 + HELMERT["s"] * 1e-6
    rx, ry, rz = (np.radians(v / 3600) for v in HELMERT_R.values())
    x2 = HELMERT["tx"] + s * (x - rz * y + ry * z)
    y2 = HELMERT["ty"] + s * (rz * x + y - rx * z)
    z2 = HELMERT["tz"] + s * (-ry * x + rx * y + z)

    e2w = 1 - WGS_B**2 / WGS_A**2
    p = np.hypot(x2, y2)
    lat2 = np.arctan2(z2, p * (1 - e2w))
    for _ in range(10):
        nu2 = WGS_A / np.sqrt(1 - e2w * np.sin(lat2) ** 2)
        lat2 = np.arctan2(z2 + e2w * nu2 * np.sin(lat2), p)
    return np.degrees(lat2), np.degrees(np.arctan2(y2, x2))


def bng_to_wgs84(easting, northing):
    return osgb36_to_wgs84(*bng_to_osgb36(easting, northing))


def parse_archive(payload, site_id):
    """One Open-Meteo archive response to rows of site_id, timestamp, rain_mm."""
    hourly = payload["hourly"]
    return pd.DataFrame(
        {
            "site_id": site_id,
            "timestamp": pd.to_datetime(hourly["time"], utc=True),
            "rain_mm": pd.to_numeric(pd.Series(hourly["precipitation"])),
        }
    )


def fetch_archive(lat, lon, start, end, retries=5):
    params = {
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "start_date": start,
        "end_date": end,
        "hourly": "precipitation",
        "timezone": "GMT",
    }
    for attempt in range(retries):
        r = requests.get(ARCHIVE, params=params, timeout=120)
        if r.status_code == 429:
            # minutely / hourly quota, wait it out
            wait = 60 * (attempt + 1)
            print(f"  Open-Meteo rate limit, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Open-Meteo still rate limiting")


def fetch_site_rain(sites, start, end, cache_dir, pause=8.0):
    """Rain for each site, one cached file per site so reruns resume.

    Long date ranges count as many "calls" in Open-Meteo's free quota (about
    one per 14 days), so requests are spaced out to stay under it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, row in enumerate(sites.itertuples(index=False), 1):
        path = cache_dir / f"{row.site_id}_{start}_{end}.csv.gz"
        if path.exists():
            frames.append(pd.read_csv(path, parse_dates=["timestamp"]))
            continue
        print(f"  rain {i}/{len(sites)}: {row.site_id} {row.name}")
        df = parse_archive(fetch_archive(row.lat, row.lon, start, end), row.site_id)
        df.to_csv(path, index=False)
        frames.append(df)
        time.sleep(pause)
    rain = pd.concat(frames, ignore_index=True)
    rain["timestamp"] = pd.to_datetime(rain["timestamp"], utc=True)
    return rain
