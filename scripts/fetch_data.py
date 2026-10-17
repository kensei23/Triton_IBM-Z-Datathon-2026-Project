"""Download spill events and hourly rain for the busiest Thames Water overflows.

Usage: uv run python scripts/fetch_data.py [--top-n 40] [--refresh]

Every download is cached in data/raw, so rerunning only fetches what's missing.
--refresh downloads the spill sources again (rain is cached per site and date
range, delete data/raw/rain/ to refetch it).
"""

import argparse

import pandas as pd

from triton.constants import RAW_DIR
from triton.data import rain, spills

START = "2024-01-01"
# API alerts are fetched back into the EA 2025 file so the two can be compared
ALERTS_SINCE = pd.Timestamp("2025-10-01", tz="UTC")


def cached(path, fetch, refresh):
    if path.exists() and not refresh:
        print(f"  using cached {path.name}")
        return pd.read_csv(path)
    df = fetch()
    df.to_csv(path, index=False)
    print(f"  saved {path.name} ({len(df)} rows)")
    return df


def main(top_n, refresh):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Overflow sites")
    sites = spills.tidy_sites(
        cached(RAW_DIR / "tw_sites.csv.gz", spills.download_sites, refresh)
    )

    print("Spill history (EA EDM files)")
    ea = pd.concat(
        [
            spills.ea_to_events(
                cached(
                    RAW_DIR / f"ea_edm_{year}.csv.gz",
                    lambda n=name: spills.download_ea_file(n),
                    refresh,
                )
            )
            for year, name in spills.EA_FILES.items()
        ]
    )

    print("Recent spills (Thames Water API)")
    alerts = cached(
        RAW_DIR / "tw_alerts.csv.gz",
        lambda: spills.download_alerts(ALERTS_SINCE),
        refresh,
    )
    api = spills.alerts_to_events(alerts, sites)

    events = spills.combine_events(ea, api)
    events = events[events["start"] >= pd.Timestamp(START, tz="UTC")]
    events.to_csv(RAW_DIR / "spills.csv.gz", index=False)
    print(f"  {len(events)} spill events at {events['site_id'].nunique()} overflows")
    print(events.groupby([events["start"].dt.year, "source"]).size().to_string())

    chosen = spills.top_sites(events, top_n)
    picked = sites[sites["site_id"].isin(chosen)].copy()
    picked["lat"], picked["lon"] = rain.bng_to_wgs84(
        picked["easting"], picked["northing"]
    )
    picked["n_events"] = picked["site_id"].map(events["site_id"].value_counts())
    picked = picked.sort_values("n_events", ascending=False)
    picked.to_csv(RAW_DIR / "sites_selected.csv.gz", index=False)
    print(f"Selected {len(picked)} overflows (most spills, data in every year)")
    print(picked[["site_id", "name", "river", "n_events"]].to_string(index=False))

    # the archive lags real time by a few days
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    print(f"Hourly rain {START} to {end} (Open-Meteo)")
    hourly = rain.fetch_site_rain(picked, START, end, RAW_DIR / "rain")
    hourly.to_csv(RAW_DIR / "rain_hourly.csv.gz", index=False)
    print(f"  {len(hourly)} rows, {hourly['rain_mm'].isna().mean():.1%} missing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=40, help="overflows to keep")
    parser.add_argument(
        "--refresh", action="store_true", help="download spill sources again"
    )
    args = parser.parse_args()
    main(args.top_n, args.refresh)
