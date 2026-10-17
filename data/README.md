# Data

Raw data is not committed. One command downloads everything into `data/raw/`
(about 10 minutes the first time, cached after that):

    uv run python scripts/fetch_data.py --top-n 40

`--top-n` sets how many overflows to keep (the ones with the most spills that
report in every year). `--refresh` downloads the spill sources again.

## Storm overflow spills, 2024 and 2025 (target)

- Source: Environment Agency, Event Duration Monitoring (EDM) detailed data, Thames Water
- URL: `https://environment.data.gov.uk/api/file/download?fileDataSetId=c95ecc47-fa70-4a24-b8cc-098232987526&fileName=<file>`
  with files `Thames 2024 Detailed EDM Data.csv` and `Thames Water 2025 EDM Detailed Data.xlsx`
- Licence: Open Government Licence v3.0
- Downloaded: 2026-10-18
- Saved as: `ea_edm_2024.csv.gz`, `ea_edm_2025.csv.gz`

### Notes
- One row per spill with start and stop times in GMT, keyed by the Thames Water unique ID (`TWL…`).
- Both files are really Excel even though the 2024 one is named `.csv`; the code checks the content.
- 2025 has a `Data Correction` column. Rows whose correction says the event was removed or invalid are dropped; `Amended` rows are kept.

## Storm overflow spills, 2026 onwards, and overflow locations

- Source: Thames Water Open Data API v2 (`/discharge/alerts`, `/discharge/status`)
- URL: https://api.thameswater.co.uk/opendata/v2/discharge (docs: https://docs.api.thameswater.co.uk/)
- Licence: Thames Water open data terms, https://data.thameswater.co.uk/s/terms-of-service
- Downloaded: 2026-10-18
- Saved as: `tw_alerts.csv.gz` (raw start/stop alerts), `tw_sites.csv.gz` (every monitored overflow with its grid reference)

### Notes
- No API key is needed for v2. The older `prod-tw-opendata-app` API (which needed a client id and secret) was returning 504 errors when we built this.
- Alerts come newest first, 1,000 per page. Each `Start` is paired with the next `Stop` at the same location; `Offline` alerts mean the monitor was down and are ignored. Spills still running when the data was fetched have no stop yet and are left out.
- Locations are British National Grid easting/northing. They are converted to WGS84 latitude/longitude in pure Python (`src/triton/data/rain.py`).

## Hourly rainfall at each overflow

- Source: Open-Meteo Historical Weather API (reanalysis, about 9 to 11 km grid)
- URL: https://open-meteo.com/en/docs/historical-weather-api
- Licence: CC BY 4.0, attribution "Weather data by Open-Meteo.com"
- Downloaded: 2026-10-18
- Saved as: `rain/<site>_<start>_<end>.csv.gz` (one file per overflow, used as a cache) and `rain_hourly.csv.gz` (all selected overflows)

### Notes
- Fetched at each selected overflow's own coordinates, hourly, in UTC.
- The archive lags real time by a few days, so rain stops about 3 days before the fetch date.
- Long date ranges count as many calls in the free quota (roughly one per 14 days of data), so the script waits 8 seconds between overflows.

## Files built from the raw data

- `spills.csv.gz`: every spill event (`site_id`, `start`, `end`, `source`) from 2024-01-01, EA files before 2026 and the API after.
- `sites_selected.csv.gz`: the overflows used for modelling, with name, river, grid reference, lat/lon and number of spills.
