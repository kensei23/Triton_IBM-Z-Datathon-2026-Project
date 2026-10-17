# Data

No data is committed. Rebuild everything with (from any folder inside the project):

```bash
uv run python -m triton.download   # data/raw: sources, unchanged
uv run python -m triton.parse      # data/interim: parsed tables
uv run python -m triton.build      # data/processed: cleaned data and labels
```

`download` skips files that already exist; add `--force` to download again.
Every raw file's URL, parameters and download time are recorded in
`data/raw/manifest.json`.

## Layers

| Folder | Contents | Rule |
|---|---|---|
| `raw/` | Files exactly as the sources returned them | Never edited |
| `interim/` | Raw files parsed into tables | No cleaning, nothing dropped except unreadable records |
| `processed/` | Cleaned readings, labels and the modelling table | What the model and app read |

Read processed CSVs with `triton.io.load(name)`, which restores dates as UTC timestamps.

## Sources

### EA Hydrology API

- Source: Environment Agency
- URL: https://environment.data.gov.uk/hydrology
- Licence: Open Government Licence v3.0
- Contents: water quality sensors (dissolved oxygen, ammonium, conductivity,
  turbidity, temperature, pH), rainfall and river flow
- Place in: `data/raw/ea_hydrology/`
  - `stations.json`: stations within 70 km of the Thames Valley
  - `measures/{station}_{property}.json`: the sensor series available per station
  - `measures/rainfall.json`, `measures/waterFlow.json`: all rain and flow gauges
  - `readings/{measure}.csv`: readings from 1 January 2024, one file per series

### EA storm overflow start/stop detailed data (EDM)

- Source: Environment Agency
- URL: https://www.data.gov.uk/dataset/39ba1e24-69ca-4031-badf-e02649b74a7b/event-duration-monitoring-storm-overflow-start-stop-detailed-data
- Licence: Open Government Licence v3.0
- Contents: every storm overflow spill with start and stop times, per water company
  and year. The EA states the data has not been formally verified.
- Place in: `data/raw/ea_edm/`
  - `Thames 2024 Detailed EDM Data.csv` (an Excel workbook despite the extension)
  - `Thames Water 2025 EDM Detailed Data.xlsx`

### Thames Water open data API

- Source: Thames Water Utilities Limited
- URL: https://api.thameswater.co.uk/opendata/v2/discharge
- Licence: https://data.thameswater.co.uk/s/terms-of-service
- Contents: outfall list with locations (`status`) and live spill alerts (`alerts`).
  The alerts endpoint keeps roughly one year of history, so it is used only from
  1 January 2026; earlier spills come from the EA files.
- Place in: `data/raw/thames_water/{status,alerts}/page_*.json`

## Processed files

### `dataset.csv`: one row per station and hour

| Column | Unit | Description |
|---|---|---|
| `station_id` | | EA station ID |
| `timestamp` | UTC | Start of the hour |
| `dissolved_oxygen_mg_l` | mg/L | Oxygen dissolved in the water |
| `ammonium_mg_l` | mg/L | Ammonium, as reported by the EA |
| `conductivity_us_cm` | µS/cm | Electrical conductivity |
| `turbidity_ntu` | NTU | Cloudiness |
| `temperature_c` | °C | Water temperature |
| `ph` | | Acidity |
| `rain_mm` | mm | Rain in the hour at the nearest gauge |
| `flow_m3s` | m³/s | Mean river flow in the hour at the matched gauge |
| `upstream_spill_active` | 0/1 | An upstream outfall is spilling during this hour |
| `hours_since_spill` | h | Hours since the latest upstream spill started; empty before the first |
| `label` | | 1 pollution event this hour, 0 none, empty when no baseline exists yet |
| `target_6h` | | 1 if any of the next 6 hours has `label = 1`; the model target |

Each station has a regular hourly grid; missing hours are empty cells, not missing rows.

### Modelling rules

- Target: `target_6h`. An early warning must predict ahead, so the target looks at
  hours t+1 to t+6 only.
- Never use as features: `label`, `target_6h`, and everything in `labels.csv`
  (`spill_recent`, `z_do`, `z_nh4`). They define the target, so using them leaks it.
- Allowed: sensor values at hour t and earlier, rain, flow and the spill columns,
  which the Thames Water API provides live.
- Split by time (for example train 2024, validate 2025, test 2026), never randomly:
  neighbouring hours are near copies of each other.
- Positives are 1 to 3% of hours: use class weights and judge models on PR-AUC
  and recall at a fixed alert rate, not accuracy.

## Cleaning

All readings are flagged "Unchecked" by the EA, so these checks are the only cleaning.
Settings live in `src/triton/config.py`; every removal is counted in
`quality_report.csv`.

| Step | Rule | Why |
|---|---|---|
| Fault periods | Readings inside `FAULT_PERIODS` removed | Faults found in the EDA that range checks only partly catch |
| Impossible values | Outside `VALID_RANGE` removed | Error codes such as 99999 and broken sensors |
| Frozen sensors | Identical readings for 12 hours or more removed | A stuck logger repeats its last value |
| Hourly mean | Raw readings averaged per hour | Stations log hourly (Lee) or every 15 minutes (Thames) |
| Short gaps | Gaps of 3 hours or less interpolated | Longer gaps stay empty |
| Spikes | Robust z-score above 5 counted, kept | A sudden spike is the pollution signal |

Known sensor problems:

- Brentford turbidity: negative readings from the start and no data after
  13 December 2024. Removed in full; do not use turbidity at Brentford.
- Brentford pH, 13 January to 7 April 2025: drift up to 18 and down to -5.
- Hammersmith conductivity, 30 May to 6 June 2024: jumps to 23,000 µS/cm.
- Hammersmith, 23 July 2025: all sensors frozen for 45 hours.
- Hammersmith conductivity above 5,000 µS/cm in autumn 2026 is real: tidal salt
  water in a dry spell. The upper limit is 10,000 µS/cm for this reason.

### Other files

- `labels.csv`: the label and the columns that define it (`spill_recent`, `z_do`,
  `z_nh4`). These must not be used as model features.
- `spills.csv`: all spill events with outfall, river, location, start, end and source.
- `quality_report.csv`: per station and sensor, missing data, readings removed by
  each rule, filled hours, longest gap and spikes.
- `stations.json`: station names, coordinates, matched gauges and upstream outfalls.
