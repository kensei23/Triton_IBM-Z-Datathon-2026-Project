from pathlib import Path

import pandas as pd

# Anchored to the project folder so commands work from any directory.
DATA = Path(__file__).resolve().parents[2] / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

EA_HYDROLOGY = "https://environment.data.gov.uk/hydrology"
EA_EDM = "https://environment.data.gov.uk/api/file/download"
EA_EDM_DATASET = "c95ecc47-fa70-4a24-b8cc-098232987526"
THAMES_WATER = "https://api.thameswater.co.uk/opendata/v2/discharge"

START = "2024-01-01"
# EA yearly files hold spill history; the Thames Water API holds the current year.
EA_EDM_FILES = [
    "Thames 2024 Detailed EDM Data.csv",
    "Thames Water 2025 EDM Detailed Data.xlsx",
]
API_FROM = pd.Timestamp("2026-01-01", tz="UTC")
# Outfalls "via the Tideway tunnel" spilled into the river until full connection.
TIDEWAY_DATE = pd.Timestamp("2025-02-14", tz="UTC")
REMOVED_CORRECTION = r"remov|delet|invalid|not a (?:discharge|spill)|false"

# EA observed property -> wide column name with unit.
SENSORS = {
    "dissolved-oxygen": "dissolved_oxygen_mg_l",
    "ammonium": "ammonium_mg_l",
    "conductivity": "conductivity_us_cm",
    "turbidity": "turbidity_ntu",
    "temperature": "temperature_c",
    "ph": "ph",
}
USABLE_QUALITY = ["Good", "Unchecked", "Estimated"]
# Deliberately wide: only physically impossible river values count as faults.
VALID_RANGE = {
    "dissolved-oxygen": (0, 20),
    "ammonium": (0, 50),
    "conductivity": (1, 10000),  # tidal Thames turns brackish in dry spells
    "turbidity": (0, 1000),
    "temperature": (-1, 35),
    "ph": (4, 10),
}
STUCK_HOURS = 12
MAX_GAP_HOURS = 3
# Sensor faults found in the EDA that range checks only partly catch.
FAULT_PERIODS = {
    ("BREPON", "ph"): [("2025-01-13", "2025-04-07")],
    ("HAMME2", "conductivity"): [("2024-05-30", "2024-06-06")],
}
OUTLIER_Z = 5

BASELINE_WINDOW = "7D"
BASELINE_MIN_HOURS = 24
Z_THRESHOLD = 2.5
SPILL_AFTER = "6h"
# Early warning horizon: target_6h looks this many hours ahead.
TARGET_HOURS = 6

LEE = r"\b(?:Lee|Lea|Pymmes|Salmon|Ching)\b"

# direction: where upstream outfalls lie. The Lee flows south and the tidal Thames
# flows both ways. Brent and Roding stations were tested and dropped: their sensors
# showed no clear response to upstream spills (anomaly lift 0.9 to 1.7).
STATIONS = {
    "E01336A": {
        "name": "Lee Springfield Park",
        "outfall_river": LEE,
        "direction": "north",
        "km": 15,
        "gauge_river": r"\b(?:Lee|Lea)\b",
    },
    "E26020A": {
        "name": "Lee Lea Bridge Weir",
        "outfall_river": LEE,
        "direction": "north",
        "km": 15,
        "gauge_river": r"\b(?:Lee|Lea)\b",
    },
    "HAMME2": {
        "name": "Thames Hammersmith",
        "outfall_river": r"\bThames\b",
        "direction": "both",
        "km": 8,
        "gauge_river": r"\bThames\b",
    },
    "BREPON": {
        "name": "Thames Brentford",
        "outfall_river": r"\b(?:Thames|Brent)\b",
        "direction": "both",
        "km": 8,
        "gauge_river": r"\bThames\b",
    },
}
# Search area for stations and gauges: the Thames Water region.
SEARCH_AREA = {"lat": 51.6, "long": -0.8, "dist": 70}
