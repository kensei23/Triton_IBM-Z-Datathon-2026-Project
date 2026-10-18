"""Load the files the pipeline writes, with times parsed as UTC."""

import pandas as pd

from triton.constants import PROCESSED_DIR, RAW_DIR


def load_panel():
    panel = pd.read_csv(PROCESSED_DIR / "model_table.csv.gz")
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    return panel


def load_episodes():
    episodes = pd.read_csv(PROCESSED_DIR / "episodes.csv.gz")
    for col in ("start", "end"):
        episodes[col] = pd.to_datetime(episodes[col], utc=True, format="ISO8601")
    return episodes


def load_sites():
    sites = pd.read_csv(RAW_DIR / "sites_selected.csv.gz").set_index("site_id")
    sites["name"] = sites["name"].str.strip()
    sites["river"] = sites["river"].str.strip()
    return sites
