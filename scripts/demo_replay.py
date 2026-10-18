"""Live replay of a real storm from the test period: detect, decide, respond.

Usage: uv run python scripts/demo_replay.py [--delay 0.6] [--no-llm]
       uv run python scripts/demo_replay.py --start "2026-01-26 00:00" --hours 36

Steps through the storm hour by hour as if it were live. Each hour the model
scores every overflow, decides INTERVENE / NOTIFY / MONITOR, and Granite (on
the local CPU) writes an alert for each site that escalates. Real spill starts
are shown as they happen, and a scorecard is printed at the end.
Writes outputs/demo_replay.json and outputs/demo_timeline.html.
"""

import argparse
import json
import platform
import re
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd

from triton import alerts, replay, timeline
from triton import model as m
from triton.constants import OUTPUTS_DIR
from triton.decision import load_thresholds
from triton.tables import load_episodes, load_panel, load_sites

IBM_Z = {
    "9175": "IBM z17",
    "9176": "IBM z17",
    "3931": "IBM z16",
    "3932": "IBM z16",
    "8561": "IBM z15",
    "8562": "IBM z15",
}
COLOURS = {
    "INTERVENE": "\033[1;31m",
    "NOTIFY": "\033[1;33m",
    "SPILL": "\033[1;36m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "end": "\033[0m",
}


class Printer:
    def __init__(self, colour):
        self.colour = colour

    def c(self, text, style):
        if not self.colour:
            return text
        return f"{COLOURS[style]}{text}{COLOURS['end']}"


def wrap(text, indent=9):
    return textwrap.fill(
        text, width=100, initial_indent=" " * indent, subsequent_indent=" " * indent
    )


def describe_machine():
    cpu = platform.processor() or platform.machine()
    try:
        info = Path("/proc/cpuinfo").read_text()
    except OSError:
        info = ""
    # IBM Z lists its machine type, e.g. "machine = 9175" on a z17
    found = re.search(r"machine\s*=\s*(\w+)", info)
    if found:
        kind = IBM_Z.get(found.group(1), "IBM Z")
        cpu = f"{kind} (machine type {found.group(1)})"
    elif "model name" in info:
        cpu = re.search(r"model name\s*:\s*(.+)", info).group(1).strip()
    python = platform.python_version()
    return f"{platform.machine()} | {platform.system()} | {cpu} | Python {python}"


def main(args):
    out = Printer(args.colour)
    bundle = m.load_bundle()
    thresholds = load_thresholds()
    panel, episodes, sites = load_panel(), load_episodes(), load_sites()
    test_from = pd.Timestamp(bundle["test_from"], tz="UTC")

    if args.start:
        t0 = pd.Timestamp(args.start, tz="UTC")
        t1 = t0 + pd.Timedelta(hours=args.hours)
    else:
        t0, t1 = replay.pick_storm(episodes, test_from, hours=args.hours)
    if t0 < test_from:
        print(f"note: {t0} is before the test period, the model trained on it")

    granite, info = (None, "off (--no-llm)") if args.no_llm else alerts.load_granite()
    print(out.c("TRITON  storm overflow early warning  |  live replay", "bold"))
    print(f"machine : {describe_machine()}")
    n_sites = len(bundle["site_codes"])
    print(f"model   : calibrated gradient boosting on {n_sites} overflows")
    print(
        f"decide  : NOTIFY at >= {thresholds['notify']:.1%}, "
        f"INTERVENE at >= {thresholds['intervene']:.1%} spill chance in 6 h"
    )
    print(f"granite : {info if granite else 'template alerts only, ' + info}")
    print(
        f"storm   : {t0:%a %d %b %Y %H:%M} to {t1:%a %d %b %H:%M} UTC "
        "(test period, never seen in training)"
    )
    print(out.c("every number below uses only data available at that hour\n", "dim"))

    hours, alert_log = [], []
    llm_times, sources = [], {"granite": 0, "template": 0}
    for hour in replay.replay(panel, episodes, bundle, thresholds, t0, t1):
        start = time.perf_counter()
        hours.append(hour)
        rows, levels = hour["rows"], hour["levels"]
        counts = pd.Series(levels).value_counts()
        rain = rows["rain_1h"]
        intervene = out.c(f"INTERVENE {counts.get('INTERVENE', 0):2d}", "INTERVENE")
        notify = out.c(f"NOTIFY {counts.get('NOTIFY', 0):2d}", "NOTIFY")
        print(
            f"{hour['timestamp']:%a %d %b %H:%M} | rain avg {rain.mean():4.1f} "
            f"max {rain.max():4.1f} mm/h | {intervene}  {notify}  "
            f"spilling {counts.get('SPILLING', 0):2d}"
        )
        for esc in hour["escalations"]:
            site_id = esc["site_id"]
            row = rows.loc[site_id].copy()
            row["site_id"], row["timestamp"] = site_id, hour["timestamp"]
            site = {"site_id": site_id, **sites.loc[site_id].to_dict()}
            summary = alerts.build_summary(
                site,
                hour["timestamp"],
                esc["p"],
                esc["level"],
                alerts.drivers(bundle, row),
                alerts.similar_event(panel, episodes, site_id, hour["timestamp"]),
            )
            use_llm = granite if len(llm_times) < args.max_llm else None
            result = alerts.generate_alert(summary, use_llm)
            sources[result["source"]] = sources.get(result["source"], 0) + 1
            if result["source"] != "template" or use_llm:
                llm_times.append(result["seconds"])
            tag = f"[{result['source']} {result['seconds']}s]"
            print(
                f"   {out.c('>> ' + esc['level'].ljust(9), esc['level'])} "
                f"{summary['site_name']} ({summary['river']})  "
                f"{summary['probability_pct']}%  {out.c(tag, 'dim')}"
            )
            print(wrap(result["text"]))
            alert_log.append(
                {
                    **summary,
                    "timestamp": hour["timestamp"],
                    "p": esc["p"],
                    "text": result["text"],
                    "source": result["source"],
                }
            )
        for spill in hour["spill_starts"]:
            name = sites.loc[spill["site_id"], "name"]
            lead = replay.warning_lead(hours, spill["site_id"], spill["start"])
            note = (
                f"warned {lead:.1f} h ahead" if lead else "no warning in the 6 h before"
            )
            print(
                f"   {out.c('!! SPILL    ', 'SPILL')} {name} started "
                f"{spill['start']:%H:%M} - {note}"
            )
        if args.delay:
            time.sleep(max(0.0, args.delay - (time.perf_counter() - start)))

    result = replay.score(hours, episodes, t0, t1)
    print(out.c("\nscorecard", "bold"))
    print(f"  spills that started (after the first 6 h): {result['spills']}")
    print(
        f"  warned beforehand (NOTIFY or higher in the 6 h before): {result['warned']}"
        f"  median warning {result['median_lead_h']} h"
    )
    print(
        f"  alerts raised: {result['alerts']}  "
        f"({result['false_alarms']} not followed by a spill within 6 h)"
    )
    if llm_times:
        print(
            f"  alert text: {sources.get('granite', 0)} by Granite, "
            f"{sources.get('template', 0)} by template, "
            f"Granite avg {sum(llm_times) / len(llm_times):.1f}s on this CPU"
        )

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "demo_replay.json").write_text(
        json.dumps(
            {
                "start": str(t0),
                "end": str(t1),
                "machine": describe_machine(),
                "granite": info,
                "score": result,
                "alerts": alert_log,
            },
            indent=1,
            default=str,
        )
        + "\n"
    )
    write_timeline(hours, alert_log, episodes, sites, thresholds, t0, t1)
    print("\nsaved outputs/demo_replay.json and outputs/demo_timeline.html")


def write_timeline(hours, alert_log, episodes, sites, thresholds, t0, t1):
    rows = pd.concat([h["rows"].assign(timestamp=h["timestamp"]) for h in hours])
    probs = pd.concat(
        [
            h["probs"].rename("p").to_frame().assign(timestamp=h["timestamp"])
            for h in hours
        ]
    )
    ep = episodes[(episodes["end"] >= t0) & (episodes["start"] <= t1)]
    busy = pd.Series(
        [a["site_id"] for a in alert_log] + ep["site_id"].tolist()
    ).value_counts()
    site_data = []
    for site_id in busy.index[:12]:
        r = rows.loc[site_id].set_index("timestamp")
        p = (
            probs.loc[[site_id]].set_index("timestamp")["p"]
            if site_id in probs.index
            else pd.Series(dtype=float)
        )
        site_alerts = [
            {"timestamp": a["timestamp"], "p": a["p"], "level": a["decision"]}
            for a in alert_log
            if a["site_id"] == site_id
        ]
        spills = [
            (s, e)
            for s, e in ep[ep["site_id"] == site_id][["start", "end"]].itertuples(
                index=False
            )
        ]
        site_data.append(
            {
                "name": f"{sites.loc[site_id, 'name']} ({sites.loc[site_id, 'river']})",
                "rows": r,
                "probs": p,
                "alerts": site_alerts,
                "spills": spills,
            }
        )
    fig = timeline.build(
        site_data,
        thresholds,
        t0,
        t1,
        f"Replayed storm {t0:%d %b %Y}: rain, predicted spill chance and real spills",
    )
    fig.write_html(OUTPUTS_DIR / "demo_timeline.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--start", help="UTC start hour; default picks the busiest storm"
    )
    parser.add_argument("--hours", type=int, default=36, help="hours to replay")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds per hour, e.g. 0.6 for recording",
    )
    parser.add_argument("--no-llm", action="store_true", help="template alerts only")
    parser.add_argument(
        "--max-llm",
        type=int,
        default=40,
        help="alerts written by Granite, rest by template",
    )
    parser.add_argument(
        "--no-colour", dest="colour", action="store_false", help="plain text output"
    )
    args = parser.parse_args()
    if args.colour and not sys.stdout.isatty():
        args.colour = False
    main(args)
