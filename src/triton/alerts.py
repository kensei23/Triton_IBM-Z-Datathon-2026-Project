"""Explain a spill prediction and write the operator alert.

The summary is built from facts only: site, probability, the features that
pushed the prediction up, and the most similar past spill at that site.
IBM Granite, running locally on the CPU through llama.cpp, writes the explanation;
any number it uses must come from the facts, otherwise a template is used, so
an alert is always produced.
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from triton.constants import CONFIGS_DIR, MODELS_DIR, ROOT_DIR
from triton.decision import ACTIONS
from triton.features import target_name

SHOT_FILE = CONFIGS_DIR / "llm" / "spill_shot.txt"
ALERT_CONFIG = CONFIGS_DIR / "llm" / "alerts.json"
SIMILAR_COLS = ["rain_24h", "rain_7d", "rain_30d"]
MIN_EFFECT = 0.003  # ignore drivers that move the probability by under 0.3 points


def _mm(v):
    return f"{v:.0f}" if v >= 1 else f"{v:.1f}"


def _hours(n):
    return f"{n} hour" if n == 1 else f"{n} hours"


# plain-English phrase for each feature's current value
PHRASES = {
    "rain_1h": lambda v: f"{_mm(v)} mm of rain in the last hour",
    "rain_3h": lambda v: f"{_mm(v)} mm of rain in the last 3 hours",
    "rain_6h": lambda v: f"{_mm(v)} mm of rain in the last 6 hours",
    "rain_24h": lambda v: f"{_mm(v)} mm of rain in the last 24 hours",
    "rain_7d": lambda v: f"{_mm(v)} mm of rain in the last 7 days",
    "rain_30d": lambda v: f"{_mm(v)} mm of rain in the last 30 days",
    "rain_max_1h_24h": lambda v: f"bursts of up to {_mm(v)} mm an hour today",
    "hours_since_spill": lambda v: (
        "no spill in the last 90 days"
        if v >= 24 * 90
        else f"the last spill ended {v:.0f} hours ago"
    ),
    "spills_7d": lambda v: f"{v:.0f} spills in the last 7 days",
    "hour": lambda v: (
        "the morning peak in sewer flow"
        if 6 <= v <= 10
        else "the evening peak in sewer flow"
        if 17 <= v <= 21
        else f"the time of day ({v:.0f}:00 UTC)"
    ),
    "month": lambda v: "the time of year",
    "site_code": lambda v: "this overflow's spill history",
}


def drivers(bundle, row, top=3):
    """Features that pushed this prediction up the most.

    Each feature in turn is reset to its typical (training median) value; the
    drop in probability is how much that feature is contributing right now.
    """
    features = bundle["features"]
    x = row[features].to_frame().T.astype(float)
    trials = pd.concat([x] * (len(features) + 1), ignore_index=True)
    for i, f in enumerate(features, start=1):
        trials.loc[i, f] = bundle["reference"][f]
    p = bundle["model"].predict_proba(trials)[:, 1]
    effects = p[0] - p[1:]
    order = np.argsort(-effects)
    out = []
    for i in order[:top]:
        if effects[i] < MIN_EFFECT:
            break
        f = features[i]
        out.append(
            {
                "feature": f,
                "value": float(row[f]),
                "effect": round(float(effects[i]), 4),
                "text": PHRASES[f](float(row[f])),
            }
        )
    return out


def similar_event(panel, episodes, site_id, t, horizon=6):
    """The past hour at this site, before a spill, with the closest rain.

    Only looks at hours more than 7 days before t, so it is history, not the
    storm happening now.
    """
    target = target_name(horizon)
    now = panel[(panel["site_id"] == site_id) & (panel["timestamp"] == t)]
    past = panel[
        (panel["site_id"] == site_id)
        & (panel[target] == 1)
        & (panel["timestamp"] < t - pd.Timedelta(days=7))
    ]
    if now.empty or past.empty:
        return None
    scale = panel[SIMILAR_COLS].std().replace(0, 1)
    dist = (((past[SIMILAR_COLS] - now[SIMILAR_COLS].iloc[0]) / scale) ** 2).sum(axis=1)
    best = past.loc[dist.idxmin()]
    ep = episodes[
        (episodes["site_id"] == site_id) & (episodes["start"] > best["timestamp"])
    ].sort_values("start")
    if ep.empty:
        return None
    spill = ep.iloc[0]
    return {
        "date": f"{spill['start'].day} {spill['start']:%b %Y}",
        "duration_h": max(
            1, round((spill["end"] - spill["start"]).total_seconds() / 3600)
        ),
        "rain_7d": float(best["rain_7d"]),
    }


def build_summary(site, timestamp, p, decision, driver_list, similar, horizon=6):
    """Everything the alert may say, and nothing else."""
    return {
        "site_id": site["site_id"],
        "site_name": site["name"].strip(),
        "river": site["river"].strip(),
        "time": pd.Timestamp(timestamp).strftime("%H:%M %d %b %Y UTC"),
        "probability_pct": max(1, round(100 * float(p))),
        "decision": decision,
        "action": ACTIONS[decision],
        "horizon_h": horizon,
        "drivers": [d["text"] for d in driver_list],
        "similar": similar,
    }


def _place(s):
    river = s["river"]
    if not river.lower().startswith(("river", "the ")):
        return f"{s['site_name']} on {river}"
    return f"{s['site_name']} on the {river}"


def facts_line(s):
    line = (
        f"Facts: {_place(s)}. Spill chance {s['probability_pct']}% in the next "
        f"{s['horizon_h']} hours."
    )
    if s["drivers"]:
        line += " Reasons: " + "; ".join(s["drivers"]) + "."
    if s["similar"]:
        sim = s["similar"]
        spilled = _hours(sim["duration_h"])
        line += f" Similar past case: {sim['date']}, spilled for {spilled}."
    return line


RISK_WORDS = {"INTERVENE": "a high", "NOTIFY": "a raised", "MONITOR": "a low"}


def headline(s):
    return (
        f"{_place(s)} has {RISK_WORDS[s['decision']]} risk of spilling within "
        f"{s['horizon_h']} hours ({s['probability_pct']}% chance)."
    )


def build_prompt(s):
    """Few-shot prompt for a base model; it writes what follows the headline."""
    shot = SHOT_FILE.read_text().rstrip("\n")
    return f"{shot}\n\n{facts_line(s)}\nAlert: {headline(s)}"


def chat_messages(s, system):
    """Instruction prompt for an instruct (chat) model."""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"{facts_line(s)}\nWrite one sentence explaining why a "
            "spill is likely, using the reasons and the similar past case.",
        },
    ]


NUMBER = r"\d+(?:\.\d+)?"
# "18 mm of rain in the last 24 hours" -> ("18", "24 hours")
RAIN_WINDOW = re.compile(rf"({NUMBER}) mm (?:of rain )?in (?:the last )?(hour|\d+ \w+)")
TIME_WORDS = re.compile(r"\b(week|month|year|today|tonight)\b", re.IGNORECASE)
# the facts never compare anything, so a comparison is made up
COMPARISON = re.compile(r"\b(higher|lower|more|less|greater|fewer) than\b", re.I)


def facts_ok(text, s):
    """The generated text may only restate the facts.

    Every number must appear in the facts, every "N mm in the last X" must pair
    N with the same window as the facts do, and no time spans the facts never
    mention (week, month, ...) may appear. The probability is already in the
    headline, so a "%" means the model is echoing it, which is where small
    models start to ramble; comparisons are rejected for the same reason.
    """
    if "%" in text or COMPARISON.search(text):
        return False
    facts = facts_line(s)
    allowed = set(re.findall(NUMBER, facts))
    if not all(n in allowed for n in re.findall(NUMBER, text)):
        return False
    if not set(RAIN_WINDOW.findall(text)) <= set(RAIN_WINDOW.findall(facts)):
        return False
    return all(w.lower() in facts.lower() for w in TIME_WORDS.findall(text))


def template_reason(s):
    parts = []
    if s["drivers"]:
        parts.append("Main reasons: " + "; ".join(s["drivers"]) + ".")
    if s["similar"]:
        sim = s["similar"]
        parts.append(
            f"Conditions are like {sim['date']}, when it spilled for "
            f"{_hours(sim['duration_h'])}."
        )
    return " ".join(parts)


def _finish(s, reason, source, seconds):
    text = f"{headline(s)} {reason}".strip()
    if s["decision"] == "INTERVENE":
        # the action taken is stated by code, never by the language model
        text += " Automatic action: storm tank emptying to treatment has started."
    return {"text": text, "source": source, "seconds": seconds}


def generate_alert(s, granite=None):
    """Alert text for a summary; Granite if it gives a clean answer, else template."""
    if s["decision"] == "MONITOR":
        # log only: no explanation, no model call
        return _finish(s, "", "log", 0.0)
    if granite is None:
        return _finish(s, template_reason(s), "template", 0.0)
    start = time.time()
    try:
        tail = granite.write(s).strip()
    except Exception as err:  # the demo must keep going whatever the model does
        print(f"  granite failed ({err}), using template")
        tail = ""
    seconds = round(time.time() - start, 1)
    # keep whole sentences only
    if "." in tail:
        tail = tail[: tail.rfind(".") + 1]
    if len(tail) > 20 and tail.endswith(".") and facts_ok(tail, s):
        return _finish(s, tail, "granite", seconds)
    return _finish(s, template_reason(s), "template", seconds)


class Granite:
    """A local llama.cpp Granite model, in chat (instruct) or few-shot (base) mode."""

    def __init__(self, llama, chat, name, settings):
        self.llama, self.chat, self.name, self.settings = llama, chat, name, settings

    def write(self, s):
        cfg = self.settings
        if self.chat:
            out = self.llama.create_chat_completion(
                messages=chat_messages(s, cfg["system"]),
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
            )
            return out["choices"][0]["message"]["content"]
        out = self.llama(
            build_prompt(s),
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            stop=["\n"],
        )
        return out["choices"][0]["text"]


def alert_settings():
    return json.loads(ALERT_CONFIG.read_text())


def model_dirs():
    """Where GGUF files may live: the team's model dir, then models/llm, models/."""
    # configs/ is a top-level folder, importable once the repo root is on the path
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from configs import settings

    dirs = [Path(settings["model_dir"]), MODELS_DIR / "llm", MODELS_DIR]
    return list(dict.fromkeys(dirs)), settings


def load_granite():
    """Granite instruct if present, else the team's base model, else None.

    Returns (Granite or None, a line saying what was loaded or why not).
    The instruct model needs a big-endian GGUF on IBM Z, see get_granite.py.
    """
    cfg = alert_settings()
    try:
        from llama_cpp import Llama

        dirs, llm_settings = model_dirs()
    except Exception as err:  # llama_cpp not built, configs missing, ...
        return None, f"{type(err).__name__}: {err}"

    chat_file = cfg["chat_model_big_endian" if sys.byteorder == "big" else "chat_model"]
    candidates = [(chat_file, True), (llm_settings["active_model"], False)]
    for filename, chat in candidates:
        for folder in dirs:
            path = folder / filename
            if not path.exists():
                continue
            try:
                llama = Llama(
                    model_path=str(path),
                    n_ctx=llm_settings["n_ctx"],
                    n_threads=llm_settings["n_threads"],
                    n_threads_batch=llm_settings["n_threads"],
                    verbose=False,
                )
            except Exception as err:
                return None, f"could not load {path}: {err}"
            mode = "instruct, chat" if chat else "base, few-shot"
            return Granite(llama, chat, filename, cfg), f"{filename} ({mode})"
    looked = ", ".join(str(d) for d in dirs)
    return None, f"no Granite GGUF found in {looked}; run scripts/get_granite.py"
