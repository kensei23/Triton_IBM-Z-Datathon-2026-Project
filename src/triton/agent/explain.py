import re
import time
from datetime import datetime
from pathlib import Path

from configs import settings

SHOT_FILE = Path(__file__).resolve().parents[3] / "configs/llm/shot.txt"


def _minutes_since_spill(event):
    spills = event.get("upstream_spills") or []
    if not spills:
        return None
    latest = max(spills, key=lambda s: s["start"])
    now = datetime.fromisoformat(event["timestamp"])
    start = datetime.fromisoformat(latest["start"])
    return round((now - start).total_seconds() / 60)


def build_facts(event, decision):
    """Pull out the only facts granite is allowed to talk about."""
    names = [f["name"].replace("_", " ") for f in event["top_features"]]
    return {
        "features": " and ".join(names) or "sensor readings",
        "spill_min": _minutes_since_spill(event),
        "pct": decision["probability_pct"],
        "eta": decision["eta_minutes"],
    }


def build_prompt(facts):
    if facts["spill_min"] is None:
        spill = "no spill recorded upstream"
    else:
        spill = f"spill upstream {facts['spill_min']} min ago"
    line = (
        f"Event: unusual {facts['features']}, {spill}, "
        f"probability {facts['pct']}%, ETA to intake {facts['eta']} min."
    )
    shot = SHOT_FILE.read_text().rstrip("\n")
    # ends on "The" so the first token isnt a newline
    return f"{shot}\n\n{line}\nExplanation: The"


def numbers_ok(text, facts):
    """Every number must come from the facts and the ETA has to be there."""
    allowed = {str(facts["pct"]), str(facts["eta"])}
    if facts["spill_min"] is not None:
        allowed.add(str(facts["spill_min"]))
    found = re.findall(r"\d+(?:\.\d+)?", text)
    return all(n in allowed for n in found) and str(facts["eta"]) in found


def fallback_text(facts, decision):
    text = (
        f"Pollution at {decision['station_id']} is {facts['pct']}% likely "
        f"and should reach intake {decision['intake_id']} in about "
        f"{facts['eta']} minutes."
    )
    if facts["spill_min"] is not None:
        text += f" A spill was recorded upstream {facts['spill_min']} minutes earlier."
    return text


def explain(event, decision, llm):
    facts = build_facts(event, decision)
    start = time.time()
    out = llm(
        build_prompt(facts),
        max_tokens=settings["max_tokens"],
        temperature=0,
        stop=["\n"],
    )
    seconds = round(time.time() - start, 1)

    text = "The" + out["choices"][0]["text"].rstrip()
    # keep just the first sentence
    if "." in text:
        text = text.split(".")[0] + "."

    if numbers_ok(text, facts):
        return {"text": text, "source": "granite", "seconds": seconds}
    return {
        "text": fallback_text(facts, decision),
        "source": "template",
        "seconds": seconds,
    }
