import json
from pathlib import Path

from triton.agent.decide import decide
from triton.agent.explain import build_facts, build_prompt, explain, numbers_ok

ROOT = Path(__file__).resolve().parent.parent
EVENT = json.loads((ROOT / "src/triton/agent/mock_event.json").read_text())
DECISION = decide(EVENT)
FACTS = build_facts(EVENT, DECISION)


def fake_llm(text):
    # stands in for granite so these tests dont need the model
    def run(prompt, **kwargs):
        return {"choices": [{"text": text}]}

    return run


def test_facts_from_mock_event():
    assert FACTS["pct"] == 91
    assert FACTS["eta"] == 95
    assert FACTS["spill_min"] == 105


def test_prompt_ends_with_seed():
    prompt, seed = build_prompt(FACTS)
    assert prompt.endswith("should reach the intake in about")
    assert "ETA to intake 95 min" in prompt


def test_checker_rejects_invented_number():
    assert numbers_ok("The pollution is 95 min away.", FACTS)
    assert not numbers_ok("The event ends in 3 hours, 95 min away.", FACTS)
    assert not numbers_ok("The pollution is close.", FACTS)


def test_good_output_is_used():
    text = " 95 minutes."
    result = explain(EVENT, DECISION, fake_llm(text))
    assert result["source"] == "granite"
    assert result["text"].startswith("The sensors show")


def test_bad_output_falls_back():
    result = explain(EVENT, DECISION, fake_llm(" 3 hours."))
    assert result["source"] == "template"
    assert "95" in result["text"]
