import time

from triton.agent.decide import decide
from triton.agent.explain import explain
from triton.events import load_event
from triton.llm import load_llm

start = time.time()
event = load_event("src/triton/agent/mock_event.json")
decision = decide(event)
# decision is instant so show it before waiting on granite
print("decision:", decision)

llm = load_llm()
result = explain(event, decision, llm)
print("explanation:", result["text"])
print("source:", result["source"], "| granite time:", result["seconds"], "s")
print("total:", round(time.time() - start, 1), "s")
