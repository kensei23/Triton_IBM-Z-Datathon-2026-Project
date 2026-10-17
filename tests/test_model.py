import sys
import time
from pathlib import Path

# tests/ isn't on the path by default so add the repo root to import configs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import settings
from llama_cpp import Llama

MODEL_PATH = Path(settings["model_dir"]) / settings["active_model"]

# Grab the pre-made promt from prompt.txt which is used to simulate
PROMPT_FILE = Path(__file__).resolve().parent.parent / "configs/llm/prompt.txt"
PROMPT = PROMPT_FILE.read_text().rstrip("\n")


def main():
    assert MODEL_PATH.exists(), f"model file not found: {MODEL_PATH}"

    t = time.time()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=settings["n_ctx"],
        n_threads=settings["n_threads"],
        verbose=False,
    )
    load_s = time.time() - t

    t = time.time()
    # stop at the first newline
    out = llm(PROMPT, max_tokens=settings["max_tokens"], temperature=0.2, stop=["\n"])
    gen_s = time.time() - t

    text = out["choices"][0]["text"].rstrip()
    n_tokens = out["usage"]["completion_tokens"]
    tok_s = n_tokens / gen_s

    print("model:", MODEL_PATH.name)
    print(f"load: {load_s:.1f}s | generate: {gen_s:.2f}s")
    print(f"{n_tokens} tokens | {tok_s:.1f} tok/s")
    print("output: The" + text)

    assert text, "model returned empty text"
    assert gen_s < 15, f"generation too slow: {gen_s:.1f}s"
    print("PASS")


if __name__ == "__main__":
    main()
