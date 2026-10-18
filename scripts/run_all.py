"""Run the whole pipeline: fetch data, build features, train, replay a storm.

Usage: uv run python scripts/run_all.py [--top-n 40] [--skip-fetch] [--no-llm]

Downloads are cached, so a rerun after a successful fetch takes a couple of
minutes. Granite (366 MB) is downloaded once into models/llm unless --no-llm.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def run(name, *args):
    print(f"\n=== {name} ===", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable, str(SCRIPTS / name), *args])
    if result.returncode != 0:
        sys.exit(f"{name} failed (exit {result.returncode}), stopping")
    print(f"--- {name} done in {time.time() - start:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", default="40", help="overflows to model")
    parser.add_argument(
        "--skip-fetch", action="store_true", help="use the data already in data/raw"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="skip Granite, template alerts only"
    )
    args = parser.parse_args()

    if not args.skip_fetch:
        run("fetch_data.py", "--top-n", args.top_n)
    run("build_features.py")
    run("train.py")
    if args.no_llm:
        run("demo_replay.py", "--no-llm")
    else:
        run("get_granite.py")
        run("demo_replay.py")


if __name__ == "__main__":
    main()
