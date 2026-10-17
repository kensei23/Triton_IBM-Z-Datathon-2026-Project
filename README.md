# Triton

Sensors detect a pollution event upstream, the system predicts when it will reach a drinking water intake, and an agent decides whether to close the intake and alert the Environment Agency. This happens in real time, on critical infrastructure.


## Run it

    git clone https://github.com/kensei23/Triton_IBM-Z-Datathon-2026-Project.git
    cd Triton_IBM-Z-Datathon-2026-Project
    uv sync
    uv run pre-commit install
    uv run pytest

Requires [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv` on Windows). `uv sync` installs Python 3.13 and all dependencies from the lockfile.

Data isn't in the repo. See `data/README.md` for how to get it.

## Team

- Aaron Derin
- Oscar Denche
- Daniel Jones
