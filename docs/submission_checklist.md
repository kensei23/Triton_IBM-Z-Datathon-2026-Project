# Submission checklist

Deadline: **Sunday 18 October 2026, 10:00 UK time.**

## Required

- [ ] **Title** in the format `[Team Name_IBM Z Datathon 2026 Project]`, for example `[Triton_IBM Z Datathon 2026 Project]` (confirm the registered team name)
- [ ] **Repository link:** https://github.com/kensei23/Triton_IBM-Z-Datathon-2026-Project, with the `daniel` branch merged into `main` and the repo public or shared with the judges
- [ ] **Demo video**, 3 minutes or less, script in [`video_script.md`](video_script.md), showing the system running on the LinuxONE VM
- [ ] **Flowchart:** the Mermaid diagram in the README (GitHub renders it; export a PNG too if the form asks for an image)
- [ ] **Links to all open-source libraries and data sources:** README sections "Data sources" and "Open-source software"

## Before submitting

- [ ] `uv run pytest -q` and `uv run ruff check .` pass on `main`
- [ ] `models/spill_model.joblib` and `outputs/` committed from the VM run (trained on IBM Z)
- [ ] README results match `outputs/metrics.json` from the VM run
- [ ] No LLM weights or data files committed (`git ls-files | grep -E "gguf|data/raw|data/processed"` prints nothing)
- [ ] Video uploaded and its link works without signing in

## What judges score, and where we show it

| Criterion | Evidence |
|---|---|
| Community value, theme fit | Sewage spills into rivers; Detect → Decide → Respond; README "The problem" |
| Originality | Spill prediction per overflow from open data, with a fact-checked language model writing operator alerts |
| IBM Z / LinuxONE with AI/ML on it | Training, inference and Granite all on the z17 CPU; llama.cpp built with s390x VXE2; big-endian handling; README "How it uses IBM Z and LinuxONE" |
| Presentation | Video, live replay demo, interactive timeline, README figures |
