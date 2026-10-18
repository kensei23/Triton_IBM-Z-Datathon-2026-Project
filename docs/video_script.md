# Video script (under 3 minutes)

About 420 words of narration, which is roughly 2 minutes 50 seconds at a calm
pace. Times are targets. **[Screen]** says what to show, the rest is what to say.

## 0:00 - 0:25 · The problem

**[Screen]** A river photo or a map of Thames Water storm overflows. Then the number "31,861" large on screen.

> When heavy rain overwhelms a sewer, storm overflows release raw sewage into
> rivers. Thames Water's overflows did it 31,861 times in 2024. Today operators
> find out once a spill has already started. We built Triton to give them
> warning a few hours before, so they can act.

## 0:25 - 0:50 · The data

**[Screen]** `data/README.md` or the EDA notebook: spills-per-month chart, then the "risk rises with rain" chart.

> Triton uses only open data: every spill start and stop recorded at Thames
> Water's overflows since January 2024, from the Environment Agency and Thames
> Water's live API, and hourly rainfall at each overflow from Open-Meteo. Spills
> follow rain, but also how wet the ground already is, and even the time of day:
> they peak in the morning, when everyone showers and a full treatment works
> tips over.

## 0:50 - 1:25 · The solution

**[Screen]** The Mermaid architecture diagram from the README.

> Every hour, Triton detects, decides and responds. A calibrated
> gradient-boosting model scores every overflow: the chance a new spill starts
> in the next six hours. Above 12 percent, Triton intervenes: it starts emptying
> the storm tank and raises an alert. Above 5 percent, it notifies an operator.
> Otherwise it keeps watching. For every alert, IBM Granite writes a
> plain-English explanation from the facts, and a fact check throws out anything
> the model made up.

## 1:25 - 2:25 · Live demo on IBM LinuxONE

**[Screen]** Terminal, SSH session on the VM. Type `uname -m` (shows `s390x`), then:

```
uv run python scripts/demo_replay.py --delay 0.6
```

Let the header show `machine : s390x | Linux | IBM z17 ...` and the Granite line, then let the storm play.

> This is running on an IBM z17 LinuxONE machine, CPU only, no GPU and no cloud.
> We're replaying the storm of the 7th of January 2026 hour by hour, data the
> model never saw in training. As the rain builds, risk climbs... here Triton
> notifies an operator, and Granite explains why: the rain over the last 30 days
> and the morning peak in sewer flow. And here the spill really starts, about
> two hours after the warning.

**[Screen]** At the end, the scorecard. Then open `outputs/demo_timeline.html` and pick a site from the dropdown.

> In this storm, 21 of 33 spills were warned before they started. The timeline
> shows rain, the predicted risk crossing the thresholds, and the real spill.

## 2:25 - 2:50 · Results and impact

**[Screen]** README results table, then `outputs/figures/pr_curve.png`.

> Over the whole of 2026 so far, Triton warned about 76 percent of spill
> episodes, a median five hours ahead, with almost five times fewer alerts than
> a rain-only rule. Everything, the model, the decisions and the language model,
> runs on IBM Z, close to the data and secure.

## 2:50 - 3:00 · Close

**[Screen]** Repo URL and team names.

> Triton: detect, decide, respond, before the sewage reaches the river. Thank you.

## Recording tips

- Record the terminal at a large font size (18pt or more) and a dark theme.
- Run the demo once before recording so the model and Granite are cached on the VM.
- Granite takes about 13 s per alert on the VM, so the full replay runs about 7 minutes. For the recording use `--max-llm 6` (Granite writes the first 6 alerts, the rest use the template) and cut the waits in editing.
- If the replay feels long, trim the quiet hours in editing; don't speed up the numbers.
- Keep the final cut under 3:00 (the rules say 3 minutes or less).
