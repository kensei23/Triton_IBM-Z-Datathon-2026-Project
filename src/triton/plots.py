"""Figures for the evaluation, one consistent style."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

MODEL, BASELINE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "legend.frameon": False,
        "figure.dpi": 120,
    }
)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pr_curves(y, p_model, p_base, thresholds, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for p, color, name in [(p_model, MODEL, "model"), (p_base, BASELINE, "rain rule")]:
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, color=color, linewidth=2, label=name)
    for name, t in thresholds.items():
        hit = p_model >= t
        prec, rec = y[hit].mean(), hit[y == 1].mean()
        ax.plot(rec, prec, "o", color=MODEL, markersize=8, markeredgecolor="white")
        ax.annotate(
            f"{name} (p ≥ {t:.2f})",
            (rec, prec),
            textcoords="offset points",
            xytext=(8, 4),
            color=INK,
        )
    ax.axhline(y.mean(), color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("recall (share of spill hours flagged)")
    ax.set_ylabel("precision")
    ax.set_title("Precision vs recall, 2026 test set")
    ax.legend(loc="upper right")
    _save(fig, path)


def calibration(y, p_model, p_base, path):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    top = max(p_model.max(), p_base.max()) * 1.05
    ax.plot([0, top], [0, top], color=MUTED, linewidth=1, linestyle="--")
    for p, color, name in [(p_model, MODEL, "model"), (p_base, BASELINE, "rain rule")]:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "-o", color=color, linewidth=2, markersize=6, label=name)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed spill rate")
    ax.set_title("Calibration, 2026 test set")
    ax.legend(loc="upper left")
    _save(fig, path)


def lead_hist(leads, threshold, path):
    caught = leads[~np.isnan(leads)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(caught, bins=np.arange(0, 6.5, 0.5), color=MODEL, rwidth=0.9)
    ax.set_xlabel("hours of warning before the spill started")
    ax.set_ylabel("spill episodes")
    ax.set_title(f"Warning time at NOTIFY (p ≥ {threshold:.2f})")
    ax.text(
        0.02,
        0.95,
        f"{len(caught)} of {len(leads)} spill episodes warned",
        transform=ax.transAxes,
        va="top",
        color=MUTED,
    )
    _save(fig, path)


def importance(names, values, path):
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(names)[order], np.array(values)[order], color=MODEL, height=0.6)
    ax.set_xlabel("drop in PR-AUC when the feature is shuffled")
    ax.set_title("What drives the prediction")
    _save(fig, path)
