"""
Stage 5 — ANALYZE (model metrics overview)
Reads:  Results/*.csv
Writes: Results/Analysis/Metrics_Bars_byModel.png
        Results/Analysis/Metrics_Bars_byMetric.png
        Results/Analysis/Metrics_Radar_<model>.png
        Results/Analysis/Signed_Bias.png
        Results/Analysis/Confidence_Distribution.png

Stage 4 charted each model's calibration in detail. This stage steps back
and scores each model on FOUR simple, single-number metrics, then shows
them side by side so the models can be compared at a glance.

Every metric is reported two ways:
  - its RAW value (the natural, honest number), and
  - a NORMALIZED 0–1 version where 1 is always best and 0 is worst,
    so they can share one chart.

The four metrics:

  ACCURACY     fraction of answers correct.                  (raw 0–1)
  RESOLUTION   mean confidence when right  −  mean confidence
               when wrong. "Does the model get less sure on
               the ones it misses?" Higher = more informative
               confidence.                                   (raw −1–1)
  SPREAD       standard deviation of stated confidence. Low =
               the model says the same thing every time; high
               = it actually varies its confidence.          (raw 0–~0.5)
  BIAS         mean of (confidence − correct). Sign = lean:
               + overconfident, − underconfident, 0 centered. (raw −1–1)

The normalized BIAS column (1 − |bias|) measures how well-CENTERED a model
is, so it fits the "1 = best" charts. But that throws away the DIRECTION —
over vs. under — which is the whole point of this project. So bias also gets
its own dedicated chart (Signed_Bias.png) showing the raw signed value.

Requires matplotlib (pip install matplotlib).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("Results")
ANALYSIS_DIR = RESULTS_DIR / "Analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Order the metrics appear in tables and charts.
METRIC_ORDER = ["Accuracy", "Resolution", "Spread", "Bias"]


def pause(message="Press Enter to continue..."):
    input(f"\n  >>> {message}\n")


def step(number, title):
    print()
    print("=" * 60)
    print(f"STEP {number}: {title}")
    print("=" * 60)


def model_name_from_path(path):
    """Pull a friendly model name from a CSV filename:
       'Confidence_-_Claude_4_8_High.csv' -> 'Claude 4 8 High'"""
    import re
    stem = path.stem.replace("_", " ")
    stem = re.sub(r"^Confidence\s*-\s*", "", stem)
    return stem


# === Metrics ==================================================
# Each model's CSV becomes a small dict of raw metric values, plus a
# matching dict of normalized values where 1 = best.

def compute_metrics(df):
    """Return (raw, normalized) dicts of the four metrics for one model."""
    conf = df["Confidence"].str.rstrip("%").astype(float) / 100
    correct = (df["Answer"] == df["Correct"]).astype(int)

    accuracy = correct.mean()

    # Resolution: how much higher the model's confidence is on the
    # questions it got right vs. the ones it got wrong. Guard against a
    # model with zero right or zero wrong answers (then it's undefined).
    if correct.sum() > 0 and (1 - correct).sum() > 0:
        resolution = conf[correct == 1].mean() - conf[correct == 0].mean()
    else:
        resolution = 0.0

    spread = conf.std()
    bias = (conf - correct).mean()

    raw = {
        "Accuracy":   float(accuracy),
        "Resolution": float(resolution),
        "Spread":     float(spread),
        "Bias":       float(bias),
    }

    # Normalize so 1 is always best, 0 worst, all on a shared 0–1 scale.
    #   Accuracy   already 0–1, higher better → as is.
    #   Resolution −1..1, higher better → shift to (x+1)/2.
    #   Spread     0..~0.5, scaled ×2 to roughly fill 0–1 (clamped).
    #   Bias       best is 0 (centered); 1−|bias| so centered → 1.
    norm = {
        "Accuracy":   raw["Accuracy"],
        "Resolution": (raw["Resolution"] + 1) / 2,
        "Spread":     min(raw["Spread"] * 2, 1.0),
        "Bias":       1 - abs(raw["Bias"]),
    }
    return raw, norm


# === Charts ===================================================
# A soft, muted palette — gentler versions of the default matplotlib
# colors. One color PER METRIC, reused across every chart so a metric
# keeps its identity (Accuracy is always the same blue, etc).

METRIC_COLORS = {
    "Accuracy":   "#6b8fb5",   # soft blue
    "Resolution": "#e0a86b",   # soft orange
    "Spread":     "#7fae7f",   # soft green
    "Bias":       "#c98a8a",   # soft red
}
# A matching muted color per MODEL, for the per-model radar outlines.
MODEL_PALETTE = ["#6b8fb5", "#e0a86b", "#7fae7f", "#c98a8a",
                 "#a892c0", "#b5a16b", "#8aa9c9", "#c0929f"]


def plot_bars_by_model(norm_by_model, output_path):
    """Grouped bars: one cluster per MODEL, the metrics inside it.
    Good for 'how well-rounded is each model'."""
    models = list(norm_by_model)
    n_models = len(models)
    n_metrics = len(METRIC_ORDER)

    fig, ax = plt.subplots(figsize=(2 + 1.7 * n_models, 6))
    group_width = 0.8
    bar_width = group_width / n_metrics
    x = np.arange(n_models)

    for j, metric in enumerate(METRIC_ORDER):
        offsets = x - group_width / 2 + (j + 0.5) * bar_width
        values = [norm_by_model[m][metric] for m in models]
        ax.bar(offsets, values, bar_width, label=metric,
               color=METRIC_COLORS[metric])

    # Ticks sit at the cluster centers (x), labels centered under them.
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (1 = best)")
    ax.set_title("Metrics grouped by Model")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_bars_by_metric(norm_by_model, output_path):
    """Grouped bars: one cluster per METRIC, the models inside it.
    Good for 'how does each metric change across models'."""
    models = list(norm_by_model)
    n_models = len(models)
    n_metrics = len(METRIC_ORDER)

    fig, ax = plt.subplots(figsize=(2 + 1.7 * n_metrics, 6))
    group_width = 0.8
    bar_width = group_width / n_models
    x = np.arange(n_metrics)

    for j, model in enumerate(models):
        offsets = x - group_width / 2 + (j + 0.5) * bar_width
        values = [norm_by_model[model][m] for m in METRIC_ORDER]
        ax.bar(offsets, values, bar_width, label=model,
               color=MODEL_PALETTE[j % len(MODEL_PALETTE)])

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_ORDER, rotation=0, ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (1 = best)")
    ax.set_title("Metrics grouped by Metric")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_radar_single(model, norm, color, output_path):
    """One radar/spider chart for a single model."""
    labels = METRIC_ORDER
    n = len(labels)
    # Angles for each axis, closing the loop back to the start.
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    values = [norm[m] for m in labels]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, values, linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(f"{model}", pad=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_signed_bias(raw_by_model, output_path):
    """Diverging vertical bars of RAW signed bias (confidence − accuracy).
    Bars ABOVE zero = overconfident; BELOW = underconfident; height = how
    much. This is the project's headline metric, shown with its direction
    intact (the normalized metric charts can only show magnitude)."""
    models = list(raw_by_model)
    x = np.arange(len(models))
    biases = np.array([raw_by_model[m]["Bias"] for m in models])

    # Color by direction: over = warm, under = cool.
    colors = ["#c98a8a" if b >= 0 else "#6b8fb5" for b in biases]

    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(models), 6))
    ax.bar(x, biases, 0.55, color=colors)
    ax.axhline(0, color="black", linewidth=1)

    # Label each bar with its value, just outside the bar end.
    for xi, b in zip(x, biases):
        va = "bottom" if b >= 0 else "top"
        off = 0.004 if b >= 0 else -0.004
        ax.text(xi, b + off, f"{b:+.1%}", ha="center", va=va, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha="center", fontsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_ylabel("Signed bias  (confidence − accuracy)")
    # Symmetric limits so the zero line sits in the middle.
    span = max(0.05, np.abs(biases).max() * 1.3)
    ax.set_ylim(-span, span)
    ax.set_title("Overconfident (above) vs. Underconfident (below)")

    # Light directional guides in the margins.
    ax.text(0.01, 0.97, "overconfident", transform=ax.transAxes,
            color="#c98a8a", fontsize=9, va="top")
    ax.text(0.01, 0.03, "underconfident", transform=ax.transAxes,
            color="#6b8fb5", fontsize=9, va="bottom")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def confidence_distribution(df, levels):
    """Fraction of a model's answers that fell at each confidence level.
    Returns a list aligned with `levels` (each in 0–1, summing to 1)."""
    conf = df["Confidence"].str.rstrip("%").astype(float) / 100
    counts = conf.value_counts(normalize=True)  # level -> fraction
    return [float(counts.get(level, 0.0)) for level in levels]


def plot_confidence_distribution(dist_by_model, levels, output_path):
    """Stacked bar chart: one bar per model, segments showing what
    fraction of its answers landed at each confidence level. Shows how
    a model SPENDS its confidence — does it hedge low, commit high, or
    spread out? Each bar sums to 100%."""
    models = list(dist_by_model)
    x = np.arange(len(models))

    # A sequential shade per confidence level (darker = more confident),
    # so the stack reads like a gradient from low to high confidence.
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(levels) - 1, 1)) for i in range(len(levels))]

    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(models), 6))
    bottom = np.zeros(len(models))
    for k, level in enumerate(levels):
        heights = np.array([dist_by_model[m][k] for m in models])
        ax.bar(x, heights, 0.6, bottom=bottom, color=colors[k],
               label=f"{level:.0%}")
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Share of answers")
    ax.set_title("How each model spends its confidence")
    ax.legend(title="Confidence", loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# === The main pipeline ========================================
def main():
    print()
    print("STAGE 5: ANALYZE — METRICS OVERVIEW")
    print()
    print("Instead of one deep chart per model, this stage scores every")
    print("model on FOUR quick numbers and lines them up for comparison.")
    print("Each number captures a different question:")
    print()
    print("  ACCURACY    — how often is it right at all?")
    print("  RESOLUTION  — is it less confident on the ones it gets wrong?")
    print("                (confidence-when-right minus confidence-when-wrong)")
    print("  SPREAD      — does it vary its confidence, or say the same")
    print("                thing every time? (std of stated confidence)")
    print("  BIAS        — does it lean over- or under-confident overall?")
    print("                (average of confidence minus correctness)")
    print()
    print("These have different natural ranges and directions — some go")
    print("negative, etc. To chart them together we NORMALIZE each to a")
    print("0–1 scale where 1 is always best:")
    print()
    print("  Accuracy    already 0–1                → unchanged")
    print("  Resolution  −1..1, higher better       → (x + 1) / 2")
    print("  Spread      0..~0.5                     → x × 2 (capped at 1)")
    print("  Bias        best is 0 (centered)        → 1 − |x|")
    print()
    print("Note on BIAS: the normalized 1−|bias| only shows HOW CENTERED a")
    print("model is, not WHICH WAY it leans. Since over- vs. under-confidence")
    print("is the heart of this project, bias also gets its own chart showing")
    print("the raw signed value with direction intact.")
    print()
    print("We'll print the raw and normalized values, then draw: bars grouped")
    print("by model, bars grouped by metric, one radar per model, a signed-")
    print("bias chart, and a stacked confidence-distribution chart.")
    pause("Press Enter to start...")

    # ----------------------------------------------------------
    step(1, "Score every model")
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"\n  No CSV files found in {RESULTS_DIR}.")
        print("  Add some model result CSVs and run again.")
        return

    raw_by_model = {}
    norm_by_model = {}
    dist_by_model = {}
    for csv_file in csv_files:
        name = model_name_from_path(csv_file)
        df = pd.read_csv(csv_file)
        raw, norm = compute_metrics(df)
        raw_by_model[name] = raw
        norm_by_model[name] = norm
        dist_by_model[name] = df  # keep frame for the distribution chart

    print("\n  RAW metric values:")
    header = "    " + f"{'model':<26}" + "".join(f"{m:>11}" for m in METRIC_ORDER)
    print(header)
    print("    " + "-" * (26 + 11 * len(METRIC_ORDER)))
    for name, raw in raw_by_model.items():
        row = "".join(f"{raw[m]:>+11.3f}" for m in METRIC_ORDER)
        print(f"    {name:<26}{row}")

    print("\n  NORMALIZED (0–1, 1 = best):")
    print(header)
    print("    " + "-" * (26 + 11 * len(METRIC_ORDER)))
    for name, norm in norm_by_model.items():
        row = "".join(f"{norm[m]:>11.2f}" for m in METRIC_ORDER)
        print(f"    {name:<26}{row}")
    print()
    print("  Reminder while reading the raw table: Accuracy/Resolution/Spread")
    print("  higher is better; Bias closer to 0 is better. The normalized")
    print("  table flips them all to 'higher = better' — but watch the bias")
    print("  caveat above.")
    pause()

    # ----------------------------------------------------------
    step(2, "Bar charts — two groupings")
    print("First, bar charts. The same numbers can be grouped two ways, and")
    print("each answers a different question:")
    print()
    print("  BY MODEL: each model gets a cluster of its metrics. Tall bars")
    print("  everywhere = a well-rounded model; a short bar flags a specific")
    print("  weakness. Best for sizing up one model at a time.")
    print()
    print("  BY METRIC: each metric gets a cluster of the models. Now the")
    print("  same metric's bars sit side by side, so you can see how it")
    print("  RISES OR FALLS across models — who wins accuracy, who's least")
    print("  biased, and so on. Best for comparing models on one quality.")
    print()
    print("Colors are consistent: each metric keeps its color in the")
    print("by-model chart, each model keeps its color in the by-metric chart.")

    bars_model_path = ANALYSIS_DIR / "Metrics_Bars_byModel.png"
    plot_bars_by_model(norm_by_model, bars_model_path)
    print(f"\n  Wrote {bars_model_path}")

    bars_metric_path = ANALYSIS_DIR / "Metrics_Bars_byMetric.png"
    plot_bars_by_metric(norm_by_model, bars_metric_path)
    print(f"  Wrote {bars_metric_path}")
    pause()

    # ----------------------------------------------------------
    step(3, "Radar charts — one per model")
    print("Next, a RADAR (spider) chart for EACH model. The metrics become")
    print("spokes; the model is a closed shape. A big, round shape means")
    print("strong and balanced; a dented or lopsided shape shows where a")
    print("model trades one quality for another. One chart per model keeps")
    print("each shape clean and easy to read on its own.")
    print()

    radar_paths = []
    for j, (model, norm) in enumerate(norm_by_model.items()):
        color = MODEL_PALETTE[j % len(MODEL_PALETTE)]
        safe = model.replace(" ", "_")
        radar_path = ANALYSIS_DIR / f"Metrics_Radar_{safe}.png"
        plot_radar_single(model, norm, color, radar_path)
        radar_paths.append(radar_path)
        print(f"  Wrote {radar_path}")
    pause()

    # ----------------------------------------------------------
    step(4, "Signed bias — the headline metric")
    print("The normalized charts above can only show how CENTERED each model")
    print("is, because everything there is forced onto a '1 = best' scale.")
    print("But the central question of this whole project is DIRECTIONAL:")
    print("is a model over- or under-confident, and by how much?")
    print()
    print("This chart shows the raw signed bias (confidence − accuracy) as")
    print("diverging vertical bars around a zero line:")
    print()
    print("  - bars ABOVE zero  → OVERCONFIDENT (claimed more than earned)")
    print("  - bars BELOW zero  → UNDERCONFIDENT (earned more than claimed)")
    print("  - bar height       → how far from perfectly centered")
    print()
    print("Zero is perfect calibration; the further a bar runs in either")
    print("direction, the more that model's confidence misleads.")

    bias_path = ANALYSIS_DIR / "Signed_Bias.png"
    plot_signed_bias(raw_by_model, bias_path)
    print(f"\n  Wrote {bias_path}")
    pause()

    # ----------------------------------------------------------
    step(5, "Confidence distribution (stacked bars)")
    print("The metrics above are all about whether confidence was JUSTIFIED.")
    print("This last chart asks a simpler question: where did each model PUT")
    print("its confidence? For every model we count what fraction of its")
    print("answers landed at each level — 30%, 40%, ... 100% — and stack")
    print("those fractions into one bar per model (each bar sums to 100%).")
    print()
    print("It makes the 'spread' metric concrete. A model that commits to")
    print("100% on most answers shows one big top segment; a model that")
    print("hedges across the range shows a more even stack. This is often")
    print("the clearest single picture of a model's confidence 'personality'.")

    # Union of all confidence levels any model used, sorted low → high.
    levels = sorted({
        float(c.rstrip("%")) / 100
        for df in dist_by_model.values()
        for c in df["Confidence"]
    })
    dist = {name: confidence_distribution(df, levels)
            for name, df in dist_by_model.items()}

    dist_path = ANALYSIS_DIR / "Confidence_Distribution.png"
    plot_confidence_distribution(dist, levels, dist_path)
    print(f"\n  Wrote {dist_path}")
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print("One honest caveat: normalizing different metrics onto a shared")
    print("0–1 scale makes them comparable to LOOK at, but the scalings are")
    print("choices, not laws (why ×2 for spread, not ×1.8?). Treat the charts")
    print("as a quick overview and go back to the raw numbers — and Stage 4's")
    print("detailed charts — before drawing firm conclusions.")


if __name__ == "__main__":
    main()
