"""
Stage 4 — CALIBRATION CHARTS
Reads:  Results/*.csv
Writes: Results/Charts/*.png

Each CSV is one model's run over the question set. We plot how well
each model's stated CONFIDENCE matches its actual ACCURACY. Perfect
calibration is the y=x diagonal. Points above the line mean the model
is too cautious (right more often than it claimed); below the line
means overconfident.

We produce one chart per model (with error bars showing statistical
uncertainty at each point) and one combined chart with all models
overlaid (smoothed curves, no error bars).

Requires matplotlib and scipy (pip install matplotlib scipy).
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.interpolate import PchipInterpolator


RESULTS_DIR = Path("Results")
CHARTS_DIR = RESULTS_DIR / "Charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# With at most 5 multiple-choice options, pure random guessing
# scores about 20%, so there's no meaningful calibration below
# that threshold. We anchor both axes there.
AXIS_MIN = 0.2
AXIS_MAX = 1.0
TICKS = np.round(np.arange(AXIS_MIN, AXIS_MAX + 0.05, 0.1), 1)


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


# === Reference curves ========================================
# Two smooth reference curves that show what under- and over-
# confidence look like in the chart space. Both are anchored at
# the chart's lower-left and upper-right corners; they bow away
# from the diagonal in between. We control how much they bow with
# the coefficient — bigger numbers = more bowing.

def underconfidence_curve(x):
    """Bows ABOVE the diagonal — a model whose accuracy beats its claims."""
    return x + 1.5 * (x - AXIS_MIN) * (1 - x)


def overconfidence_curve(x):
    """Bows BELOW the diagonal — a model whose claims beat its accuracy."""
    return x - 1.5 * (x - AXIS_MIN) * (1 - x)


# === Calibration data =========================================

def compute_calibration(df):
    """
    Group rows by confidence level. For each bucket, compute:
      - proportion of rows where the model was correct
      - standard deviation of that proportion estimate
      - bucket size

    Returns a sorted list of (confidence, proportion, sd, count).
    """
    df = df.copy()
    # "80%" -> 0.80
    df["Confidence"] = df["Confidence"].str.rstrip("%").astype(float) / 100
    df["IsCorrect"] = (df["Answer"] == df["Correct"]).astype(int)

    buckets = []
    for conf, group in df.groupby("Confidence"):
        n = len(group)
        p = group["IsCorrect"].mean()
        # Standard deviation of a binomial proportion — a measure of
        # how much our estimate might wobble from the true value.
        # Bigger n → smaller wobble → narrower error bars.
        sd = math.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        buckets.append((conf, p, sd, n))

    return sorted(buckets)


# === Chart drawing ============================================

def setup_axes(ax, title):
    """Common axis configuration shared by both chart types."""
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    # PercentFormatter turns the underlying 0.3 tick into the label "30%".
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Proportion Correct")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def draw_reference_lines(ax):
    """Draw the three reference curves: perfect, under, over.
    The under/over curves are labeled directly on the chart so an
    observer can see at a glance which region is which, without
    having to glance back at the legend."""
    x = np.linspace(AXIS_MIN, AXIS_MAX, 200)
    # Perfect calibration — the diagonal y = x. Stays in the legend.
    ax.plot(x, x, "--", color="gray", linewidth=1.2, alpha=0.8,
            label="Perfect calibration")
    # Underconfident reference (above the diagonal). Dotted, faint,
    # no legend entry — labeled on-chart below.
    ax.plot(x, underconfidence_curve(x), ":", color="gray",
            linewidth=1, alpha=0.45)
    # Overconfident reference (below the diagonal).
    ax.plot(x, overconfidence_curve(x), ":", color="gray",
            linewidth=1, alpha=0.45)
    # In-chart labels for each region. Placed symmetrically above and
    # below the diagonal so it's visually obvious which side is which.
    ax.text(0.36, 0.78, "Underconfident", color="gray", alpha=0.75,
            style="italic", ha="center", va="center", fontsize=10)
    ax.text(0.78, 0.36, "Overconfident", color="gray", alpha=0.75,
            style="italic", ha="center", va="center", fontsize=10)


def plot_single_model(name, data, output_path):
    """One model's calibration curve with error bars."""
    confidences = [b[0] for b in data]
    proportions = [b[1] for b in data]
    errors = [b[2] for b in data]

    fig, ax = plt.subplots(figsize=(7, 7))
    setup_axes(ax, f"Calibration: {name}")
    draw_reference_lines(ax)

    ax.errorbar(confidences, proportions, yerr=errors,
                fmt="o-", capsize=4, markersize=6, linewidth=1.5,
                label=name)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_combined(all_data, output_path):
    """Every model on one chart, smoothed lines, no error bars."""
    fig, ax = plt.subplots(figsize=(8, 7))
    setup_axes(ax, "Calibration: All Models")
    draw_reference_lines(ax)

    for name, data in all_data.items():
        confidences = np.array([b[0] for b in data])
        proportions = np.array([b[1] for b in data])

        if len(confidences) >= 2:
            # PCHIP — a monotonic cubic interpolator. It draws a
            # smooth curve through every data point and won't
            # overshoot the way a plain cubic spline might.
            spline = PchipInterpolator(confidences, proportions)
            x_smooth = np.linspace(confidences.min(), confidences.max(), 200)
            y_smooth = spline(x_smooth)
            line, = ax.plot(x_smooth, y_smooth, "-",
                            linewidth=1.5, label=name)
            # Show the original data points in the same color.
            ax.plot(confidences, proportions, "o",
                    color=line.get_color(), markersize=4)
        else:
            ax.plot(confidences, proportions, "o-", markersize=5,
                    linewidth=1.5, label=name)

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# === The main pipeline ========================================
def main():
    print()
    print("STAGE 4: CALIBRATION CHARTS")
    print()
    print("After running our 600-question benchmark, each model produced")
    print("a CSV listing its answers and the CONFIDENCE it claimed in each")
    print("one. Now we ask the real question: were those confidence claims")
    print("trustworthy? When a model said it was 80% sure, was it actually")
    print("right about 80% of the time?")
    print()
    print("The match between SAID-confidence and ACTUAL-accuracy is called")
    print("CALIBRATION. A well-calibrated model is right 70% of the time")
    print("when it claims 70%, right 90% when it claims 90%, and so on.")
    print()
    print("There are two ways a model can be miscalibrated:")
    print()
    print("  OVERCONFIDENT: claims high confidence but is often wrong.")
    print("  E.g., says '90% sure' but gets only 60% right. Dangerous —")
    print("  this model SOUNDS reliable when it isn't.")
    print()
    print("  UNDERCONFIDENT: hedges more than it needs to. Says '70%' on")
    print("  questions it actually gets right 95% of the time. Annoying")
    print("  but less risky — it warns you more than necessary.")
    print()
    print("Calibration chart anatomy:")
    print("  - X axis: the confidence the model expressed (20% to 100%).")
    print("    Why start at 20%? Because every question has 5 options, so")
    print("    pure random guessing scores ~20%. Anything below that just")
    print("    means the model is confused, not calibrated.")
    print("  - Y axis: the fraction of those answers that were correct.")
    print("  - The DASHED diagonal (y = x) is PERFECT CALIBRATION.")
    print("  - Two faint DOTTED curves show what underconfidence and")
    print("    overconfidence look like as a visual reference.")
    print("  - Points ABOVE the diagonal: UNDERCONFIDENT at that level.")
    print("  - Points BELOW the diagonal: OVERCONFIDENT at that level.")
    print()
    print("We'll make two kinds of charts:")
    print("  1. One per model, with error bars showing how much uncertainty")
    print("     there is in each data point (driven by the sample size).")
    print("  2. One combined chart with all models overlaid as smoothed")
    print("     curves, no error bars (with several models stacked, the")
    print("     bars would get unreadable).")
    pause("Press Enter to start...")

    # ----------------------------------------------------------
    step(1, "Find all model CSVs in Results/")
    print("Each model's run lives in its own CSV in the Results/ folder.")
    print("We look at the filenames to figure out which models we have.")

    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"\n  No CSV files found in {RESULTS_DIR}!")
        print("  Put some model result CSVs in there and run again.")
        return

    print(f"\n  Found {len(csv_files)} model CSV(s):")
    for f in csv_files:
        name = model_name_from_path(f)
        print(f"    {f.name}")
        print(f"        → display name: '{name}'")
    pause()

    # ----------------------------------------------------------
    step(2, "Compute calibration data for each model")
    print("For each CSV we BUCKET the rows by confidence level — all the")
    print("rows where the model said '70%' go together, all the '80%' rows")
    print("together, and so on. These groups are the 'buckets'.")
    print()
    print("Within each bucket we compute the model's actual accuracy:")
    print()
    print("    proportion correct = (rows with right answer) / (rows in bucket)")
    print()
    print("That's our Y-coordinate for that confidence level — the point we")
    print("plot against the diagonal.")
    print()
    print("But there's a catch: each bucket has a LIMITED NUMBER of rows.")
    print("The proportion we compute is just an ESTIMATE of the model's")
    print("'true' accuracy at that confidence level. If a bucket has only")
    print("5 questions, we shouldn't trust the estimate much; if it has")
    print("100, we can trust it more. The ERROR BAR captures this — it")
    print("shows how wide a range we should expect the true accuracy to")
    print("fall in.")
    print()
    print("The formula is the STANDARD DEVIATION of a proportion:")
    print()
    print("    SD = sqrt( p * (1 - p) / n )")
    print()
    print("where p is the proportion correct and n is the bucket size.")
    print("Standard deviation is just a measure of SPREAD — how much")
    print("the estimate might wobble from the true value. Bigger buckets")
    print("give smaller SDs (we're more sure); smaller buckets give")
    print("bigger SDs (we're less sure).")
    print()
    print("We draw the error bars at ±1 SD around each dot — a 'typical'")
    print("range the real accuracy is likely to land in.")
    print()
    print("Heads up: at low confidence levels n is often small, so the")
    print("bars come out wider. That's not a bug — it reflects how")
    print("little data we have to estimate from in those buckets.")
    print()
    print("Bucket counts and proportions for each model:")
    print()

    all_calibrations = {}
    for csv_file in csv_files:
        name = model_name_from_path(csv_file)
        df = pd.read_csv(csv_file)
        data = compute_calibration(df)
        all_calibrations[name] = data

        print(f"  {name}:")
        print(f"    {'conf':>6}  {'correct':>8}  {'±SD':>7}  {'n':>5}")
        for conf, p, sd, n in data:
            print(f"    {conf:>6.0%}  {p:>8.0%}  {sd:>7.0%}  {n:>5}")
        print()
    pause()

    # ----------------------------------------------------------
    step(3, "Make a per-model chart with error bars")
    print("Time to draw. For each model we plot:")
    print()
    print("  - The dashed gray diagonal — PERFECT CALIBRATION, where the")
    print("    model would land if its confidence claims were exactly")
    print("    right on average.")
    print("  - Two faint DOTTED reference curves bowing above and below")
    print("    the diagonal, showing what underconfidence and over-")
    print("    confidence look like in chart space. The model's actual")
    print("    curve sits somewhere relative to these.")
    print("  - A solid line with dots at each bucket's (confidence,")
    print("    proportion). Above the diagonal = underconfident here;")
    print("    below = overconfident here.")
    print("  - Vertical error bars on each dot — ±1 standard deviation,")
    print("    from step 2. Long bars mean we're not very sure where the")
    print("    dot 'really' is; short bars mean we have lots of data.")
    print()
    print("matplotlib is the standard Python plotting library. We hand")
    print("it lists of x-values, y-values, and error sizes, and it draws")
    print("the chart. fig.savefig() writes the result to a PNG file.")
    print()
    print("Two other small details:")
    print("  - The axis ticks are formatted as PERCENTAGES (20%, 30%,")
    print("    ..., 100%) — easier to read than decimals (0.2, 0.3...).")
    print("  - Both axes start at 20% and end at 100%, with a tick every")
    print("    10% so every confidence bin gets its own label.")
    print()

    for csv_file in csv_files:
        name = model_name_from_path(csv_file)
        data = all_calibrations[name]
        filename = name.replace(" ", "_") + ".png"
        output_path = CHARTS_DIR / filename
        plot_single_model(name, data, output_path)
        print(f"  Wrote {output_path}")
    pause()

    # ----------------------------------------------------------
    step(4, "Make the combined chart (all models, smoothed)")
    print("Now the comparison chart. All models on the same axes, each")
    print("drawn as its own colored line. This is the chart that lets")
    print("you answer questions like:")
    print()
    print("  - Which model is best-calibrated overall (hugs the diagonal)?")
    print("  - Is there a confidence level where ALL the models struggle?")
    print("  - Are the cheaper models more overconfident than the big ones?")
    print()
    print("Two differences from the per-model charts:")
    print()
    print("  1. NO error bars. With several lines stacked on top of each")
    print("     other, the bars would overlap and turn the chart into a")
    print("     tangle. The per-model charts already show uncertainty.")
    print()
    print("  2. SMOOTHED lines. Instead of straight segments between data")
    print("     points, we draw a smooth curve through them. We use a")
    print("     technique called PCHIP — short for 'Piecewise Cubic")
    print("     Hermite Interpolating Polynomial'. It's a kind of cubic")
    print("     spline that passes exactly through every data point but")
    print("     won't 'overshoot' between them (a problem with plain")
    print("     cubic splines, which can wiggle past the data). The")
    print("     result reads as one continuous curve per model rather")
    print("     than as a zigzag.")
    print()
    print("     The original data points are still drawn as small dots,")
    print("     so you can see where the actual measurements are versus")
    print("     where the smoothing fills in between them.")

    combined_path = CHARTS_DIR / "Combined.png"
    plot_combined(all_calibrations, combined_path)

    print(f"\n  Wrote {combined_path}")
    pause()

    # ----------------------------------------------------------
    step(5, "Done")
    print(f"All charts saved in {CHARTS_DIR}/")
    print()
    print("Open them up and look for:")
    print("  - Models that hug the diagonal closely — these are well-")
    print("    calibrated and can be trusted to know what they know.")
    print("  - Curves that DROP BELOW at high confidence — these are")
    print("    bluffers, claiming 90% certainty but missing too often.")
    print("  - Curves that RISE ABOVE at low confidence — these are")
    print("    hedgers, claiming uncertainty even when they're usually")
    print("    right.")


if __name__ == "__main__":
    main()
