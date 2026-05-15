"""
Stage 3 of 3 — SAMPLE
Reads:  Questions/Cleaned/UnifiedQA.csv
Writes: Questions/UnifiedQA_sample_600.csv

Sample 200 questions from each source, combine, shuffle, save.
The random seed makes the result reproducible — anyone running
this script gets the same 600 questions in the same order.
"""

from pathlib import Path
import pandas as pd


INPUT_FILE = Path("Questions/Cleaned/UnifiedQA.csv")
OUTPUT_FILE = Path("Questions/UnifiedQA_sample_600.csv")

SAMPLE_PER_SOURCE = 200
RANDOM_SEED = 42


def pause(message="Press Enter to continue..."):
    input(f"\n  >>> {message}\n")


def step(number, title):
    print()
    print("=" * 60)
    print(f"STEP {number}: {title}")
    print("=" * 60)


def main():
    print()
    print("STAGE 3: SAMPLE")
    print()
    print("After Stage 2 we have ONE big unified dataset. But it has two")
    print("problems that make it inconvenient to actually use:")
    print()
    print("  1. It's TOO BIG. There are tens of thousands of questions,")
    print("     more than any human is going to answer.")
    print()
    print("  2. It's UNBALANCED. One source contributed thousands of rows")
    print("     while another contributed only a few hundred. If we ran a")
    print("     quiz on this dataset, the largest source would dominate —")
    print("     the results wouldn't tell us much about the smaller ones.")
    print()
    print("This stage fixes both problems at once: we take a SAMPLE — a")
    print(f"random subset — of {SAMPLE_PER_SOURCE} questions from each source. That gives")
    print(f"us {SAMPLE_PER_SOURCE * 3} questions total, equally split. Then we shuffle them")
    print("into a random order so the sources are mixed throughout.")
    pause("Press Enter to start...")

    # ----------------------------------------------------------
    step(1, "Read the unified dataset")
    print("First, load the unified CSV that Stage 2 produced. pandas reads")
    print("it into a DataFrame — its tabular data structure, kind of like")
    print("a programmable spreadsheet.")

    df = pd.read_csv(INPUT_FILE)
    print(f"\n  Loaded {len(df)} rows from {INPUT_FILE}")
    print(f"  Columns: {list(df.columns)}")
    pause()

    # ----------------------------------------------------------
    step(2, "See how unbalanced the dataset is")
    print("Before we can balance the dataset, we need to SEE the imbalance.")
    print("Each row has an 'ID' that starts with its source name, like:")
    print()
    print("    LogiQA-042")
    print("    CommonsenseQA-1337")
    print("    OpenTriviaQA-8910")
    print()
    print("We can extract the source name by SPLITTING each ID on the '-'")
    print("character and taking the first piece. Then pandas' value_counts()")
    print("tallies up how many rows we have per source.")

    sources = df["ID"].astype(str).str.split("-").str[0]
    counts_before = sources.value_counts().sort_index()

    print(f"\n  Rows per source BEFORE sampling:")
    for name, count in counts_before.items():
        print(f"    {name:<20} {count:>6}")
    print()
    biggest = counts_before.idxmax()
    smallest = counts_before.idxmin()
    print(f"  {biggest} has {counts_before.max()} rows; {smallest} has")
    print(f"  only {counts_before.min()}. That's the imbalance we need to fix.")
    pause()

    # ----------------------------------------------------------
    step(3, "Take an equal sample from each source")
    print(f"For each source, we pull out {SAMPLE_PER_SOURCE} rows at random. Whichever")
    print("source had thousands of rows contributes 200; whichever had only")
    print("hundreds still contributes 200. After this step, every source")
    print("has equal weight in our final dataset.")
    print()
    print("Now we need to talk about RANDOMNESS — because this is where")
    print("most beginners trip up.")
    print()
    print("We use random sampling so we're not just taking the FIRST 200")
    print("rows from each source, which might be biased (maybe the original")
    print("file was sorted by difficulty, by date, by topic, etc). Random")
    print("sampling gives us a fair, representative slice.")
    print()
    print("But we ALSO need REPRODUCIBILITY. If you run this script today")
    print("and a classmate runs it tomorrow, we want you both to get THE")
    print("SAME 200 rows from each source — otherwise comparing results")
    print("would be meaningless. 'Same input, same output' is one of the")
    print("core promises a well-behaved program should make.")
    print()
    print(f"The 'random_state={RANDOM_SEED}' parameter solves the conflict. It")
    print("SEEDS the random number generator with a fixed value (42 is the")
    print("traditional choice). The samples are still 'random' in the sense")
    print("that they're spread evenly across the source, but the SAME 200")
    print("rows are picked every single time the script runs.")
    print()
    print("This kind of randomness is called PSEUDO-RANDOM: it looks random")
    print("but it's actually fully determined by the seed.")
    print()
    print("Sampling...")

    sampled_parts = []
    for source in sorted(sources.unique()):
        source_rows = df[sources == source]
        if len(source_rows) < SAMPLE_PER_SOURCE:
            raise ValueError(
                f"{source} only has {len(source_rows)} rows, "
                f"not enough for {SAMPLE_PER_SOURCE}."
            )
        sample = source_rows.sample(n=SAMPLE_PER_SOURCE, random_state=RANDOM_SEED)
        sampled_parts.append(sample)
        print(f"  {source:<20} {len(sample)} rows sampled")

    combined = pd.concat(sampled_parts, ignore_index=True)
    print(f"\n  Concatenated together: {len(combined)} rows total.")
    print()
    print("  pd.concat() stacks DataFrames on top of each other. The")
    print("  ignore_index=True throws away the old row numbers and gives")
    print("  us fresh ones from 0 to 599.")
    pause()

    # ----------------------------------------------------------
    step(4, "Shuffle the combined sample")
    print("There's still a problem: the 600 rows are GROUPED by source.")
    print("The first 200 are all from one source, the next 200 from")
    print("another, and so on. If somebody used this for a quiz, they'd")
    print("hit 200 commonsense questions in a row, then 200 logic")
    print("questions, then 200 trivia — which could distort their")
    print("performance through fatigue or expectation.")
    print()
    print("To fix it, we SHUFFLE the rows into a random order.")
    print()
    print("Here's a neat pandas trick. The .sample() method lets you")
    print("randomly pick a fraction of a DataFrame. If you ask for")
    print("frac=0.1, you get a random 10%. If you ask for FRAC=1.0, you")
    print("get 100% of the rows — every single one — in a RANDOM ORDER.")
    print("That's exactly what shuffling means. So:")
    print()
    print("    shuffled = combined.sample(frac=1, random_state=42)")
    print()
    print("...is just pandas shorthand for 'shuffle this DataFrame'. We")
    print("use the same random seed so the shuffle is also reproducible.")
    print()
    print("Then reset_index(drop=True) gives the shuffled rows fresh")
    print("row numbers 0 through 599 (otherwise they'd keep their original")
    print("indexes, which would now be out of order and confusing).")

    shuffled = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n  First 8 IDs after shuffling:")
    for sid in shuffled["ID"].head(8).tolist():
        print(f"    {sid}")
    print()
    print("  Notice the sources are now mixed up — they appear in")
    print("  random order rather than in groups.")
    pause()

    # ----------------------------------------------------------
    step(5, "Save the final sample")
    print("Last step: write the shuffled DataFrame to a CSV file. That's")
    print("the file you'll actually use for whatever's downstream — a")
    print("quiz, an evaluation, an experiment.")

    shuffled.to_csv(OUTPUT_FILE, index=False)

    print(f"\n  Wrote {len(shuffled)} rows to {OUTPUT_FILE}")
    print()
    print("  Final source counts (should each be exactly 200):")
    final_sources = shuffled["ID"].str.split("-").str[0]
    for name, count in final_sources.value_counts().sort_index().items():
        print(f"    {name:<20} {count:>6}")
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print()
    print("Recap of what just happened across all three stages:")
    print()
    print("  Stage 1 (EXTRACT)    Three messy raw formats")
    print("                       → three CSVs in Dirty/")
    print()
    print("  Stage 2 (TRANSFORM)  Three different CSV shapes")
    print("                       → one unified CSV in Cleaned/")
    print()
    print("  Stage 3 (SAMPLE)     One big imbalanced dataset")
    print("                       → one small balanced shuffled sample")
    print()
    print("This three-stage flow — extract, transform, sample (or load)")
    print("— is the backbone of nearly every real-world data project.")
    print("You'll see the same shape, sometimes under different names,")
    print("in everything from machine learning pipelines to business")
    print("analytics to scientific research.")


if __name__ == "__main__":
    main()