"""
Stage 2 of 3 — TRANSFORM
Reads:  Questions/Dirty/*.csv   (three different shapes)
Writes: Questions/Cleaned/UnifiedQA.csv   (one shared shape)

Each source's loader knows the quirks of its CSV and returns rows
in the same shape:

    {
        "ID":       <source>-<row number>
        "Question": question text
        "Correct":  letter (A, B, C, ...)
        "Category": category string or None
        "answers":  list of answer texts
    }

Once every row is in that shape, building the final CSV is easy.
"""

import ast
import re
from pathlib import Path

import pandas as pd


DIRTY_DIR = Path("Questions/Dirty")
CLEANED_DIR = Path("Questions/Cleaned")
OUTPUT_FILE = CLEANED_DIR / "UnifiedQA.csv"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


def pause(message="Press Enter to continue..."):
    input(f"\n  >>> {message}\n")


def step(number, title):
    print()
    print("=" * 60)
    print(f"STEP {number}: {title}")
    print("=" * 60)


def clean_text(value):
    """Tidy a cell: collapse whitespace, strip, return None if missing."""
    if pd.isna(value):
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def letter_for_position(position):
    """1 -> 'A', 2 -> 'B', ..., 26 -> 'Z', 27 -> 'AA', ..."""
    result = ""
    while position > 0:
        position, remainder = divmod(position - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


# === CommonsenseQA's 'choices' field is messy =================
# It's a string that LOOKS like a Python dict containing two
# numpy arrays:
#   {'label': array(['A','B','C','D','E'], dtype=object),
#    'text':  array(['ignore','enforce',...], dtype=object)}
# We pull each list out with regex, then turn the matched text
# into a real Python list with ast.literal_eval — like eval(),
# but SAFE: it only evaluates literals, never runs code.

def parse_commonsense_choices(raw):
    """Turn the messy choices string into a {letter: text} dict."""
    labels_match = re.search(
        r"'label':\s*array\((\[.*?\])\s*,\s*dtype=object\)", str(raw), re.S
    )
    texts_match = re.search(
        r"'text':\s*array\((\[.*?\])\s*,\s*dtype=object\)", str(raw), re.S
    )
    if not labels_match or not texts_match:
        raise ValueError(f"Couldn't parse choices field: {raw}")

    labels = ast.literal_eval(labels_match.group(1))
    texts = ast.literal_eval(texts_match.group(1))
    return dict(zip(labels, texts))


# === Loaders — one per source =================================

def load_commonsense(csv_path):
    """CommonsenseQA: unpack 'choices', use 'answerKey' as Correct."""
    df = pd.read_csv(csv_path)
    rows = []

    for i, row in df.iterrows():
        choice_map = parse_commonsense_choices(row["choices"])
        ordered_letters = sorted(choice_map.keys())
        answers = [clean_text(choice_map[letter]) for letter in ordered_letters]

        rows.append({
            "ID":       f"{csv_path.stem}-{i + 1:03d}",
            "Question": clean_text(row["question"]),
            "Correct":  clean_text(row["answerKey"]),
            "Category": None,
            "answers":  answers,
        })

    return rows


def load_logiqa(csv_path):
    """LogiQA: Correct is already a letter; everything else copies through."""
    df = pd.read_csv(csv_path)
    answer_cols = [c for c in df.columns if c not in {"Questions", "Correct"}]
    rows = []

    for i, row in df.iterrows():
        answers = [clean_text(row[col]) for col in answer_cols]

        rows.append({
            "ID":       f"{csv_path.stem}-{i + 1:03d}",
            "Question": clean_text(row["Questions"]),
            "Correct":  clean_text(row["Correct"]),
            "Category": None,
            "answers":  answers,
        })

    return rows


def load_opentrivia(csv_path):
    """OpenTriviaQA: Correct is TEXT — look up which option letter matches."""
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    answer_cols = [c for c in ["A", "B", "C", "D"] if c in df.columns]
    rows = []
    skipped = 0

    for i, row in df.iterrows():
        question = clean_text(row.get("Questions"))
        correct_text = clean_text(row.get("Correct"))
        answers = [clean_text(row.get(col)) for col in answer_cols]

        # Skip rows missing essentials, or where any option is empty
        # (True/False rows have empty C and D, so they fall out here).
        if not question or not correct_text or any(a is None for a in answers):
            skipped += 1
            continue

        # Find which option's text matches the correct text.
        correct_letter = None
        for pos, ans in enumerate(answers, start=1):
            if ans == correct_text:
                correct_letter = letter_for_position(pos)
                break

        if correct_letter is None:
            skipped += 1
            continue

        rows.append({
            "ID":       f"{csv_path.stem}-{i + 1:03d}",
            "Question": question,
            "Correct":  correct_letter,
            "Category": clean_text(row.get("category")),
            "answers":  answers,
        })

    return rows, skipped


# === The main pipeline ========================================
def main():
    print()
    print("STAGE 2: TRANSFORM")
    print()
    print("Stage 1 dumped three CSV files into Dirty/. Each one is tidy")
    print("internally — but they don't agree with each other on column")
    print("names, on what 'Correct' means, or on which fields exist.")
    print("Trying to use them together right now would be a mess.")
    print()
    print("This stage NORMALIZES them: every row from every source gets")
    print("rewritten into the SAME shape (the same set of columns, the")
    print("same meaning for each one). The output is one big CSV that")
    print("treats all three sources as interchangeable.")
    print()
    print("The shared shape we're aiming for is:")
    print("    ID          - a unique label, like 'CommonsenseQA-001'")
    print("    Question    - the question text")
    print("    Correct     - the LETTER of the right answer (A, B, C, ...)")
    print("    A1..A5      - the answer texts, one per column")
    print("    Category    - a topic label (only OpenTriviaQA has these)")
    print()
    print("We use three LOADER functions, one per source. Each loader")
    print("knows the quirks of its source CSV, and they all return rows")
    print("in the same shape — like translators that all speak different")
    print("source languages but produce English.")
    pause("Press Enter to start...")

    # ----------------------------------------------------------
    step(1, "Inspect what we're working with")
    print("Before transforming anything, it's worth looking at the three")
    print("Dirty CSVs side by side. Each one has its own SCHEMA — the set")
    print("of column names and what they mean. Mismatched schemas are")
    print("exactly why we need a transform stage at all.")
    print()
    print("Look at how different they are:")
    for source in ["CommonsenseQA", "LogiQA", "OpenTriviaQA"]:
        df = pd.read_csv(DIRTY_DIR / f"{source}.csv")
        print(f"\n  {source}  ({len(df)} rows)")
        print(f"    columns: {list(df.columns)}")
    print()
    print("  Three sets of columns, three different conventions. Stage 2")
    print("  has to work with each one on its own terms before it can")
    print("  merge them.")
    pause()

    # ----------------------------------------------------------
    step(2, "Load CommonsenseQA")
    print("CommonsenseQA is the trickiest of the three because it packs")
    print("all its answers into ONE column called 'choices'. That column")
    print("isn't simple text — each cell is a string that LOOKS like a")
    print("Python dictionary, with two lists hidden inside it.")
    print()
    print("Here's what the raw 'choices' field looks like for row 1:")
    print()

    cs_csv = DIRTY_DIR / "CommonsenseQA.csv"
    raw_choices = pd.read_csv(cs_csv).iloc[0]["choices"]
    preview = raw_choices[:200] + ("..." if len(raw_choices) > 200 else "")
    print(f"    {preview}")
    print()
    print("Notice the two lists inside the braces:")
    print("  - 'label' holds the answer letters: ['A','B','C','D','E']")
    print("  - 'text' holds the matching answer texts")
    print("Position 0 in 'label' goes with position 0 in 'text', and so on.")
    print()
    print("Our problem is: this is a STRING, not a real Python dict. We")
    print("can't just access list1[0] — Python sees the whole thing as")
    print("one big piece of text. We have to dig the lists back out.")
    print()
    print("We use two tools to do that:")
    print()
    print("  REGULAR EXPRESSIONS (regex): patterns for finding text that")
    print("  matches a description. We write two regexes — one that finds")
    print("  the 'label' list inside the cell, and one that finds the")
    print("  'text' list. Python's `re.search()` returns the matching")
    print("  substring (like '[A,B,C,D,E]').")
    print()
    print("  ast.literal_eval: a function that takes a string like \"[1,2,3]\"")
    print("  and turns it into a REAL Python list [1, 2, 3]. Without it,")
    print("  we'd just have a string of characters that LOOKS like a list.")
    print()
    print("  You might wonder: why not just use Python's eval() function?")
    print("  Because eval() will run ANY code you hand it — if someone")
    print("  snuck malicious code into the data file, eval() would execute")
    print("  it. ast.literal_eval is the SAFE version: it only accepts")
    print("  literal values (numbers, strings, lists, dicts, tuples,")
    print("  booleans, None), never code. Always reach for literal_eval")
    print("  when you're parsing data from outside sources.")
    print()
    print("Loading...")

    cs_rows = load_commonsense(cs_csv)
    print(f"\n  Loaded {len(cs_rows)} rows.")
    print()
    print("  Here's the first row, now in the shared unified shape:")
    for k, v in cs_rows[0].items():
        print(f"    {k}: {v}")
    print()
    print("  The 'choices' string has become a real list of answer texts,")
    print("  and 'answerKey' is now in the 'Correct' field as a clean letter.")
    pause()

    # ----------------------------------------------------------
    step(3, "Load LogiQA")
    print("After CommonsenseQA, LogiQA feels almost too easy. Look back")
    print("at its dirty CSV:")
    print()
    print("    Questions, Correct, A, B, C, D")
    print()
    print("It already has answers in separate columns. The 'Correct'")
    print("column already holds a LETTER (A/B/C/D), exactly what we want.")
    print("So our loader has almost nothing to do — just copy each row's")
    print("fields straight into the unified shape.")
    print()
    print("The only flexible bit: we don't HARDCODE that the answer")
    print("columns are A/B/C/D. Instead we do this:")
    print()
    print("    answer_cols = every column that isn't 'Questions' or 'Correct'")
    print()
    print("That way, if a future LogiQA-like source has 3 answers or 5,")
    print("the same loader still works without code changes.")
    print()
    print("Loading...")

    lq_rows = load_logiqa(DIRTY_DIR / "LogiQA.csv")

    print(f"\n  Loaded {len(lq_rows)} rows.")
    print()
    print("  Here's the first row in unified shape:")
    for k, v in lq_rows[0].items():
        print(f"    {k}: {v}")
    print()
    print("  Notice 'Category' is None — LogiQA questions aren't grouped")
    print("  by topic. The 'answers' list has 4 items because LogiQA")
    print("  always has 4 options.")
    pause()

    # ----------------------------------------------------------
    step(4, "Load OpenTriviaQA")
    print("OpenTriviaQA is the most awkward of the three. Look at its")
    print("dirty CSV columns:")
    print()
    print("    Questions, Correct, A, B, C, D, category")
    print()
    print("The trouble is that 'Correct' holds the answer TEXT, not a")
    print("letter. For example:")
    print()
    print("    Questions: Three of these animals hibernate. Which doesn't?")
    print("    Correct:   Sloth          <-- text, not a letter!")
    print("    A:         Mouse")
    print("    B:         Sloth          <-- the right answer is here")
    print("    C:         Frog")
    print("    D:         Snake")
    print()
    print("Our unified shape needs 'Correct' to be a letter. So this")
    print("loader has to do a LOOKUP: walk through A, B, C, D and find")
    print("which option's text matches the 'Correct' text. In the example")
    print("above, B's text equals the Correct text, so Correct becomes 'B'.")
    print()
    print("Some rows have to be skipped:")
    print("  - True/False questions only have A and B filled in (C and D")
    print("    are empty). We treat those as malformed and drop them, to")
    print("    keep the dataset consistent.")
    print("  - Occasionally the 'Correct' text doesn't exactly match any")
    print("    option (typo, punctuation difference). Those get dropped too.")
    print()
    print("Loading...")

    ot_rows, ot_skipped = load_opentrivia(DIRTY_DIR / "OpenTriviaQA.csv")

    print(f"\n  Loaded {len(ot_rows)} rows.")
    print(f"  Skipped {ot_skipped} rows (True/False or no match).")
    print()
    print("  Here's the first kept row in unified shape:")
    for k, v in ot_rows[0].items():
        print(f"    {k}: {v}")
    print()
    print("  See how 'Correct' is now a letter, and 'Category' is filled.")
    pause()

    # ----------------------------------------------------------
    step(5, "Combine all three sources")
    print("Now the magic of doing the hard work earlier: because every")
    print("loader returned rows in the same shape, COMBINING them is just")
    print("list addition. Python's `+` operator concatenates lists:")
    print()
    print("    all_rows = cs_rows + lq_rows + ot_rows")
    print()
    print("After that one line, we have one big list of dicts where every")
    print("dict has the same keys, regardless of which source it came from.")

    all_rows = cs_rows + lq_rows + ot_rows
    max_answers = max(len(r["answers"]) for r in all_rows)

    print()
    print(f"  CommonsenseQA contributed: {len(cs_rows)}")
    print(f"  LogiQA contributed:        {len(lq_rows)}")
    print(f"  OpenTriviaQA contributed:  {len(ot_rows)}")
    print(f"  ---------------------------")
    print(f"  Total rows now combined:   {len(all_rows)}")
    print()
    print(f"  Widest answer set: {max_answers} options.")
    print()
    print("  CommonsenseQA's questions have 5 options (A-E); the other")
    print("  two sources have 4. The CSV needs enough columns for the")
    print(f"  widest set, so it'll have A1 through A{max_answers}. Rows from")
    print("  the smaller sources will leave the last column blank.")
    pause()

    # ----------------------------------------------------------
    step(6, "Flatten answers into columns and save")
    print("There's one last shape mismatch to fix. Right now each row in")
    print("memory looks like this:")
    print()
    print("    { 'ID': ..., 'Question': ..., 'Correct': ..., 'Category': ...,")
    print("      'answers': ['option1', 'option2', 'option3', 'option4'] }")
    print()
    print("A CSV can't store a LIST inside a single cell — each column has")
    print("to be a flat value (one string, one number). So we FLATTEN the")
    print("answers list: spread its items into separate columns named A1,")
    print(f"A2, ..., A{max_answers}. Each row keeps the same shape; rows with fewer")
    print("answers leave the extra columns empty.")
    print()
    print("This is the final move that turns our in-memory data into a")
    print("clean tabular file that any spreadsheet program can read.")

    output_rows = []
    for r in all_rows:
        flat = {
            "ID":       r["ID"],
            "Question": r["Question"],
            "Correct":  r["Correct"],
        }
        for i in range(max_answers):
            flat[f"A{i + 1}"] = r["answers"][i] if i < len(r["answers"]) else None
        flat["Category"] = r["Category"]
        output_rows.append(flat)

    columns = (
        ["ID", "Question", "Correct"]
        + [f"A{i + 1}" for i in range(max_answers)]
        + ["Category"]
    )
    out_df = pd.DataFrame(output_rows, columns=columns)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n  Wrote {len(out_df)} rows to {OUTPUT_FILE}")
    print()
    print("  One sample row from each source — notice they all match shape:")
    for source in ["CommonsenseQA", "LogiQA", "OpenTriviaQA"]:
        matching = out_df[out_df["ID"].str.startswith(source)]
        if len(matching):
            r = matching.iloc[0]
            print(f"\n    [{r['ID']}]")
            print(f"      Question: {str(r['Question'])[:70]}...")
            print(f"      Correct:  {r['Correct']}")
            print(f"      Category: {r['Category']}")
    print()
    print("Stage 2 complete. Three different shapes are now one. The")
    print("unified file is ready for Stage 3, which will sample it down")
    print("to a balanced, manageable size.")


if __name__ == "__main__":
    main()