"""
Stage 1 of 3 — EXTRACT
Reads:  Questions/Source/<dataset>/...
Writes: Questions/Dirty/<dataset>.csv

CommonsenseQA already comes as a CSV — just copy it.
LogiQA comes as a custom text format — parse it.
OpenTriviaQA comes as many text files in the #Q format — parse them.

The Dirty/ CSVs are intermediate. Stage 2 will normalize them.
"""

from pathlib import Path
import shutil
import pandas as pd


SOURCE_DIR = Path("Questions/Source")
DIRTY_DIR = Path("Questions/Dirty")
DIRTY_DIR.mkdir(parents=True, exist_ok=True)


def pause(message="Press Enter to continue..."):
    input(f"\n  >>> {message}\n")


def step(number, title):
    print()
    print("=" * 60)
    print(f"STEP {number}: {title}")
    print("=" * 60)


# === LogiQA parser ============================================
# Each LogiQA question is a 7-line block, separated by blank lines:
#   line 1: correct answer letter (lowercase: a, b, c, or d)
#   line 2: context paragraph
#   line 3: question
#   lines 4-7: options prefixed with "A.", "B.", "C.", "D."
#
# We combine context + question into one "Questions" field, so
# the student reading the final row sees the full prompt at once.

def parse_logiqa_block(lines):
    """Turn a 7-line block into a question dict. Returns None if malformed."""
    if len(lines) != 7:
        return None

    correct = lines[0].strip().upper()
    context = lines[1].strip()
    question = lines[2].strip()
    # Options look like "A.text" — drop the first 2 characters.
    options = [line.strip()[2:].strip() for line in lines[3:7]]

    return {
        "Question": f"{context} {question}",
        "Correct": correct,
        "Options": options,
    }


def parse_logiqa_file(path):
    """Read a LogiQA text file and return a list of question dicts."""
    questions = []
    current_block = []

    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                current_block.append(line)
            else:
                parsed = parse_logiqa_block(current_block)
                if parsed:
                    questions.append(parsed)
                current_block = []

    # Don't forget the last block (no trailing blank line).
    parsed = parse_logiqa_block(current_block)
    if parsed:
        questions.append(parsed)

    return questions


# === OpenTriviaQA parser ======================================
# Each OpenTriviaQA question is a block in the "#Q format":
#   "#Q ..."  - the question (may span MULTIPLE lines)
#   "^ ..."   - the correct answer text (NOT a letter)
#   "A ..."   - first option
#   "B ..."   - second option
#   ... and so on (usually up to D; True/False has only A and B)
#
# Lines that don't start with one of those markers are CONTINUATIONS
# of whatever section we last saw — that's how questions can span
# multiple lines. We track the current section in `state` and append
# unmarked lines there.
#
# The "^" text always reappears as one of the lettered options.
# Stage 2 will look it up to find the matching letter.

def parse_trivia_block(lines):
    """Parse a #Q-format block. Returns None if not a valid question."""
    question_parts = []
    correct_parts = []
    option_parts = {}   # letter -> list of text fragments
    state = None        # "question", "correct", or "A"/"B"/"C"/"D"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#Q "):
            question_parts.append(stripped[3:].strip())
            state = "question"
        elif stripped.startswith("^ "):
            correct_parts.append(stripped[2:].strip())
            state = "correct"
        elif len(stripped) >= 2 and stripped[0] in "ABCD" and stripped[1] == " ":
            # Option marker — only A/B/C/D, so a question line that
            # happens to start with "I " or "E " won't be confused.
            letter = stripped[0]
            option_parts[letter] = [stripped[2:].strip()]
            state = letter
        elif state == "question":
            question_parts.append(stripped)
        elif state == "correct":
            correct_parts.append(stripped)
        elif state in option_parts:
            option_parts[state].append(stripped)

    # Join the pieces of each multi-line section with a space.
    question = " ".join(question_parts).strip()
    correct = " ".join(correct_parts).strip()
    options = [
        " ".join(option_parts[letter]).strip()
        for letter in "ABCD" if letter in option_parts
    ]

    if not question or not correct or not options:
        return None
    # Some files use "#Q #..." for section headers; skip them.
    if question.startswith("#"):
        return None

    return {
        "Question": question,
        "Correct": correct,
        "Options": options,
    }


def parse_trivia_file(path):
    """Read one OpenTriviaQA category file."""
    questions = []
    current_block = []

    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                current_block.append(line)
            else:
                parsed = parse_trivia_block(current_block)
                if parsed:
                    questions.append(parsed)
                current_block = []

    parsed = parse_trivia_block(current_block)
    if parsed:
        questions.append(parsed)

    return questions


# === The main pipeline ========================================
def main():
    print()
    print("STAGE 1: EXTRACT")
    print()
    print("This is the first stage of a three-stage data pipeline. Every")
    print("data project starts with the same question: 'what raw materials")
    print("do we have, and how do we get them into a form we can work with?'")
    print()
    print("Our raw materials are three different question datasets, each")
    print("downloaded in its own format:")
    print("  - CommonsenseQA arrives as a CSV file (already tabular)")
    print("  - LogiQA arrives as a custom plain-text format")
    print("  - OpenTriviaQA arrives as many text files in YET ANOTHER format")
    print()
    print("Stage 1's only job is to land all three in the same KIND of file")
    print("(a CSV) in the Dirty/ folder. We don't yet try to make the columns")
    print("match — each source can keep its own shape for now. Unifying the")
    print("shapes is Stage 2's job.")
    print()
    print("The folder is called 'Dirty' because the data inside is workable")
    print("but not yet cleaned. Think of it like a workshop: raw materials")
    print("have arrived and been sorted, but nothing is finished yet.")
    pause("Press Enter to start...")

    # ----------------------------------------------------------
    step(1, "Copy CommonsenseQA")
    print("Of our three sources, CommonsenseQA is the easiest to handle —")
    print("it arrives as a CSV file already. CSV stands for 'comma-separated")
    print("values': a plain text file where the first line names the columns,")
    print("and every line after that is one row with values separated by")
    print("commas. It's a universal tabular format that Excel, Google Sheets,")
    print("and most programming languages can open directly.")
    print()
    print("Since CommonsenseQA is already a CSV, we don't have to write a")
    print("parser. We just COPY the file from Source/ into Dirty/ using")
    print("shutil.copy — a built-in Python function for copying files. The")
    print("copy lands in Dirty/ with the same name, ready for Stage 2 to read.")

    src = SOURCE_DIR / "CommonsenseQA" / "CommonsenseQA.csv"
    dst = DIRTY_DIR / "CommonsenseQA.csv"
    shutil.copy(src, dst)

    df = pd.read_csv(dst)
    print(f"\n  Copied {len(df)} rows to {dst}")
    print(f"  Columns in the file: {list(df.columns)}")
    print()
    print("  These three columns are CommonsenseQA's native shape. The")
    print("  'choices' column is suspicious-looking — we'll see in Stage 2")
    print("  that it's the messiest field in the whole pipeline.")
    pause()

    # ----------------------------------------------------------
    step(2, "Peek at LogiQA's text format")
    print("LogiQA does NOT arrive as a CSV — it's a plain text file with")
    print("its own custom layout. Before we write a parser, we always want")
    print("to look at the raw file with our own eyes and figure out what")
    print("the format actually is. So let's PEEK at the first few lines.")
    print()
    print("LogiQA organizes its questions into BLOCKS of 7 lines each:")
    print("    line 1: correct answer letter (lowercase: a, b, c, or d)")
    print("    line 2: a context paragraph that sets up the question")
    print("    line 3: the actual question")
    print("    lines 4-7: four options, each starting with 'A.', 'B.',")
    print("               'C.', 'D.' (letter + dot, then the option text)")
    print()
    print("Each block is separated from the next by a single BLANK LINE.")
    print("That blank line is the signal we'll use to know when one")
    print("question ends and the next begins.")
    print()

    src = SOURCE_DIR / "LogiQA" / "LogiQA.txt"
    print(f"  First 8 lines of {src.name}:")
    print("  " + "-" * 50)
    with src.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 8:
                break
            print(f"  | {line.rstrip()}")
    print("  " + "-" * 50)
    print()
    print("  See the pattern? Letter, context, question, four options.")
    print("  Then a blank line (line 8 above) and the next block starts.")
    pause()

    # ----------------------------------------------------------
    step(3, "Parse LogiQA into a CSV")
    print("PARSING means reading the raw text and pulling out the meaningful")
    print("pieces — converting 'a string of characters' into 'structured")
    print("data we can work with.' Our parser works like this:")
    print()
    print("  1. Walk through the file one line at a time.")
    print("  2. Collect lines into a 'current block'.")
    print("  3. When we hit a blank line, the block is done: extract the")
    print("     question's parts from it, save them, and start a fresh block.")
    print("  4. Repeat until the file ends.")
    print()
    print("For each block we keep:")
    print("  - The correct LETTER, uppercased ('c' becomes 'C') so that")
    print("    later we can compare it cleanly with letters from other")
    print("    sources, which all use uppercase.")
    print("  - The context AND question combined into a single 'Questions'")
    print("    field, joined with a space. We combine them because the")
    print("    context is usually necessary to even understand the question")
    print("    — they aren't useful separately.")
    print("  - The four option texts, with their 'A.', 'B.', 'C.', 'D.'")
    print("    prefixes stripped off. We don't need the letters in the text")
    print("    because the COLUMN they go in (A, B, C, D) already tells us.")
    print()
    print("Parsing the file...")

    src = SOURCE_DIR / "LogiQA" / "LogiQA.txt"
    parsed = parse_logiqa_file(src)

    # Spread each row's options into columns A, B, C, D.
    rows = []
    for q in parsed:
        row = {"Questions": q["Question"], "Correct": q["Correct"]}
        for i, opt in enumerate(q["Options"]):
            row[chr(ord("A") + i)] = opt
        rows.append(row)

    df = pd.DataFrame(rows, columns=["Questions", "Correct", "A", "B", "C", "D"])
    dst = DIRTY_DIR / "LogiQA.csv"
    df.to_csv(dst, index=False)

    print(f"\n  Parsed {len(parsed)} questions out of the file.")
    print(f"  Saved to {dst}")
    print(f"  The new CSV has columns: {list(df.columns)}")
    print()
    print("  Here's what one parsed row looks like, in our new tidy shape:")
    sample = df.iloc[0]
    print(f"    Questions: {str(sample['Questions'])[:80]}...")
    print(f"    Correct:   {sample['Correct']}")
    print(f"    A:         {sample['A']}")
    print(f"    B:         {sample['B']}")
    print()
    print("  We've gone from messy 7-line blocks to one clean row per question.")
    pause()

    # ----------------------------------------------------------
    step(4, "Peek at OpenTriviaQA's text format")
    print("OpenTriviaQA is our third source, and it brings TWO new challenges:")
    print()
    print("  1. It's not one file — it's MANY files, one per category")
    print("     (animals, movies, history, geography, and so on). The")
    print("     category isn't written inside the file; it IS the filename.")
    print()
    print("  2. It uses yet a different text format, called the '#Q format'.")
    print()
    print("Here's how a single question looks:")
    print("    #Q the question text")
    print("    ^ the correct answer text (NOT a letter — the actual answer)")
    print("    A first option")
    print("    B second option")
    print("    C third option")
    print("    D fourth option")
    print()
    print("Each line starts with a MARKER: '#Q', '^', 'A', 'B', 'C', or 'D'.")
    print("The marker tells us what that line is. Blocks are separated by")
    print("a blank line, like LogiQA.")
    print()
    print("Two tricky things to notice:")
    print()
    print("  - The '^' line gives the correct answer as TEXT, not as a")
    print("    letter. The same text always reappears as one of A/B/C/D.")
    print("    So 'which letter is correct?' is something Stage 2 has to")
    print("    figure out by matching.")
    print()
    print("  - Questions sometimes span MULTIPLE lines. A long question")
    print("    might be split across 2 or 3 lines, with no marker on the")
    print("    continuation lines. Our parser has to recognize those as")
    print("    continuations of the question, not new options.")
    print()

    trivia_dir = SOURCE_DIR / "OpenTriviaQA" / "categories"
    # Category files have no extension — list everything, skip hidden.
    text_files = sorted(
        p for p in trivia_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    print(f"  Found {len(text_files)} category files in {trivia_dir}")
    print(f"  First few: {[f.name for f in text_files[:5]]}")
    print()
    print("  Each of those filenames will become the category for every")
    print("  question inside that file.")
    pause()

    # ----------------------------------------------------------
    step(5, "Parse all OpenTriviaQA files")
    print("Our trivia parser is a bit fancier than the LogiQA one because")
    print("it has to handle multi-line questions. It works like a STATE")
    print("MACHINE — at any moment it remembers what kind of line it's")
    print("expecting next:")
    print()
    print("  - See '#Q something'   → start collecting a QUESTION")
    print("  - See '^ something'    → start collecting the CORRECT answer")
    print("  - See 'A something'    → start collecting OPTION A")
    print("  - See 'B/C/D something'→ same idea for B, C, D")
    print("  - See an UNMARKED line → it's a continuation of whatever we")
    print("                           were last collecting, so append it")
    print()
    print("That last rule is the key one: it's how a question split across")
    print("three lines gets correctly recognized as one question.")
    print()
    print("We restrict option markers to JUST A/B/C/D (not any letter) so")
    print("that a question line beginning with 'I am...' or 'E equals mc^2'")
    print("doesn't get mistaken for an option.")
    print()
    print("After parsing each file, we tag every question in it with the")
    print("filename as its 'Category'. Then we combine all files into one")
    print("big CSV.")
    print()
    print("Parsing files...")
    print()

    all_questions = []
    for tf in text_files:
        parsed = parse_trivia_file(tf)
        for q in parsed:
            q["Category"] = tf.stem
        all_questions.extend(parsed)
        print(f"  {tf.name:<35} {len(parsed):>5} questions")

    # Flatten — most rows have 4 options, T/F rows have 2.
    max_options = max(len(q["Options"]) for q in all_questions)
    rows = []
    for q in all_questions:
        row = {"Questions": q["Question"], "Correct": q["Correct"]}
        for i, opt in enumerate(q["Options"]):
            row[chr(ord("A") + i)] = opt
        row["category"] = q["Category"]
        rows.append(row)

    cols = (
        ["Questions", "Correct"]
        + [chr(ord("A") + i) for i in range(max_options)]
        + ["category"]
    )
    df = pd.DataFrame(rows, columns=cols)
    dst = DIRTY_DIR / "OpenTriviaQA.csv"
    df.to_csv(dst, index=False)

    print(f"\n  Combined total: {len(all_questions)} questions across")
    print(f"  {len(text_files)} category files.")
    print(f"  Saved to {dst}")
    print(f"  Columns: {list(df.columns)}")
    print()
    print("  Notice: the 'Correct' column here holds the answer TEXT,")
    print("  not a letter. That'll need fixing in Stage 2. The 'category'")
    print("  column is the filename (e.g. 'animals', 'video-games').")
    pause()

    # ----------------------------------------------------------
    step(6, "Stage 1 complete — three CSVs in Dirty/")
    print("All three sources are now CSV files sitting side by side in")
    print("Dirty/. They share the same FILE FORMAT (CSV), but their")
    print("COLUMNS are still very different from each other:")
    print()
    for csv_file in sorted(DIRTY_DIR.glob("*.csv")):
        df = pd.read_csv(csv_file)
        print(f"  {csv_file.name:<25} {len(df):>6} rows")
        print(f"    columns: {list(df.columns)}")
    print()
    print("  CommonsenseQA still uses 'question' and 'answerKey' and a")
    print("  bizarre 'choices' field. LogiQA and OpenTriviaQA use")
    print("  'Questions' and 'Correct' but disagree on whether 'Correct'")
    print("  is a letter or a text. And only OpenTriviaQA has a 'category'.")
    print()
    print("  Stage 2's job is to take these three shapes and HAMMER them")
    print("  into one shared schema, so they can live in one dataset.")


if __name__ == "__main__":
    main()