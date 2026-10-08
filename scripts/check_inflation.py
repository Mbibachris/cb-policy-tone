"""Check the inflation numbers read from the statements.

Run from the project folder:
  python scripts/check_inflation.py              one line per meeting in YEARS, with its sentence
  python scripts/check_inflation.py 37 38 118    EVERY inflation sentence in those meetings

It only reads the statements and the inflation file. It does not touch any labels.
"""

import re
import sys
from pathlib import Path

import pandas as pd

from cb_policy_tone.sentences import split_statement

YEARS = [2009, 2010, 2024, 2025]
PERCENT = re.compile(r"per\s?cent", re.IGNORECASE)

inflation = pd.read_csv("data/processed/inflation_from_statements.csv")
statements = pd.read_csv("data/processed/statements.csv")
table = inflation.merge(statements[["doc_id", "txt_file", "meeting_end"]], on="doc_id")
table["date"] = pd.to_datetime(table["meeting_end"])
table = table.dropna(subset=["meeting_no"])
table["meeting_no"] = table["meeting_no"].astype(int)

wanted = [int(a) for a in sys.argv[1:]]
if wanted:
    table = table[table["meeting_no"].isin(wanted)]
else:
    table = table[table["date"].dt.year.isin(YEARS)]

for row in table.sort_values("date").itertuples():
    print("=" * 70)
    print(
        f"Meeting {row.meeting_no}  |  {row.date:%d %b %Y}  |  value read: {row.infl}"
    )
    sentences = [
        r["text"]
        for r in split_statement(Path(row.txt_file).read_text(encoding="utf-8"))
    ]
    if wanted:
        for sentence in sentences:
            if "inflation" in sentence.lower() and PERCENT.search(sentence):
                print("   -", sentence)
        continue
    if pd.isna(row.infl):
        continue
    forms = {f"{row.infl:g}", f"{row.infl:.1f}", f"{row.infl:.2f}"}
    for sentence in sentences:
        if any(form in sentence for form in forms):
            print("   SENTENCE:", sentence)
            break
