"""Clean the Bank of Ghana 'Historical Policy Rate Decisions' table.

Input : data/external/bog_policy_rate_decisions.csv  (downloaded from the BoG website)
        data/external/rate_corrections.csv           (fixes to the table, each with evidence)
        data/external/meetings_added.csv             (meetings missing from the table)
Output: data/processed/policy_rates.csv              (one row per policy decision)

The Bank's own table is never edited. Every change is listed in the two correction files,
with the statement that justifies it, and the output marks corrected rows.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

RAW = Path("data/external/bog_policy_rate_decisions.csv")
CORRECTIONS = Path("data/external/rate_corrections.csv")
ADDITIONS = Path("data/external/meetings_added.csv")
OUT = Path("data/processed/policy_rates.csv")

MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"],
        start=1,
    )
}  # fmt: skip

RANGE_RE = re.compile(
    r"^(?P<m1>[A-Za-z]+)\s+(?P<d1>\d{1,2})\s*[–-]\s*(?:(?P<m2>[A-Za-z]+)\s+)?"
    r"(?P<d2>\d{1,2}),\s*(?P<y>\d{4})"
)
SINGLE_RE = re.compile(r"^(?P<m1>[A-Za-z]+)\s+(?P<d1>\d{1,2}),\s*(?P<y>\d{4})")
COERCE = {
    "mpr": float,
    "meeting_no": int,
    "meeting_type": str,
    "meeting_start": pd.Timestamp,
    "meeting_end": pd.Timestamp,
    "effective_date": pd.Timestamp,
}

def parse_meeting_dates(text: str) -> tuple[date, date, bool]:
    """'August 29 – September 1, 2011' -> (start, end, is_emergency)."""
    emergency = "emergency" in text.lower()
    m = RANGE_RE.match(text.strip())
    if m:
        y = int(m["y"])
        m1 = MONTHS[m["m1"].lower()]
        m2 = MONTHS[m["m2"].lower()] if m["m2"] else m1
        return date(y, m1, int(m["d1"])), date(y, m2, int(m["d2"])), emergency
    m = SINGLE_RE.match(text.strip())
    if m:
        d = date(int(m["y"]), MONTHS[m["m1"].lower()], int(m["d1"]))
        return d, d, emergency
    raise ValueError(f"Cannot parse meeting dates: {text!r}")


def apply_corrections(df: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Apply each correction in file order. The 'old_value' must match, or we stop."""
    df = df.copy()
    for c in corrections.itertuples():
        convert = COERCE[c.field]
        mask = (df["meeting_no"] == int(c.meeting_no)) & (df["meeting_type"] == c.meeting_type)
        if mask.sum() != 1:
            raise ValueError(
                f"Correction targets {mask.sum()} rows: {c.meeting_no} {c.meeting_type}"
            )
        idx = df.index[mask][0]
        if convert(c.old_value) != df.at[idx, c.field]:
            raise ValueError(
                f"Meeting {c.meeting_no}: expected {c.field} = {c.old_value}, "
                f"found {df.at[idx, c.field]}"
            )
        df.at[idx, c.field] = convert(c.new_value)
        df.at[idx, "corrected"] = True
    return df


def add_meetings(df: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    """Append meetings that the table leaves out."""
    rows = pd.DataFrame(
        {
            "meeting_no": additions["meeting_no"].astype(int),
            "meeting_type": additions["meeting_type"],
            "meeting_start": pd.to_datetime(additions["meeting_start"]),
            "meeting_end": pd.to_datetime(additions["meeting_end"]),
            "effective_date": pd.to_datetime(additions["meeting_end"]),
            "mpr": additions["mpr"].astype(float),
            "corrected": True,
        }
    )
    clash = df.merge(rows[["meeting_no", "meeting_type"]], on=["meeting_no", "meeting_type"])
    if len(clash):
        raise ValueError("An added meeting already exists in the table")
    return pd.concat([df, rows], ignore_index=True)


def clean(
    raw: pd.DataFrame,
    corrections: pd.DataFrame | None = None,
    additions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = raw.copy()
    df.columns = ["meeting_no", "mpc_dates", "effective_raw", "mpr"]
    parsed = df["mpc_dates"].map(parse_meeting_dates)
    df["meeting_start"] = pd.to_datetime([p[0] for p in parsed])
    df["meeting_end"] = pd.to_datetime([p[1] for p in parsed])
    df["meeting_type"] = ["emergency" if p[2] else "regular" for p in parsed]
    df["effective_date"] = pd.to_datetime(df["effective_raw"], format="%d %b %Y")
    df["corrected"] = False

    if corrections is not None:
        df = apply_corrections(df, corrections)
    if additions is not None:
        df = add_meetings(df, additions)

    # The decision is announced on the last day of the meeting. The 'effective' column
    # can lag (or, in a few rows, contain typos), so flag rows where it looks wrong.
    gap = (df["effective_date"] - df["meeting_end"]).dt.days
    df["effective_date_flag"] = ~gap.between(-3, 21)

    df = df.sort_values(["meeting_end", "meeting_no"]).reset_index(drop=True)
    df["mpr_change_bp"] = (df["mpr"].diff() * 100).round().astype("Int64")
    df["decision"] = pd.cut(
        df["mpr_change_bp"].astype(float),
        bins=[-float("inf"), -0.5, 0.5, float("inf")],
        labels=["cut", "hold", "hike"],
    ).astype("string")

    keys = df[["meeting_no", "meeting_type"]].duplicated()
    assert not keys.any(), "duplicate (meeting_no, meeting_type)"
    assert df["meeting_end"].is_monotonic_increasing, "meetings out of order"
    cols = ["meeting_no", "meeting_type", "meeting_start", "meeting_end", "effective_date",
            "effective_date_flag", "mpr", "mpr_change_bp", "decision", "corrected"]  # fmt: skip
    return df[cols]


def main() -> None:
    corrections = pd.read_csv(CORRECTIONS, dtype=str) if CORRECTIONS.exists() else None
    additions = pd.read_csv(ADDITIONS, dtype=str) if ADDITIONS.exists() else None
    df = clean(pd.read_csv(RAW), corrections, additions)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"{len(df)} decisions written to {OUT}")
    print(df["meeting_type"].value_counts().to_string())
    print(df["decision"].value_counts().to_string())
    print(f"Rows corrected or added: {int(df['corrected'].sum())}")
    flagged = df[df["effective_date_flag"]]
    if len(flagged):
        print("\nRows whose effective date looks wrong:")
        print(flagged[["meeting_no", "meeting_end", "effective_date"]].to_string(index=False))


if __name__ == "__main__":
    main()