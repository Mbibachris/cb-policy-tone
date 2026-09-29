"""Clean the Bank of Ghana 'Historical Policy Rate Decisions' table.

Input : data/external/bog_policy_rate_decisions.csv  (downloaded from the BoG website)
Output: data/processed/policy_rates.csv              (one row per policy decision)
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

RAW = Path("data/external/bog_policy_rate_decisions.csv")
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


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = ["meeting_no", "mpc_dates", "effective_raw", "mpr"]
    parsed = df["mpc_dates"].map(parse_meeting_dates)
    df["meeting_start"] = pd.to_datetime([p[0] for p in parsed])
    df["meeting_end"] = pd.to_datetime([p[1] for p in parsed])
    df["meeting_type"] = ["emergency" if p[2] else "regular" for p in parsed]
    df["effective_date"] = pd.to_datetime(df["effective_raw"], format="%d %b %Y")

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
            "effective_date_flag", "mpr", "mpr_change_bp", "decision"]  # fmt: skip
    return df[cols]


def main() -> None:
    df = clean(pd.read_csv(RAW))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"{len(df)} decisions written to {OUT}")
    print(df["meeting_type"].value_counts().to_string())
    print(df["decision"].value_counts().to_string())
    flagged = df[df["effective_date_flag"]]
    if len(flagged):
        print("\nRows whose effective date looks wrong:")
        print(flagged[["meeting_no", "meeting_end", "effective_date"]].to_string(index=False))


if __name__ == "__main__":
    main()