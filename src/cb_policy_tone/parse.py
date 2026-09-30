"""Turn the downloaded MPC press-release PDFs into a meeting-level statements table.

Reads : data/manifest.csv, data/raw/pdf/*.pdf, data/processed/policy_rates.csv
Writes: data/processed/statements.csv         one primary statement per meeting
        data/processed/statements_review.csv  every candidate document and how it was matched
        data/processed/coverage.csv           every meeting, with or without a statement

Each document is matched to a meeting using three independent clues, in this order:
  1. the date printed in the letterhead (reliable for 2003-2009 and for emergency releases)
  2. the meeting number written in the text (e.g. '41st meeting')
  3. the month and year in the post title (reliable from about 2010)
If the clues disagree the document is flagged as a conflict for manual review.
A letterhead date that falls near no meeting stops the match: the document is reported as
unmatched instead of being placed by a weaker clue. A statement that fits no regular meeting
is also tried against the emergency meetings.
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import pdfplumber

MANIFEST = Path("data/manifest.csv")
RATES = Path("data/processed/policy_rates.csv")
TXT_DIR = Path("data/raw/txt")
OUT_STATEMENTS = Path("data/processed/statements.csv")
OUT_REVIEW = Path("data/processed/statements_review.csv")
OUT_COVERAGE = Path("data/processed/coverage.csv")

MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]  # fmt: skip
MONTH_NUM = {name[:3]: i for i, name in enumerate(MONTHS, start=1)}
MONTH_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"

# 'Monetary Policy Committee May 23, 2003' / 'Committee Press Release May 31, 2021'.
# Old letterheads have 'E ST. 1957' noise that can sit between the label and the date.
LETTERHEAD_RE = re.compile(
    r"(?:committee|release)(?:\s+E\s*ST\s*\.?\s*1957)?\s+"
    rf"(?P<m>{MONTH_RE})\s+(?P<d>\d{{1,2}})\s*,?\s+(?P<y>\d{{4}})",
    re.IGNORECASE,
)
TITLE_RE = re.compile(rf"\b({MONTH_RE})\s*,?\s*(\d{{4}})", re.IGNORECASE)
ORDINAL_RE = re.compile(r"\b(\d{1,3})\s?(?:st|nd|rd|th)\b", re.IGNORECASE)
RATE_TERM_RE = re.compile(r"policy rate|prime rate|\bMPR\b", re.IGNORECASE)
PCT_RE = re.compile(r"(\d{1,2}(?:\.\d+)?)\s*(?:per\s*cent|percent|%)", re.IGNORECASE)

STATEMENT_TYPES = {"press_release", "emergency", "highlights"}
LENGTH_TOLERANCE = 0.02  # two texts this close in length count as the same document
SUPPORT_WORDS = ["summary of economic", "charts", "developments report", "inflation outlook",
                 "global economic developments", "impact of covid", "infographic"]  # fmt: skip


def norm(text: str) -> str:
    return " ".join(text.split())


def doc_type(title: str) -> str:
    t = title.lower()
    if "transcript" in t:
        return "transcript"
    if t.startswith("notice"):
        return "notice"
    if "highlights" in t:
        return "highlights"
    if "emergency" in t:
        return "emergency"
    if any(w in t for w in SUPPORT_WORDS):
        return "support_data"
    if "mpc press release" in t:
        return "press_release"
    if t.startswith("mpc meeting"):
        return "meeting_notice"
    return "other"


def letterhead_date(text: str) -> date | None:
    m = LETTERHEAD_RE.search(norm(text)[:400])
    if not m:
        return None
    try:
        return date(int(m["y"]), MONTH_NUM[m["m"][:3].lower()], int(m["d"]))
    except ValueError:
        return None


def text_meeting_number(text: str) -> int | None:
    """First ordinal ('41st', '104th') that sits next to the word meeting / MPC."""
    head = norm(text)[:1500]
    for m in ORDINAL_RE.finditer(head):
        window = head[max(0, m.start() - 60) : m.end() + 80].lower()
        if "meeting" in window or "mpc" in window or "monetary policy committee" in window:
            n = int(m.group(1))
            if 1 <= n <= 140:
                return n
    return None


def title_month(title: str) -> tuple[int, int] | None:
    m = TITLE_RE.search(title)
    if not m:
        return None
    return int(m.group(2)), MONTH_NUM[m.group(1)[:3].lower()]


def rates_near_policy_rate(text: str) -> set[float]:
    """Every 'NN per cent' figure that appears shortly after a mention of the policy rate."""
    t = norm(text)
    found: set[float] = set()
    for m in RATE_TERM_RE.finditer(t):
        for p in PCT_RE.finditer(t[m.start() : m.start() + 250]):
            found.add(float(p.group(1)))
    return found


def meeting_from_date(d: date, regular: pd.DataFrame) -> int | None:
    """Regular meeting whose end date is closest to d, from 5 days before to 30 days after."""
    diff = (pd.Timestamp(d) - regular["meeting_end"]).dt.days
    cand = diff[diff.between(-5, 30)]
    if cand.empty:
        return None
    return int(regular.loc[cand.abs().idxmin(), "meeting_no"])


def meeting_from_title(ym: tuple[int, int], regular: pd.DataFrame) -> int | None:
    year, month = ym
    ends = regular["meeting_end"]
    eff = regular["effective_date"]
    hit_end = (ends.dt.year == year) & (ends.dt.month == month)
    hit_eff = ~regular["effective_date_flag"] & (eff.dt.year == year) & (eff.dt.month == month)
    hits = regular[hit_end | hit_eff]
    return int(hits.iloc[0]["meeting_no"]) if len(hits) == 1 else None


def match_regular(text: str, title: str, regular: pd.DataFrame) -> dict:
    known = set(regular["meeting_no"])
    d = letterhead_date(text)
    by_date = meeting_from_date(d, regular) if d else None
    n = text_meeting_number(text)
    by_number = n if n in known else None
    ym = title_month(title)
    by_title = meeting_from_title(ym, regular) if ym else None

    if by_date is not None:
        chosen, method = by_date, "letterhead_date"
    elif d is not None:
        chosen, method = None, "letterhead_no_meeting"
    elif by_number is not None and by_title in (None, by_number):
        chosen, method = by_number, "text_number"
    elif by_title is not None:
        chosen, method = by_title, "title"
    else:
        chosen, method = None, "unmatched"
    clues = {x for x in (by_date, by_number, by_title) if x is not None}
    return {
        "meeting_no": chosen,
        "meeting_type": "regular",
        "match_method": method,
        "conflict": len(clues) > 1,
        "title_ok": (by_title == chosen) if ym else None,
        "by_date": by_date,
        "by_number": by_number,
        "by_title": by_title,
        "letterhead_date": d.isoformat() if d else "",
        "text_number": n,
        "title_ym": f"{ym[0]}-{ym[1]:02d}" if ym else "",
    }


def match_emergency(text: str, title: str, emergency: pd.DataFrame) -> dict:
    """Match against emergency meetings, by letterhead date (+/- 3 days) or title month."""
    d = letterhead_date(text)
    by_date = None
    if d is not None and not emergency.empty:
        diff = (pd.Timestamp(d) - emergency["meeting_end"]).dt.days.abs()
        if diff.min() <= 3:
            by_date = int(emergency.loc[diff.idxmin(), "meeting_no"])
    ym = title_month(title)
    by_title = None
    if ym is not None and not emergency.empty:
        ends = emergency["meeting_end"]
        hits = emergency[(ends.dt.year == ym[0]) & (ends.dt.month == ym[1])]
        by_title = int(hits.iloc[0]["meeting_no"]) if len(hits) == 1 else None
    chosen = by_date if by_date is not None else by_title
    if by_date is not None:
        method = "letterhead_date"
    elif by_title is not None:
        method = "title"
    else:
        method = "no_rate_row"
    clues = {x for x in (by_date, by_title) if x is not None}
    return {
        "meeting_no": chosen,
        "meeting_type": "emergency",
        "match_method": method,
        "conflict": len(clues) > 1,
        "title_ok": (by_title == chosen) if ym else None,
        "by_date": by_date,
        "by_number": None,
        "by_title": by_title,
        "letterhead_date": d.isoformat() if d else "",
        "text_number": text_meeting_number(text),
        "title_ym": f"{ym[0]}-{ym[1]:02d}" if ym else "",
    }


def match_document(
    text: str, title: str, kind: str, regular: pd.DataFrame, emergency: pd.DataFrame
) -> dict:
    """Emergency releases go to emergency meetings; anything unplaced gets a second try there."""
    if kind == "emergency":
        return match_emergency(text, title, emergency)
    result = match_regular(text, title, regular)
    if result["meeting_no"] is None:
        fallback = match_emergency(text, title, emergency)
        if fallback["meeting_no"] is not None:
            return fallback
    return result


def load_text(pdf_path: Path) -> str:
    """Extract a PDF's text, caching it next to the other raw text files."""
    cached = TXT_DIR / f"{pdf_path.stem}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n\n".join((page.extract_text() or "") for page in pdf.pages)
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text


def classify_documents(manifest: pd.DataFrame, rates: pd.DataFrame) -> list[dict]:
    regular = rates[rates["meeting_type"] == "regular"].reset_index(drop=True)
    emergency = rates[rates["meeting_type"] == "emergency"].reset_index(drop=True)
    docs = []
    for r in manifest[manifest["status"] == "ok"].itertuples():
        kind = doc_type(r.title)
        doc = {
            "doc_id": r.id,
            "title": r.title,
            "doc_type": kind,
            "pdf_file": r.file,
            "page_url": r.page_url,
            "pdf_url": r.pdf_url,
            "meeting_type": "",
            "role": "not_a_statement",
            "duplicate_of": "",
            "text_identical": "",
        }
        if kind in STATEMENT_TYPES:
            text = load_text(Path(r.file))
            body = norm(text)
            doc.update(match_document(text, r.title, kind, regular, emergency))
            doc["n_chars"] = len(body)
            doc["text_hash"] = hashlib.sha256(body.encode()).hexdigest()[:12]
            doc["txt_file"] = (TXT_DIR / f"{Path(r.file).stem}.txt").as_posix()
            doc["rates_in_text"] = sorted(rates_near_policy_rate(text))
            doc["opening"] = body[:120]
            if doc["meeting_no"] is not None:
                doc["role"] = "candidate"
            elif kind == "emergency":
                doc["role"] = "primary_no_rate_row"
            else:
                doc["role"] = "unmatched"
        docs.append(doc)
    return docs


def near_identical(a: dict, b: dict) -> bool:
    if a["text_hash"] == b["text_hash"]:
        return True
    longer = max(a["n_chars"], b["n_chars"])
    return abs(a["n_chars"] - b["n_chars"]) / longer <= LENGTH_TOLERANCE


def choose_primaries(docs: list[dict]) -> None:
    """Pick one primary document per meeting; mark the rest as duplicates or summaries."""
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for d in docs:
        if d["role"] == "candidate":
            groups[(d["meeting_type"], d["meeting_no"])].append(d)
    for group in groups.values():
        full = [d for d in group if d["doc_type"] != "highlights"]
        pool = full or group

        pool.sort(key=lambda d: (d["title_ok"] is False, int(d["doc_id"])))
        primary = pool[0]
        primary["role"] = "primary"
        for d in group:
            if d is primary:
                continue
            is_summary = d["doc_type"] == "highlights" and primary["doc_type"] != "highlights"
            if is_summary:
                d["role"] = "summary_of_primary"
            elif near_identical(d, primary):
                d["role"] = "duplicate"
            else:
                d["role"] = "conflicting_version"
            d["duplicate_of"] = primary["doc_id"]
            d["text_identical"] = d["text_hash"] == primary["text_hash"]


INT_COLUMNS = ["meeting_no", "by_date", "by_number", "by_title", "text_number", "n_chars"]


def to_frame(docs: list[dict]) -> pd.DataFrame:
    """Documents as a DataFrame, with meeting numbers as nullable integers (not floats)."""
    df = pd.DataFrame(docs)
    for col in INT_COLUMNS:
        if col in df:
            df[col] = df[col].astype("Int64")
    return df


def build_tables(docs: list[dict], rates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    review = to_frame(docs)
    primaries = review[review["role"].isin(["primary", "primary_no_rate_row"])].copy()
    keep = ["meeting_no", "meeting_type", "meeting_end", "mpr", "mpr_change_bp", "decision"]
    statements = primaries.merge(rates[keep], on=["meeting_no", "meeting_type"], how="left")
    statements["rate_agrees"] = [
        (mpr in found) if found and pd.notna(mpr) else None
        for mpr, found in zip(statements["mpr"], statements["rates_in_text"], strict=True)
    ]
    cols = ["meeting_no", "meeting_type", "meeting_end", "mpr", "mpr_change_bp", "decision",
            "doc_id", "title", "doc_type", "match_method", "conflict", "rate_agrees",
            "letterhead_date", "text_number", "title_ym", "n_chars", "pdf_file", "txt_file",
            "page_url", "pdf_url"]  # fmt: skip
    statements = statements[cols].sort_values("meeting_end").reset_index(drop=True)
    for col in ("mpr_change_bp", "n_chars"):
        statements[col] = statements[col].astype("Int64")

    have = statements[["meeting_no", "meeting_type", "doc_id", "match_method", "conflict",
                       "rate_agrees"]]  # fmt: skip
    coverage = rates.merge(have, on=["meeting_no", "meeting_type"], how="left")
    coverage["has_statement"] = coverage["doc_id"].notna()
    return statements, coverage


def report(docs: list[dict], statements: pd.DataFrame, coverage: pd.DataFrame) -> None:
    review = to_frame(docs)
    print("Documents by role:")
    print(review["role"].value_counts().to_string())

    reg = coverage[coverage["meeting_type"] == "regular"]
    missing = reg[~reg["has_statement"]]
    print(f"\nRegular meetings with a statement: {reg['has_statement'].sum()} of {len(reg)}")
    print(f"Meetings with NO statement ({len(missing)}):")
    for y, grp in missing.groupby(missing["meeting_end"].dt.year):
        print(f"   {y}: " + ", ".join(f"#{n}" for n in grp["meeting_no"]))

    matched = ["primary", "duplicate", "summary_of_primary", "conflicting_version"]
    cand = review[review["role"].isin(matched)]
    conflicts = cand[cand["conflict"].eq(True)]
    print(f"\nDocuments whose clues disagree ({len(conflicts)}):")
    for d in conflicts.itertuples():
        print(f"   id {d.doc_id} | {d.title[:34]} | chose #{d.meeting_no} via {d.match_method}"
              f" | date->{d.by_date} number->{d.by_number} title->{d.by_title}")  # fmt: skip

    off = cand[cand["title_ok"].eq(False)]
    print(f"\nTitle month does not match the meeting the document was assigned to ({len(off)}):")
    for d in off.itertuples():
        print(
            f"   id {d.doc_id} | {d.title[:34]} | title says {d.title_ym} | assigned #{d.meeting_no}"
            f" (letterhead {d.letterhead_date or 'none'}) | {d.role}"
        )

    un = review[review["role"] == "unmatched"]
    print(f"\nStatements that could not be matched to any meeting ({len(un)}):")
    for d in un.itertuples():
        print(f"   id {d.doc_id} | {d.title[:34]} | {d.match_method} | letterhead "
              f"{d.letterhead_date or 'none'} | {d.opening[:60]}")  # fmt: skip

    diff = review[review["role"] == "conflicting_version"]
    print(f"\nDifferent documents matched to the same meeting ({len(diff)}):")
    for d in diff.itertuples():
        print(f"   id {d.doc_id} ({d.n_chars} chars) vs primary {d.duplicate_of} | "
              f"#{d.meeting_no} {d.meeting_type} | {d.title[:34]}")  # fmt: skip

    dup = review[review["role"] == "duplicate"]
    print(
        f"\nDuplicates set aside: {len(dup)} "
        f"(identical text: {int(dup['text_identical'].eq(True).sum())})"
    )
    for d in dup[dup["text_identical"].eq(False)].itertuples():
        print(f"   near-identical only: id {d.doc_id} vs primary {d.duplicate_of} | {d.title[:40]}")

    bad = statements[statements["rate_agrees"].eq(False)]
    print(
        f"\nRate check: the stated policy rate is not found in the text for {len(bad)} of "
        f"{int(statements['rate_agrees'].notna().sum())} statements checked"
    )
    for d in bad.itertuples():
        print(f"   meeting #{d.meeting_no} | {d.title[:36]} | table rate {d.mpr}")


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    manifest = pd.read_csv(MANIFEST, dtype=str).fillna("")
    rates = pd.read_csv(RATES, parse_dates=["meeting_start", "meeting_end", "effective_date"])
    docs = classify_documents(manifest, rates)
    choose_primaries(docs)
    statements, coverage = build_tables(docs, rates)

    OUT_STATEMENTS.parent.mkdir(parents=True, exist_ok=True)
    to_frame(docs).drop(columns=["rates_in_text"]).to_csv(OUT_REVIEW, index=False)
    statements.to_csv(OUT_STATEMENTS, index=False, date_format="%Y-%m-%d")
    coverage.to_csv(OUT_COVERAGE, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(statements)} statements to {OUT_STATEMENTS}\n")
    report(docs, statements, coverage)


if __name__ == "__main__":
    main()