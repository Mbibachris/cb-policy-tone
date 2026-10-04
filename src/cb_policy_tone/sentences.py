"""Clean statement text and split it into sentences.

Reads : data/processed/statements.csv and the extracted text files it points to
Writes: data/processed/sentences.csv   (one row per sentence)

Each sentence carries its position, the section heading it sits under (when the statement has
headings), and a flag for sentences that announce the rate decision. Those are kept out of the
tone index later, so that the index is not validated against the decision it contains.

sentences.csv is kept out of git until the Bank's terms of use have been checked.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

STATEMENTS = Path("data/processed/statements.csv")
OUT = Path("data/processed/sentences.csv")

MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
EST = r"(?:E\s*ST\s*\.?\s*1957)"
LETTERHEAD = re.compile(
    rf"\A.{{0,300}}?(?:committee|release)\s*(?:{EST}\s*)?{MONTH}\s*(?:\d{{1,2}}\s*,?\s*)?\d{{4}}"
    rf"(?:\s*{EST})?",
    re.IGNORECASE | re.DOTALL,
)
NUMBERED = re.compile(r"^\(?\d{1,3}[.)]\s+(?=\S)")
BULLET_GLYPHS = "•▪◦●■·\uf0b7\uf0a7\uf0d8\uf076"
BULLET = re.compile(rf"^(?:[{BULLET_GLYPHS}]\s*|\(cid:\d+\)\s*|[-–—]\s+(?=[A-Z]))")
INLINE_BULLET = re.compile(rf"\s+[{BULLET_GLYPHS}]\s+|\(cid:\d+\)")
LETTERHEAD_WORDS = re.compile(
    r"(?:(?:bank of ghana|monetary policy committee|press release|press briefing|public)\s*)+",
    re.IGNORECASE,
)
DEBRIS = re.compile(
    r"press statement|for immediate release|release date|\best\b|full text|kofgh",
    re.IGNORECASE,
)
MID_NUMBER = re.compile(r"(?<=[.!?])\s+\d{1,3}\.\s+(?=[A-Z])")
LIST_MARKER = re.compile(r"^[A-Z]\.\s+")
INLINE_HEADING = re.compile(
    r"^(?P<head>(?:[A-Z][A-Za-z’'/&-]*\s+){0,7}[A-Z][A-Za-z’'/&-]*)\s+(?=\d{1,3}\.\s+[A-Z])"
)
STOPWORDS = {
    "and",
    "of",
    "the",
    "in",
    "to",
    "for",
    "on",
    "a",
    "an",
    "with",
    "by",
    "at",
    "from",
    "&",
}

DOT = "\u2024"  # stands in for a full stop that must not end a sentence
ABBREVIATION = re.compile(
    r"\b(?:U\.S|U\.K|E\.U|e\.g|i\.e|a\.m|p\.m|p\.a|vs|Dr|Mr|Mrs|Ms|Prof|Hon|Gov|No|Nos|Fig"
    r"|approx|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|St|Ltd|Inc|Co|Corp|Dept|Govt)\."
)
INITIAL = re.compile(r"\b[A-Z]\.(?=\s+[A-Z])")
SPLIT = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["”’)\]]))\s+(?=["“(\[]?[A-Z0-9])')

RATE_TERM = re.compile(
    r"\b(?:policy rate|prime rate|monetary policy rate|MPR|policy stance)\b", re.IGNORECASE
)
ANNOUNCED_RATE = re.compile(
    r"\bdecid\w*\s+to\s+(?:\w+\s+){0,3}(?:the\s+|its\s+)?rate\b", re.IGNORECASE
)
IMPERATIVE = re.compile(
    r"^(?:increase|raise|reduce|lower|cut|decrease|maintain|keep|retain|hold)\b", re.IGNORECASE
)
ACTION = re.compile(
    r"\b(?:maintain|keep|kept|retain|hold|held|unchanged|raise|increase|reduce|lower|cut"
    r"|decrease|hike|ease|tighten)\w*\b",
    re.IGNORECASE,
)
ANNOUNCE = re.compile(r"\b(?:decid\w*|decision|voted?|agreed|resolved?)\b", re.IGNORECASE)
DIRECT_ACTION = re.compile(
    r"\b(?:committee|mpc)\s+(?:therefore\s+|accordingly\s+|also\s+|thus\s+)?"
    r"(?:maintained|kept|retained|held|raised|increased|reduced|lowered|cut|decreased|hiked)\b"
    r"|\b(?:therefore|accordingly|thus|hence)\s+(?:maintained|kept|retained|held|raised"
    r"|increased|reduced|lowered|cut|decreased|hiked)\b",
    re.IGNORECASE,
)
OTHER_BANKS = re.compile(
    r"\b(?:Fed|Federal Reserve|FOMC|ECB|European Central Bank|Bank of England|Bank of Japan"
    r"|SARB|Reserve Bank|CBN|Central Bank of Nigeria)\b"
)

SECTION_HINTS = [
    ("admin", r"information(?:al)? note"),
    ("global", r"global|world|international"),
    ("inflation", r"inflation|price"),
    ("growth", r"growth|real sector|economic activity|output"),
    ("fiscal", r"fiscal|budget|government|debt"),
    ("external", r"external|exchange rate|balance of payments|reserves|foreign|trade"),
    ("financial", r"financial|monetary|banking|credit|money market|interest rate"),
    ("decision", r"decision|conclusion|policy stance|outlook"),
]


def strip_letterhead(text: str) -> str:
    """Remove the letterhead block (logo letters, 'Press Release', date) from the start."""
    m = LETTERHEAD.match(text)
    if not m or re.search(r"[a-z]{3,}[.!?]\s", m.group(0)):
        return text
    return text[m.end() :]


def is_noise_line(line: str) -> bool:
    """Page numbers and letter-spaced logo debris such as 'K O F G H'."""
    stripped = line.strip()
    if not stripped or LETTERHEAD_WORDS.fullmatch(stripped):
        return True
    if re.fullmatch(r"(?:page\s+)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?", stripped, re.IGNORECASE):
        return True
    tokens = stripped.split()
    if stripped.isupper() and len(tokens) <= 3 and all(len(t.strip(".,")) <= 5 for t in tokens):
        return True  # 'KOFGH', 'E 7', 'THANK YOU', 'END.'
    singles = sum(len(t.strip(".,")) <= 1 for t in tokens)
    return len(tokens) >= 4 and singles / len(tokens) >= 0.6


def drop_repeated_lines(lines: list[str], min_repeats: int = 3) -> list[str]:
    """Drop short header/footer lines that repeat on several pages."""

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.lower()).strip()

    counts = Counter(key(line) for line in lines)

    def is_header(line: str) -> bool:
        ends_sentence = line.rstrip().endswith((".", "?", "!", ":", ";", ","))
        starts_like_header = line[0].isupper() or line[0].isdigit() or line[0] in "@w"
        return (
            counts[key(line)] >= min_repeats
            and len(line.split()) <= 8
            and not ends_sentence
            and starts_like_header
        )

    return [line for line in lines if not is_header(line)]


def looks_like_heading(
    line: str, prev: str | None, nxt: str | None, prev_was_heading: bool = False
) -> bool:
    """A short title-case line between two sentences, e.g. 'Global Developments'."""
    text = LIST_MARKER.sub("", line).strip()
    words = text.split()
    if not 1 <= len(words) <= 9 or not text[0].isupper():
        return False
    if line.rstrip()[-1] in ".?!:;,":
        return False
    if prev is not None and not prev_was_heading and prev.rstrip()[-1] not in ".?!:":
        return False
    if nxt is None or not (nxt[0].isupper() or nxt[0].isdigit() or nxt[0] in "•"):
        return False
    content = [w for w in words if w.lower() not in STOPWORDS]
    return all(w[0].isupper() or not w[0].isalpha() for w in content)


def section_hint(heading: str) -> str:
    for topic, pattern in SECTION_HINTS:
        if re.search(pattern, heading, re.IGNORECASE):
            return topic
    return ""


def is_letterhead_debris(line: str) -> bool:
    words = line.split()
    if len(words) > 10:
        return False
    if DEBRIS.search(line):
        return True
    logo_letters = sum(len(w.strip(".,")) == 1 and w.isupper() for w in words)
    return "bank of ghana" in line.lower() and len(words) <= 6 and logo_letters >= 2


def drop_letterhead_debris(lines: list[str], window: int = 15) -> list[str]:
    """In the first lines, drop short leftovers of the letterhead ('Press Statement No 121')."""
    return [ln for ln in lines[:window] if not is_letterhead_debris(ln)] + lines[window:]


def clean_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in strip_letterhead(text).splitlines()]
    lines = [ln for ln in lines if not is_noise_line(ln)]
    return drop_repeated_lines(drop_letterhead_debris(lines))


def build_paragraphs(lines: list[str]) -> list[tuple[str, str]]:
    """Group lines into (heading, paragraph text). Numbers and bullets start new paragraphs."""
    paragraphs: list[tuple[str, str]] = []
    heading, current, last_was_heading = "", [], False

    def flush() -> None:
        if current:
            paragraphs.append((heading, " ".join(current)))
            current.clear()

    for i, line in enumerate(lines):
        prev = lines[i - 1] if i > 0 else None
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if looks_like_heading(line, prev, nxt, last_was_heading):
            flush()
            title = LIST_MARKER.sub("", line).strip()
            heading = f"{heading} / {title}" if last_was_heading else title
            last_was_heading = True
            continue
        last_was_heading = False
        inline = INLINE_HEADING.match(line)
        if inline:  # 'Global Developments 1. The global economy ...' on one line
            flush()
            heading = inline["head"].strip()
            line = line[inline.end() :]
        if NUMBERED.match(line) or BULLET.match(line):
            flush()
            line = BULLET.sub("", NUMBERED.sub("", line, count=1), count=1)
        if current and current[-1].endswith("-"):
            current[-1] = current[-1] + line  # 'year-on-' + 'year'
        else:
            current.append(line)
    flush()
    return paragraphs


def split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    for part in INLINE_BULLET.split(paragraph):
        text = MID_NUMBER.sub(" ", part)
        text = ABBREVIATION.sub(lambda m: m.group(0)[:-1] + DOT, text)
        text = INITIAL.sub(lambda m: m.group(0)[:-1] + DOT, text)
        sentences += [s.replace(DOT, ".").strip() for s in SPLIT.split(text) if s.strip()]
    return sentences


def is_decision_sentence(text: str) -> bool:
    """Announces the rate decision: a policy-rate mention plus an announced or direct action."""
    if OTHER_BANKS.search(text) and "committee" not in text.lower():
        return False
    if ANNOUNCED_RATE.search(text):  # 'decided to reduce the rate by 50 basis points'
        return True
    if not RATE_TERM.search(text):
        return False
    announced = bool(ACTION.search(text) and ANNOUNCE.search(text))
    return announced or bool(DIRECT_ACTION.search(text)) or bool(IMPERATIVE.match(text))


def split_statement(text: str) -> list[dict]:
    """One statement's text -> one dict per sentence."""
    rows = []
    for p_no, (heading, paragraph) in enumerate(build_paragraphs(clean_lines(text)), start=1):
        for sentence in split_sentences(paragraph):
            n_words = len(sentence.split())
            rows.append(
                {
                    "paragraph": p_no,
                    "position": len(rows) + 1,
                    "text": sentence,
                    "n_words": n_words,
                    "is_short": n_words < 5,
                    "is_fragment": sentence[0].islower(),
                    "is_decision": is_decision_sentence(sentence),
                    "section_heading": heading,
                    "section_hint": section_hint(heading),
                }
            )
    return rows


def report(df: pd.DataFrame, n_cid_statements: int = 0) -> None:
    per_doc = df.groupby("doc_id").size()
    print(f"Sentences: {len(df)} from {df['doc_id'].nunique()} statements")
    print(
        f"Per statement: min {per_doc.min()}, median {int(per_doc.median())}, max {per_doc.max()}"
    )
    few = per_doc[per_doc < 15]
    print(f"Statements with fewer than 15 sentences ({len(few)}): {list(few.index)[:20]}")
    print(f"Raw texts containing '(cid:' glyph codes: {n_cid_statements}")

    words = df["n_words"]
    print(
        f"\nWords per sentence: median {int(words.median())}, 90th percentile "
        f"{int(words.quantile(0.9))}, 99th {int(words.quantile(0.99))}, max {words.max()}"
    )
    print(f"Sentences over 80 words (probably not split): {int((words > 80).sum())}")
    print(f"Sentences under 5 words: {int(df['is_short'].sum())}")
    print(f"Sentences starting with a lowercase letter: {int(df['is_fragment'].sum())}")
    closers = (".", "?", "!", ":", '"', "\u201d", ")")
    unended = df[~df["text"].str.rstrip().str.endswith(closers)]
    print(f"Sentences with no closing punctuation: {len(unended)}")

    decisions = df.groupby("doc_id")["is_decision"].sum()
    print(
        f"\nDecision sentences: {int(decisions.sum())} in total; statements with none: "
        f"{int((decisions == 0).sum())}, with more than 4: {int((decisions > 4).sum())}"
    )
    for doc_id in decisions[decisions == 0].index:
        part = df[df["doc_id"] == doc_id]
        hits = part[part["text"].str.contains("decid|policy rate|prime rate", case=False)]
        print(f"   none in doc {doc_id} (meeting {part['meeting_no'].iloc[0]}); candidates:")
        for r in hits.head(4).itertuples():
            print(f"      {r.text[:170]}")

    has_heading = df[df["section_heading"] != ""]
    headings = has_heading.groupby("section_heading")["doc_id"].nunique()
    print(f"\nStatements with at least one heading: {has_heading['doc_id'].nunique()}")
    print("Most common headings (statements containing them):")
    for name, n in headings.sort_values(ascending=False).head(12).items():
        print(f"   {n:3d}  {name[:60]}")
    rare = headings[headings == 1].sample(min(12, int((headings == 1).sum())), random_state=1)
    print(f"Headings found in only one statement ({int((headings == 1).sum())}); a sample:")
    for name in rare.index:
        print(f"      {name[:70]}")
    print(
        "Sentences per topic hint:",
        df["section_hint"].replace("", "(none)").value_counts().to_dict(),
    )

    print("\nLongest sentences:")
    for r in df.nlargest(5, "n_words").itertuples():
        print(f"   doc {r.doc_id} | {r.n_words} words | {r.text[:130]}")
    print("Most common very short sentences:")
    for text, n in df[df["is_short"]]["text"].value_counts().head(8).items():
        print(f"   {n:3d}  {text[:60]}")
    print("Lowercase starts, with the sentence before each:")
    frag = df[df["is_fragment"]]
    for r in frag.head(6).itertuples():
        before = df[(df["doc_id"] == r.doc_id) & (df["position"] == r.position - 1)]["text"]
        print(
            f"   doc {r.doc_id} | ...{before.iloc[0][-60:] if len(before) else ''} || {r.text[:70]}"
        )

    print("\nSamples from the start and middle of four statements across the period:")
    docs = df.drop_duplicates("doc_id").sort_values("meeting_end")["doc_id"].tolist()
    for doc_id in [docs[0], docs[len(docs) // 3], docs[2 * len(docs) // 3], docs[-1]]:
        part = df[df["doc_id"] == doc_id].reset_index(drop=True)
        print(f"   --- doc {doc_id} | meeting {part['meeting_no'].iloc[0]} | {len(part)} sentences")
        for i in sorted({0, 1, len(part) // 2, len(part) // 2 + 1} & set(range(len(part)))):
            r = part.iloc[i]
            print(f"      #{r['position']:3d} [{r['section_hint'] or '-'}] {r['text'][:130]}")


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    statements = pd.read_csv(STATEMENTS)
    rows, n_cid = [], 0
    for s in statements.itertuples():
        text = Path(s.txt_file).read_text(encoding="utf-8")
        n_cid += "(cid:" in text
        for r in split_statement(text):
            rows.append(
                {
                    "sentence_id": f"{s.doc_id}-{r['position']:03d}",
                    "doc_id": s.doc_id,
                    "meeting_no": s.meeting_no,
                    "meeting_type": s.meeting_type,
                    "meeting_end": s.meeting_end,
                    **r,
                }
            )
    df = pd.DataFrame(rows)
    df["meeting_no"] = df["meeting_no"].astype("Int64")
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} sentences to {OUT}\n")
    report(df, n_cid)


if __name__ == "__main__":
    main()