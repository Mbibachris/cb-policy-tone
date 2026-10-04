"""Rule-based weak labels for tone and topic, built from the codebook.

Reads : data/processed/sentences.csv
        data/labels/tone_rules.csv    (direction rules, one row per rule)
        data/labels/topic_rules.csv   (topic keywords with weights)
Writes: data/processed/weak_labels.csv   (ids and labels only, no sentence text)

A tone rule fires when a subject term and a movement word appear close together
('inflation ... rose'), or when a single phrase matches ('room to ease'). The label is the
direction with more distinct rules firing; equal numbers, or none, give neutral.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SENTENCES = Path("data/processed/sentences.csv")
TONE_RULES = Path("data/labels/tone_rules.csv")
TOPIC_RULES = Path("data/labels/topic_rules.csv")
OUT = Path("data/processed/weak_labels.csv")

MOVES = {
    "UP": (
        r"\b(?:rose|rise|rises|rising|risen|increase[sd]?|increasing|accelerat\w+|picked up|pick up"
        r"|edged up|jump\w*|surg\w+|climb\w*|spik\w+|higher|grew|grow|grows|growing|grown|expand\w*"
        r"|widen\w*|deteriorat\w+|worsen\w*|intensif\w+|escalat\w+|mount\w*|heighten\w*|improv\w+"
        r"|accumulat\w+|strengthen\w*|up(?!\s+to\b))\b"
    ),
    "DOWN": (
        r"\b(?:fell|fall|falls|falling|fallen|declin\w+|decreas\w+|drop\w*|ease|eas(?:ed|es|ing)"
        r"|moderat\w+|slow\w*|subsid\w+|lower|lowered|reduc\w+|narrow\w+|contract\w+|weaken\w*"
        r"|slump\w*|soften\w*|dampen\w*|dip\w*|retreat\w*|cool\w*|abat\w+|recede\w*|shrink\w*"
        r"|shrank|deceler\w+|halv\w+|plung\w+|collaps\w+|down)\b"
    ),
    "HIGH": (
        r"\b(?:elevated|above|exceed\w*|breach\w*|overshoot\w*|high|persist\w*|stubborn\w*"
        r"|entrenched)\b"
    ),
    "LOW": r"\b(?:below|subdued|benign|anchored|contained|muted)\b",
}
NEGATORS = {"not", "no", "never", "without", "unlikely", "fail", "failed", "fails"}
LABELS = {"hawkish", "dovish"}

TOPIC_ORDER = [
    "policy",
    "global",
    "external",
    "fiscal",
    "financial",
    "inflation",
    "growth",
    "other",
]
TOPIC_METHOD = "first"  # 'first' = earliest strong keyword; 'sum' = summed keyword weights
HINT_TO_TOPIC = {  # 'Summary and Outlook' (decision) and admin notes say nothing about the topic
    "global": "global",
    "inflation": "inflation",
    "growth": "growth",
    "fiscal": "fiscal",
    "external": "external",
    "financial": "financial",
}


@dataclass(frozen=True)
class ToneRule:
    rule_id: str
    label: str
    call: str
    subject: re.Pattern | None
    move: re.Pattern
    require: re.Pattern | None
    window: int
    note: str


def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def load_tone_rules(path: Path = TONE_RULES) -> list[ToneRule]:
    table = pd.read_csv(path, dtype=str).fillna("")
    rules = []
    for r in table.itertuples():
        if r.label not in LABELS:
            raise ValueError(f"{r.rule_id}: label must be hawkish or dovish, got {r.label!r}")
        if r.move.startswith("re:"):
            move = _compile(r.move[3:])
        elif r.move in MOVES:
            move = _compile(MOVES[r.move])
        else:
            raise ValueError(f"{r.rule_id}: unknown move {r.move!r}")
        rules.append(
            ToneRule(
                rule_id=r.rule_id,
                label=r.label,
                call=r.call,
                subject=_compile(r.subject) if r.subject else None,
                move=move,
                require=_compile(r.require) if r.require else None,
                window=int(r.window or 0),
                note=r.note,
            )
        )
    ids = [r.rule_id for r in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate rule_id in the tone rules")
    return rules


def _negated(text: str, start: int) -> bool:
    previous = re.findall(r"\w+", text[:start].lower())[-3:]
    return any(word in NEGATORS for word in previous)


def _words_between(text: str, a: tuple[int, int], b: tuple[int, int]) -> int:
    if b[0] >= a[1]:
        return len(text[a[1] : b[0]].split())
    if a[0] >= b[1]:
        return len(text[b[1] : a[0]].split())
    return 0


def rule_fires(rule: ToneRule, text: str, others: tuple[re.Pattern, ...] = ()) -> bool:
    """Does the rule fire on this sentence?

    A movement word counts for the subject nearest to it. `others` are the subject patterns
    of other rules: in 'growth slowed while inflation rose', 'slowed' belongs to growth, so
    the inflation rules must not use it.
    """
    if rule.require and not rule.require.search(text):
        return False
    moves = [m.span() for m in rule.move.finditer(text) if not _negated(text, m.start())]
    if not moves:
        return False
    if rule.subject is None:
        return True
    subjects = [m.span() for m in rule.subject.finditer(text)]
    other_spans: list[tuple[int, int]] | None = None
    for subject in subjects:
        for move in moves:
            distance = _words_between(text, subject, move)
            if distance > rule.window:
                continue
            if other_spans is None:
                other_spans = [m.span() for o in others for m in o.finditer(text)]
            nearest_other = min(
                (
                    _words_between(text, other, move)
                    for other in other_spans
                    if not (other[0] < subject[1] and subject[0] < other[1])  # not the same words
                ),
                default=None,
            )
            if nearest_other is not None and nearest_other < distance:
                continue
            return True
    return False


def score_tone(text: str, rules: list[ToneRule]) -> dict:
    fired = []
    for rule in rules:
        others = tuple(
            r.subject
            for r in rules
            if r.subject is not None
            and rule.subject is not None
            and r.subject.pattern != rule.subject.pattern
        )
        if rule_fires(rule, text, others):
            fired.append(rule)
    hawk = [r.rule_id for r in fired if r.label == "hawkish"]
    dove = [r.rule_id for r in fired if r.label == "dovish"]
    score = len(hawk) - len(dove)
    label = "hawkish" if score > 0 else "dovish" if score < 0 else "neutral"
    return {"label": label, "score": score, "hawk": hawk, "dove": dove}


def load_topic_rules(path: Path = TOPIC_RULES) -> list[tuple[str, re.Pattern, float]]:
    table = pd.read_csv(path, dtype=str).fillna("")
    rules = []
    for r in table.itertuples():
        if r.topic not in TOPIC_ORDER:
            raise ValueError(f"Unknown topic {r.topic!r}")
        rules.append((r.topic, _compile(r.pattern), float(r.weight)))
    return rules


def _topic_by_sum(matches: list[tuple[int, str, float]]) -> str:
    scores = dict.fromkeys(TOPIC_ORDER, 0.0)
    for _, topic, weight in matches:
        scores[topic] += weight
    best = max(scores.values())
    return next(t for t in TOPIC_ORDER if scores[t] == best) if best > 0 else ""


def _topic_by_first(matches: list[tuple[int, str, float]]) -> str:
    """The topic of the earliest strong keyword (a stand-in for the grammatical subject).

    A clear statement about the Bank's own stance or guidance wins first, unless another
    central bank is mentioned ('The Fed raised its policy rate' is about the Fed).
    """
    other_central_bank = any(t == "global" and w >= 4 for _, t, w in matches)
    if any(t == "policy" and w >= 3 for _, t, w in matches) and not other_central_bank:
        return "policy"
    pool = [m for m in matches if m[2] >= 2] or matches
    if not pool:
        return ""
    first = min(position for position, _, _ in pool)
    at_first = {t: w for position, t, w in pool if position == first}
    best = max(at_first.values())
    return next(t for t in TOPIC_ORDER if at_first.get(t) == best)


def score_topic(
    text: str,
    hint: str,
    rules: list[tuple[str, re.Pattern, float]],
    method: str = TOPIC_METHOD,
) -> dict:
    matches = []
    for topic, pattern, weight in rules:
        m = pattern.search(text)
        if m:
            matches.append((m.start(), topic, weight))
    topic = _topic_by_first(matches) if method == "first" else _topic_by_sum(matches)
    from_hint = HINT_TO_TOPIC.get(hint or "", "")
    if topic:
        agrees = (topic == from_hint) if from_hint else None
        return {"topic": topic, "source": "keywords", "hint_agrees": agrees}
    if from_hint:
        return {"topic": from_hint, "source": "heading", "hint_agrees": None}
    return {"topic": "", "source": "none", "hint_agrees": None}


def is_labellable(row) -> bool:
    """The pool the codebook allows: no decision sentences, fragments, short or long ones."""
    return not (
        row.is_decision
        or row.is_fragment
        or row.is_short
        or row.n_words > 80
        or row.section_hint == "admin"
    )


def label_sentences(sentences: pd.DataFrame, tone_rules, topic_rules) -> pd.DataFrame:
    rows = []
    for s in sentences.itertuples():
        out = {"sentence_id": s.sentence_id, "labellable": bool(is_labellable(s))}
        if out["labellable"]:
            tone = score_tone(s.text, tone_rules)
            topic = score_topic(s.text, s.section_hint, topic_rules)
            out.update(
                tone_weak=tone["label"],
                tone_score=tone["score"],
                hawk_rules=" ".join(tone["hawk"]),
                dove_rules=" ".join(tone["dove"]),
                topic_weak=topic["topic"],
                topic_source=topic["source"],
                hint_agrees=topic["hint_agrees"],
            )
        rows.append(out)
    return pd.DataFrame(rows)


def report(labels: pd.DataFrame, sentences: pd.DataFrame, rules: list[ToneRule]) -> None:
    pool = labels[labels["labellable"]].merge(
        sentences[["sentence_id", "text", "meeting_end", "section_hint"]], on="sentence_id"
    )
    pool["year"] = pd.to_datetime(pool["meeting_end"]).dt.year
    print(f"Sentences: {len(labels)} | labellable under the codebook: {len(pool)}")

    tone = pool["tone_weak"].value_counts()
    print("\nTone (labellable):", {k: f"{v} ({v / len(pool):.0%})" for k, v in tone.items()})
    hit = ((pool["hawk_rules"] != "") | (pool["dove_rules"] != "")).mean()
    both = ((pool["hawk_rules"] != "") & (pool["dove_rules"] != "")).sum()
    print(f"At least one rule fired: {hit:.0%} | both directions fired (often ties): {both}")

    print("\nNet tone by year (share hawkish minus share dovish; a face-validity check only):")
    by_year = pool.groupby("year")["tone_weak"].agg(
        n="size",
        net=lambda s: ((s == "hawkish").sum() - (s == "dovish").sum()) / len(s),
    )
    line = []
    for year, r in by_year.iterrows():
        line.append(f"{int(year)}: {r['net']:+.2f} (n={int(r['n'])})")
        if len(line) == 4:
            print("   " + " | ".join(line))
            line = []
    if line:
        print("   " + " | ".join(line))

    fired = pd.concat([pool["hawk_rules"], pool["dove_rules"]]).str.split().explode().dropna()
    counts = fired.value_counts()
    print("\nMost frequent rules:", ", ".join(f"{k}={v}" for k, v in counts.head(12).items()))
    never = [r.rule_id for r in rules if r.rule_id not in counts.index]
    print(f"Rules that never fired ({len(never)}): {never}")

    topic = pool["topic_weak"].replace("", "(none)").value_counts()
    print("\nTopic:", {k: f"{v} ({v / len(pool):.0%})" for k, v in topic.items()})
    print("Topic source:", pool["topic_source"].value_counts().to_dict())
    both_sources = pool[pool["hint_agrees"].isin([True, False])]
    if len(both_sources):
        agree = both_sources["hint_agrees"].astype(bool).mean()
        print(
            f"Keyword topic agrees with the heading hint in {agree:.0%} of {len(both_sources)} "
            "sentences that have both. By hint:"
        )
        by_hint = both_sources.groupby("section_hint")["hint_agrees"].agg(
            n="size", agree=lambda s: s.astype(bool).mean()
        )
        print(
            "   "
            + " | ".join(f"{h}: {r['agree']:.0%} (n={int(r['n'])})" for h, r in by_hint.iterrows())
        )
        hinted = pool[pool["section_hint"].isin(HINT_TO_TOPIC)]
        rules = load_topic_rules()
        comparison = {}
        for method in ("first", "sum"):
            topics = [
                score_topic(t, h, rules, method)["topic"]
                for t, h in zip(hinted["text"], hinted["section_hint"], strict=True)
            ]
            comparison[method] = sum(
                t == HINT_TO_TOPIC[h] for t, h in zip(topics, hinted["section_hint"], strict=True)
            ) / len(hinted)
        print(
            f"Agreement with heading hints by method (n={len(hinted)}): first-mention "
            f"{comparison['first']:.0%} | summed weights {comparison['sum']:.0%}"
        )

    print("\nExamples (random, for eyeballing; not for tuning against rate decisions):")
    for label in ("hawkish", "dovish", "neutral"):
        sample = pool[pool["tone_weak"] == label].sample(
            min(4, int((pool["tone_weak"] == label).sum())), random_state=3
        )
        print(f"  --- {label}")
        for r in sample.itertuples():
            fired_ids = (r.hawk_rules + " " + r.dove_rules).strip()
            print(f"     [{r.topic_weak or '-'}|{fired_ids or 'no rule'}] {r.text[:120]}")


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    sentences = pd.read_csv(SENTENCES)
    sentences["section_hint"] = sentences["section_hint"].fillna("")
    tone_rules = load_tone_rules()
    topic_rules = load_topic_rules()
    labels = label_sentences(sentences, tone_rules, topic_rules)
    labels.to_csv(OUT, index=False)
    print(f"Wrote {len(labels)} rows to {OUT}")
    report(labels, sentences, tone_rules)


if __name__ == "__main__":
    main()
