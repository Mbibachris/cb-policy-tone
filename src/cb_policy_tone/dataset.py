"""Build the training files for the fine-tuning notebook.

Reads : data/processed/sentences.csv, weak_labels.csv, statements.csv
        data/labels/gold_key_DO_NOT_OPEN.csv   (only the sentence ids are used)
Writes: data/processed/hf/tone.csv, topic.csv, sentences_all.csv

The files are uploaded to a PRIVATE dataset repo on the Hub (the sentence text is derived from
the Bank's statements, and the terms of use have not been checked yet).

Rules:
- Training labels are the weak labels. They are noisy, so the model learns to imitate the
  rules; the gold set is what tells us how good it is.
- Every gold sentence, and any sentence with the same text, is kept OUT of training.
- Splits are by date, never by sentence: train up to 2018, validation 2019-2021, test 2022-2026.
  The test split here is scored against weak labels, so it measures agreement with the rules
  and is only a monitor. The real test is the hand-labelled gold-test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cb_policy_tone.gold_sample import STATEMENTS, TOPICS, WEAK, build_pool

SENTENCES = Path("data/processed/sentences.csv")
GOLD_KEY = Path("data/labels/gold_key_DO_NOT_OPEN.csv")
OUT_DIR = Path("data/processed/hf")
TONE_LABELS = ["dovish", "neutral", "hawkish"]
TOPIC_LABELS = TOPICS
DEFAULT_REPO = "Kobichris/cb-policy-tone-data"


def split_of(year: float) -> str:
    if year <= 2018:
        return "train"
    if year <= 2021:
        return "validation"
    return "test"


def build_tables(pool: pd.DataFrame, gold_ids: set[str]) -> dict[str, pd.DataFrame]:
    """Tone and topic tables with a date-based split, with all gold sentences removed."""
    gold_texts = set(pool.loc[pool["sentence_id"].isin(gold_ids), "norm_text"])
    usable = pool[~pool["sentence_id"].isin(gold_ids) & ~pool["norm_text"].isin(gold_texts)].copy()
    usable["split"] = usable["year"].map(split_of)

    tone = usable[["sentence_id", "doc_id", "text", "tone_weak", "split"]].rename(
        columns={"tone_weak": "label"}
    )
    topic = usable[usable["topic_weak"] != "(none)"][
        ["sentence_id", "doc_id", "text", "topic_weak", "split"]
    ].rename(columns={"topic_weak": "label"})
    return {"tone": tone.reset_index(drop=True), "topic": topic.reset_index(drop=True)}


def summary(tables: dict[str, pd.DataFrame]) -> str:
    lines = []
    for name, table in tables.items():
        lines.append(f"{name}: {len(table)} rows")
        lines.append(table.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    return "\n".join(lines)


def push(folder: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="dataset")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--push", action="store_true", help="upload to a private Hub dataset repo")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()

    sentences = pd.read_csv(SENTENCES)
    sentences["section_heading"] = sentences["section_heading"].fillna("")
    pool = build_pool(sentences, pd.read_csv(WEAK), pd.read_csv(STATEMENTS))
    gold_ids = set(pd.read_csv(GOLD_KEY, usecols=["sentence_id"])["sentence_id"])
    tables = build_tables(pool, gold_ids)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)
    sentences[["sentence_id", "text"]].to_csv(OUT_DIR / "sentences_all.csv", index=False)
    print(f"Gold sentences kept out of training: {len(gold_ids)}")
    print(summary(tables))
    print(f"\nFiles written to {OUT_DIR}")
    if args.push:
        push(OUT_DIR, args.repo)
        print(f"Uploaded to https://huggingface.co/datasets/{args.repo} (private)")


if __name__ == "__main__":
    main()
