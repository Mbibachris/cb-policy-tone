"""Build the tone index: one row per statement.

Reads : data/processed/weak_labels.csv, predictions_tone.csv, sentences.csv, statements.csv
Writes: data/processed/tone_index.csv

Only sentences that the codebook allows are scored. Decision sentences, fragments, admin notes
and very short or very long sentences are excluded, so the index never contains the rate
decision itself.

Three versions of the index, each over a statement's scored sentences:
  lexicon_net    = share hawkish minus share dovish, by the lexicon
  model_net      = mean of (P(hawkish) - P(dovish)) from the fine-tuned model
  model_net_hard = share predicted hawkish minus share predicted dovish
All run from -1 (all dovish) to +1 (all hawkish). The level carries no meaning because the
rules are asymmetric by design; only movements over time do.

Note: the model also scored its own training sentences (up to 2018), so model_net is in-sample
for those years.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SENTENCES = Path("data/processed/sentences.csv")
WEAK = Path("data/processed/weak_labels.csv")
PREDICTIONS = Path("data/processed/predictions_tone.csv")
STATEMENTS = Path("data/processed/statements.csv")
OUT = Path("data/processed/tone_index.csv")


def build_index(
    sentences: pd.DataFrame,
    weak: pd.DataFrame,
    predictions: pd.DataFrame,
    statements: pd.DataFrame,
) -> pd.DataFrame:
    scored = sentences[["sentence_id", "doc_id"]].merge(
        weak[weak["labellable"]][["sentence_id", "tone_weak"]], on="sentence_id"
    )
    scored = scored.merge(
        predictions[["sentence_id", "p_dovish", "p_neutral", "p_hawkish"]], on="sentence_id"
    )
    scored["lex"] = scored["tone_weak"].map({"hawkish": 1, "neutral": 0, "dovish": -1})
    scored["prob"] = scored["p_hawkish"] - scored["p_dovish"]
    labels = scored[["p_dovish", "p_neutral", "p_hawkish"]].to_numpy().argmax(axis=1)
    scored["hard"] = pd.Series(labels, index=scored.index).map({0: -1, 1: 0, 2: 1})

    by_doc = scored.groupby("doc_id").agg(
        n_scored=("sentence_id", "size"),
        lexicon_net=("lex", "mean"),
        model_net=("prob", "mean"),
        model_net_hard=("hard", "mean"),
    )
    info = statements[["doc_id", "meeting_no", "meeting_type", "meeting_end", "letterhead_date"]]
    index = by_doc.reset_index().merge(info, on="doc_id", how="left")
    index["date"] = pd.to_datetime(index["meeting_end"]).fillna(
        pd.to_datetime(index["letterhead_date"])
    )
    cols = [
        "doc_id",
        "meeting_no",
        "meeting_type",
        "date",
        "n_scored",
        "lexicon_net",
        "model_net",
        "model_net_hard",
    ]
    return index[cols].sort_values("date").reset_index(drop=True)


def main() -> None:
    index = build_index(
        pd.read_csv(SENTENCES), pd.read_csv(WEAK), pd.read_csv(PREDICTIONS), pd.read_csv(STATEMENTS)
    )
    index.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(index)} statements to {OUT}")
    print(
        index[["n_scored", "lexicon_net", "model_net", "model_net_hard"]]
        .describe()
        .round(3)
        .to_string()
    )
    corr = index[["lexicon_net", "model_net", "model_net_hard"]].corr().round(2)
    print("\nCorrelation between the three versions:")
    print(corr.to_string())
    print("\nStatements with fewer than 15 scored sentences:", int((index["n_scored"] < 15).sum()))


if __name__ == "__main__":
    main()
