import pandas as pd
import pytest

from cb_policy_tone.dataset import build_tables, split_of, summary


def make_pool() -> pd.DataFrame:
    rows = []
    for i, year in enumerate([2005, 2010, 2018, 2019, 2021, 2022, 2026, 2012, 2024]):
        rows.append(
            {
                "sentence_id": f"d{i}-001", "doc_id": i, "text": f"Sentence number {i}.",
                "norm_text": f"sentence number {i}", "year": year,
                "tone_weak": ["dovish", "neutral", "hawkish"][i % 3],
                "topic_weak": "(none)" if i == 0 else "growth",
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2003, "train"), (2018, "train"), (2019, "validation"), (2021, "validation"), (2022, "test")],
)
def test_split_is_by_date(year, expected):
    assert split_of(year) == expected


def test_every_gold_sentence_is_kept_out_of_training():
    pool = make_pool()
    tables = build_tables(pool, gold_ids={"d1-001", "d5-001"})
    for table in tables.values():
        assert not table["sentence_id"].isin({"d1-001", "d5-001"}).any()


def test_a_sentence_with_the_same_text_as_a_gold_sentence_is_also_removed():
    pool = make_pool()
    clone = pool.iloc[[3]].copy()
    clone["sentence_id"] = "other-001"
    clone["norm_text"] = pool.loc[1, "norm_text"]  # same text as the gold sentence
    pool = pd.concat([pool, clone], ignore_index=True)
    tables = build_tables(pool, gold_ids={"d1-001"})
    assert "other-001" not in set(tables["tone"]["sentence_id"])


def test_splits_never_overlap_in_time():
    tables = build_tables(make_pool(), gold_ids=set())
    by_split = tables["tone"].merge(make_pool()[["sentence_id", "year"]], on="sentence_id")
    assert by_split.loc[by_split["split"] == "train", "year"].max() <= 2018
    assert by_split.loc[by_split["split"] == "validation", "year"].between(2019, 2021).all()
    assert by_split.loc[by_split["split"] == "test", "year"].min() >= 2022


def test_the_topic_table_drops_sentences_with_no_topic():
    tables = build_tables(make_pool(), gold_ids=set())
    assert len(tables["tone"]) == 9
    assert len(tables["topic"]) == 8
    assert "d0-001" not in set(tables["topic"]["sentence_id"])


def test_tables_have_the_columns_the_notebook_expects():
    tables = build_tables(make_pool(), gold_ids=set())
    for table in tables.values():
        assert list(table.columns) == ["sentence_id", "doc_id", "text", "label", "split"]
    assert "tone: 9 rows" in summary(tables)
