import pandas as pd
import pytest

from cb_policy_tone.tone_index import build_index


@pytest.fixture
def inputs():
    sentences = pd.DataFrame(
        {"sentence_id": ["a-1", "a-2", "a-3", "a-4", "b-1", "b-2"], "doc_id": [1, 1, 1, 1, 2, 2]}
    )
    weak = pd.DataFrame(
        {
            "sentence_id": ["a-1", "a-2", "a-3", "a-4", "b-1", "b-2"],
            "labellable": [True, True, True, False, True, True],
            "tone_weak": ["hawkish", "hawkish", "neutral", None, "dovish", "dovish"],
        }
    )
    predictions = pd.DataFrame(
        {
            "sentence_id": ["a-1", "a-2", "a-3", "a-4", "b-1", "b-2"],
            "p_dovish": [0.1, 0.2, 0.3, 0.9, 0.7, 0.6],
            "p_neutral": [0.2, 0.2, 0.4, 0.05, 0.2, 0.2],
            "p_hawkish": [0.7, 0.6, 0.3, 0.05, 0.1, 0.2],
        }
    )
    statements = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "meeting_no": [10, 11],
            "meeting_type": ["regular", "regular"],
            "meeting_end": ["2019-03-01", None],
            "letterhead_date": [None, "2019-05-20"],
        }
    )
    return sentences, weak, predictions, statements


def test_index_averages_only_the_scored_sentences(inputs):
    index = build_index(*inputs).set_index("doc_id")
    assert index.loc[1, "n_scored"] == 3  # the non-labellable sentence is left out
    assert index.loc[1, "lexicon_net"] == pytest.approx((1 + 1 + 0) / 3)
    assert index.loc[1, "model_net"] == pytest.approx(((0.7 - 0.1) + (0.6 - 0.2) + (0.3 - 0.3)) / 3)
    assert index.loc[2, "lexicon_net"] == -1


def test_hard_index_counts_the_most_probable_class(inputs):
    index = build_index(*inputs).set_index("doc_id")
    assert index.loc[1, "model_net_hard"] == pytest.approx((1 + 1 + 0) / 3)  # a-3 is a 0.4 neutral
    assert index.loc[2, "model_net_hard"] == -1


def test_the_date_falls_back_to_the_letterhead_date_and_rows_are_in_date_order(inputs):
    index = build_index(*inputs)
    assert index["doc_id"].tolist() == [1, 2]
    assert str(index.loc[1, "date"].date()) == "2019-05-20"
