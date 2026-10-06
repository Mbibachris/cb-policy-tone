import numpy as np
import pandas as pd
import pytest

from cb_policy_tone import validate as v


def synthetic(n: int = 160, slope: float = 1.3, seed: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tone at t drives the decision at t+1 when slope > 0."""
    rng = np.random.default_rng(seed)
    tone = rng.normal(size=n)
    direction = np.zeros(n)
    for t in range(1, n):
        latent = slope * tone[t - 1] + rng.normal()
        direction[t] = 1 if latent > 0.7 else -1 if latent < -0.7 else 0
    rates = pd.DataFrame(
        {
            "meeting_no": np.arange(1, n + 1),
            "meeting_type": "regular",
            "meeting_end": pd.date_range("2003-01-31", periods=n, freq="2ME").strftime("%Y-%m-%d"),
            "mpr_change_bp": [np.nan, *(direction[1:] * 100)],
        }
    )
    index = pd.DataFrame(
        {
            "meeting_no": np.arange(1, n + 1),
            "meeting_type": "regular",
            "lexicon_net": tone + rng.normal(scale=0.3, size=n),
            "model_net": tone + rng.normal(scale=0.2, size=n),
            "n_scored": 50,
        }
    )
    return index, rates


# ---- inflation taken from the statement text ----


@pytest.mark.parametrize(
    ("sentences", "expected"),
    [
        (["Inflation fell to 11.6 percent in January."], 11.6),
        (["The year 2002 ended with inflation at 15.2 percent, down from 21.3 per cent."], 15.2),
        (["Headline inflation rose to 22.8 percent in March 2024."], 22.8),
        (
            ["Consumer price inflation fell from 21.3 per cent in 2001 to 15.2 per cent in 2002."],
            15.2,
        ),
        (["The medium-term inflation target is 8 percent.", "Inflation was 9.4 percent."], 9.4),
        (["Core inflation rose to 10 percent.", "Inflation expectations remain anchored."], None),
        (["Growth was 5.0 percent in 2019."], None),
    ],
)
def test_extract_inflation(sentences, expected):
    assert v.extract_inflation(sentences) == expected


def test_overrides_replace_extracted_values():
    inflation = pd.DataFrame(
        {"doc_id": [1, 2], "meeting_no": [5, 6], "meeting_type": "regular", "infl": [10.0, np.nan]}
    )
    fixed = v.apply_overrides(inflation, pd.DataFrame({"meeting_no": [6], "infl": [12.5]}))
    assert fixed["infl"].tolist() == [10.0, 12.5]


# ---- the analysis frame ----


def test_frame_pairs_each_meetings_tone_with_the_next_decision():
    index, rates = synthetic(n=20)
    frame = v.build_frame(index, rates, None)
    first = frame.iloc[0]
    assert first["meeting_no"] == 2  # meeting 1 has no decision change
    expected_next = np.sign(rates.loc[rates["meeting_no"] == 3, "mpr_change_bp"].iloc[0])
    assert first["next_dir"] == expected_next
    assert frame["meeting_no"].max() == 19  # the last meeting has no next decision


# ---- the test itself ----


def test_a_tone_that_really_predicts_is_detected():
    index, rates = synthetic()
    frame = v.build_frame(index, rates, None)
    result = v.test_tone(frame, "model_net", ["dir"])
    assert result["coef"] > 0
    assert result["p_lr"] < 0.01
    assert result["aic_full"] < result["aic_base"]
    assert result["r2_full"] > result["r2_base"]


def test_a_tone_unrelated_to_the_next_decision_is_not_flagged():
    index, rates = synthetic(slope=0.0)
    frame = v.build_frame(index, rates, None)
    assert v.test_tone(frame, "model_net", ["dir"])["p_lr"] > 0.01


def test_predicted_probabilities_sum_to_one_and_hike_probability_rises_with_tone():
    index, rates = synthetic()
    frame = v.build_frame(index, rates, None)
    probs = v.hike_probabilities(v.test_tone(frame, "model_net", ["dir"]), ["dir"])
    for triple in probs.values():
        assert sum(triple) == pytest.approx(1.0)
    assert probs["tone +1 sd"][2] > probs["tone -1 sd"][2]


def test_out_of_sample_gain_is_positive_for_a_real_signal_and_absent_for_a_short_test_period():
    index, rates = synthetic(n=200)
    frame = v.build_frame(index, rates, None)
    assert v.out_of_sample_gain(frame, ["dir"], "model_net") > 0
    short = frame[frame["year"] <= v.SPLIT_YEAR + 1]
    assert v.out_of_sample_gain(short, ["dir"], "model_net") is None


def test_the_report_runs_with_and_without_an_inflation_control(capsys):
    index, rates = synthetic()
    inflation = pd.DataFrame(
        {
            "meeting_no": np.arange(1, 161),
            "meeting_type": "regular",
            "infl": np.random.default_rng(0).normal(15, 4, 160),
        }
    )
    v.report(v.build_frame(index, rates, inflation))
    assert "momentum + inflation" in capsys.readouterr().out
    v.report(v.build_frame(index, rates, None))
    assert "momentum + inflation" not in capsys.readouterr().out
