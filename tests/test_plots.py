import numpy as np
import pandas as pd
import pytest

from cb_policy_tone import plots


@pytest.fixture
def data():
    n = 40
    rng = np.random.default_rng(1)
    rates = pd.DataFrame(
        {
            "meeting_no": np.arange(1, n + 1),
            "meeting_type": "regular",
            "meeting_end": pd.date_range("2010-01-31", periods=n, freq="2ME").strftime(
                "%Y-%m-%d"
            ),
            "mpr": 15 + np.cumsum(rng.choice([-1, 0, 1], n)),
            "mpr_change_bp": rng.choice([-100, 0, 100], n),
        }
    )
    index = pd.DataFrame(
        {
            "meeting_no": np.arange(1, n + 1),
            "meeting_type": "regular",
            "lexicon_net": rng.normal(size=n),
            "model_net": rng.normal(size=n),
            "n_scored": 50,
        }
    )
    inflation = pd.DataFrame(
        {
            "meeting_no": np.arange(1, n + 1),
            "meeting_type": "regular",
            "infl": rng.uniform(5, 20, n),
        }
    )
    return index, rates, inflation


def test_zscore_has_mean_zero_and_unit_sd():
    z = plots.zscore(pd.Series([1.0, 2.0, 3.0, 10.0]))
    assert z.mean() == pytest.approx(0)
    assert z.std(ddof=0) == pytest.approx(1)


def test_zscore_constant_series_is_zero():
    assert (plots.zscore(pd.Series([2.0, 2.0, 2.0])) == 0).all()


def test_overlay_has_three_panels(data):
    fig = plots.overlay(*data)
    assert len(fig.axes) == 3


def test_overlay_works_without_inflation(data):
    index, rates, _ = data
    assert len(plots.overlay(index, rates, None).axes) == 3


def test_tone_by_next_decision_has_one_panel_per_measure(data):
    index, rates, inflation = data
    frame = plots.build_frame(index, rates, inflation)
    assert len(plots.tone_by_next_decision(frame).axes) == 2
