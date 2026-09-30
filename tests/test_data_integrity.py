"""Checks on the committed processed data, so a bad regeneration fails CI."""

from pathlib import Path

import pandas as pd
import pytest

RATES = Path("data/processed/policy_rates.csv")


@pytest.fixture(scope="module")
def rates():
    return pd.read_csv(RATES, parse_dates=["meeting_end"])


def test_keys_are_unique(rates):
    assert not rates[["meeting_no", "meeting_type"]].duplicated().any()


def test_meetings_are_in_date_order(rates):
    assert rates["meeting_end"].is_monotonic_increasing


def test_regular_meetings_are_numbered_1_to_132(rates):
    regular = rates[rates["meeting_type"] == "regular"]
    assert regular["meeting_no"].tolist() == list(range(1, 133))


def test_two_emergency_meetings(rates):
    emergency = rates[rates["meeting_type"] == "emergency"]
    assert emergency["meeting_end"].dt.strftime("%Y-%m").tolist() == ["2014-02", "2022-08"]


@pytest.mark.parametrize(
    ("meeting_no", "expected_mpr"),
    [(13, 18.5), (29, 14.25), (60, 19.0), (65, 22.0), (81, 18.0), (132, 14.0)],
)
def test_corrected_and_latest_rates(rates, meeting_no, expected_mpr):
    row = rates[(rates["meeting_no"] == meeting_no) & (rates["meeting_type"] == "regular")]
    assert row["mpr"].iloc[0] == expected_mpr