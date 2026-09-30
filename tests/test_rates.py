from datetime import date

import pandas as pd
import pytest

from cb_policy_tone.rates import clean, parse_meeting_dates


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("November 20 – 21, 2002", (date(2002, 11, 20), date(2002, 11, 21), False)),
        ("August 21 -24, 2007", (date(2007, 8, 21), date(2007, 8, 24), False)),
        ("August 29 – September 1, 2011", (date(2011, 8, 29), date(2011, 9, 1), False)),
        ("August 17, 2022 (Emergency)", (date(2022, 8, 17), date(2022, 8, 17), True)),
    ],
)
def test_parse_meeting_dates(text, expected):
    assert parse_meeting_dates(text) == expected


@pytest.fixture
def raw_table():
    return pd.DataFrame(
        {
            "No": [12, 13, 14],
            "MPC Dates": ["January 17 – 21, 2005", "March 22 – 24, 2005", "May 24 – 27, 2005"],
            "Effective": ["24 Jan 2005", "29 Mar 2005", "30 May 2005"],
            "Rate": [18.5, 15.5, 16.5],
        }
    )


def correction(**overrides):
    row = {
        "meeting_no": "13", "meeting_type": "regular", "field": "mpr",
        "old_value": "15.5", "new_value": "18.5",
    }  # fmt: skip
    row.update(overrides)
    return pd.DataFrame([row])


def test_changes_are_computed_from_the_rate_series(raw_table):
    out = clean(raw_table)
    assert out["mpr_change_bp"].tolist()[1:] == [-300, 100]


def test_correction_changes_value_and_downstream_change(raw_table):
    out = clean(raw_table, corrections=correction())
    assert out["mpr"].tolist() == [18.5, 18.5, 16.5]
    assert out["mpr_change_bp"].tolist()[1:] == [0, -200]
    assert out["corrected"].tolist() == [False, True, False]


def test_correction_with_wrong_old_value_is_rejected(raw_table):
    with pytest.raises(ValueError, match="expected mpr"):
        clean(raw_table, corrections=correction(old_value="99"))


def test_correction_for_unknown_meeting_is_rejected(raw_table):
    with pytest.raises(ValueError, match="targets 0 rows"):
        clean(raw_table, corrections=correction(meeting_no="77"))


def test_correction_can_reclassify_a_meeting(raw_table):
    fix = correction(field="meeting_type", old_value="regular", new_value="emergency")
    out = clean(raw_table, corrections=fix)
    assert out["meeting_type"].tolist() == ["regular", "emergency", "regular"]


def test_added_meeting_is_sorted_into_place_and_flagged(raw_table):
    added = pd.DataFrame(
        [
            {
                "meeting_no": "15",
                "meeting_type": "regular",
                "meeting_start": "",
                "meeting_end": "2005-07-29",
                "mpr": "16.5",
            }
        ]
    )
    out = clean(raw_table, additions=added)
    assert out["meeting_no"].tolist() == [12, 13, 14, 15]
    assert out["corrected"].tolist() == [False, False, False, True]
    assert out["decision"].tolist()[-1] == "hold"


def test_added_meeting_that_already_exists_is_rejected(raw_table):
    added = pd.DataFrame(
        [
            {
                "meeting_no": "13",
                "meeting_type": "regular",
                "meeting_start": "",
                "meeting_end": "2005-03-24",
                "mpr": "15.5",
            }
        ]
    )
    with pytest.raises(ValueError, match="already exists"):
        clean(raw_table, additions=added)

def test_correction_can_change_a_meeting_date(raw_table):
    fix = correction(field="meeting_end", old_value="2005-03-24", new_value="2005-03-20")
    out = clean(raw_table, corrections=fix)
    assert out.loc[out["meeting_no"] == 13, "meeting_end"].iloc[0] == pd.Timestamp("2005-03-20")
    assert out["corrected"].tolist() == [False, True, False]


def test_date_correction_with_wrong_old_date_is_rejected(raw_table):
    fix = correction(field="meeting_end", old_value="2005-03-25", new_value="2005-03-20")
    with pytest.raises(ValueError, match="expected meeting_end"):
        clean(raw_table, corrections=fix)        