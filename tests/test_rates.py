from datetime import date

import pytest

from cb_policy_tone.rates import parse_meeting_dates


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
    