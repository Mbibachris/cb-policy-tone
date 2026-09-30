from datetime import date

import pandas as pd
import pytest

from cb_policy_tone.parse import (
    choose_primaries,
    doc_type,
    letterhead_date,
    match_document,
    match_regular,
    rates_near_policy_rate,
    text_meeting_number,
    title_month,
)


@pytest.fixture
def meetings():
    rows = [
        (2, "regular", "2003-01-24", "2003-01-27"),
        (4, "regular", "2003-05-23", "2003-05-26"),
        (6, "regular", "2003-10-10", "2003-10-13"),
        (35, "regular", "2009-07-17", "2009-07-21"),
        (59, "regular", "2014-04-02", "2014-04-02"),
        (60, "regular", "2014-07-09", "2014-07-09"),
        (100, "regular", "2021-05-28", "2021-05-31"),
        (106, "regular", "2022-05-20", "2022-05-23"),
        (59, "emergency", "2014-02-06", "2014-02-06"),
    ]
    df = pd.DataFrame(rows, columns=["meeting_no", "meeting_type", "meeting_end", "effective_date"])
    df["meeting_end"] = pd.to_datetime(df["meeting_end"])
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    df["effective_date_flag"] = False
    return df


@pytest.fixture
def regular(meetings):
    return meetings[meetings["meeting_type"] == "regular"].reset_index(drop=True)


@pytest.fixture
def emergency(meetings):
    return meetings[meetings["meeting_type"] == "emergency"].reset_index(drop=True)


# ---- small parsing helpers ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BANK OF GHANA K O F G H Press Release Monetary Policy Committee May 23, 2003 E ST. 1957",
         date(2003, 5, 23)),
        ("Press Release B A Monetary Policy Committee E ST. 1957 December 18, 2006 1. The",
         date(2006, 12, 18)),
        ("Bank of Ghana Monetary Policy Committee Press Release May 31, 2021 Good morning",
         date(2021, 5, 31)),
        ("Good morning, Ladies and Gentlemen of the Media and welcome to the briefing", None),
        ("Monetary Policy Committee February 31, 2003 is not a real date", None),
        ("Monetary Policy Committee Press Release APRIL 2014 Let me welcome you", None),
    ],
)  # fmt: skip
def test_letterhead_date(text, expected):
    assert letterhead_date(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("welcome to the press briefing of the 41st meeting of the MPC", 41),
        ("Press briefing after the 106th Monetary Policy Committee meetings", 106),
        ("HIGHLIGHTS PRESS RELEASE 105TH MPC MEETING MARCH 2022", 105),
        ("the bank celebrated its 25th anniversary last year", None),
        ("Good morning, everyone", None),
    ],
)
def test_text_meeting_number(text, expected):
    assert text_meeting_number(text) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("MPC Press Release – May 2018", (2018, 5)),
        ("MPC Press Release – Jan. 2018", (2018, 1)),
        ("MPC Press Release – MAY 2020", (2020, 5)),
        ("MPC Press Release  -April 2010", (2010, 4)),
        ("Emergency MPC Press Release – July 18, 2025", None),
        ("Highlights:105th MPC Meeting Press Release", None),
    ],
)
def test_title_month(title, expected):
    assert title_month(title) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("MPC Press Release – May 2018", "press_release"),
        ("MPC Press Release  -April 2010", "press_release"),
        ("Emergency MPC Press Release – August 2022", "emergency"),
        ("Highlights:105th MPC Meeting Press Release", "highlights"),
        ("Transcript of MPC Press Briefing Held on March 21, 2022", "transcript"),
        ("Notice of 103rd Monetary Policy Committee Meetings", "notice"),
        ("MPC Meeting – January 2021", "meeting_notice"),
        ("Real Sector Charts – MPC July 2021", "support_data"),
        ("MPC Infographics – July 2018", "support_data"),
    ],
)
def test_doc_type(title, expected):
    assert doc_type(title) == expected


def test_rates_near_policy_rate_finds_old_and_new_rate():
    text = "The Committee decided to increase the Prime Rate from 13.5 percent to 14.25 percent."
    assert rates_near_policy_rate(text) == {13.5, 14.25}


def test_rates_ignore_figures_far_from_a_policy_rate_mention():
    text = "Inflation rose to 14.0 percent. " + "x " * 200 + "The policy rate is 18 percent."
    assert rates_near_policy_rate(text) == {18.0}


# ---- matching a document to a meeting ----


def test_letterhead_date_places_the_document(regular):
    text = "Monetary Policy Committee May 23, 2003 E ST. 1957 1. The Committee"
    m = match_regular(text, "MPC Press Release – May 2003", regular)
    assert (m["meeting_no"], m["match_method"], m["conflict"]) == (4, "letterhead_date", False)


def test_mislabelled_title_loses_to_the_letterhead_date(regular):
    text = "Monetary Policy Committee July 21, 2009 E ST.1957 1. Let me welcome you"
    m = match_regular(text, "MPC Press Release – July 2007", regular)
    assert m["meeting_no"] == 35
    assert m["title_ok"] is False


def test_letterhead_date_near_no_meeting_is_not_placed_by_weaker_clues(regular):
    # the title month fits meeting 6, but the letterhead date fits nothing
    text = "Monetary Policy Committee October 10, 2001 E ST. 1957 1. The prime rate"
    m = match_regular(text, "MPC Press Release – October 2003", regular)
    assert m["meeting_no"] is None
    assert m["match_method"] == "letterhead_no_meeting"


def test_number_and_title_place_a_briefing_without_a_date(regular):
    text = (
        "Good morning and welcome to the press briefing after the 106th Monetary Policy Committee"
    )
    m = match_regular(text, "MPC Press Release – May 2022", regular)
    assert (m["meeting_no"], m["match_method"]) == (106, "text_number")


def test_title_alone_is_the_last_resort(regular):
    text = "Members of the Press, let me welcome you to the briefing"
    m = match_regular(text, "MPC Press Release – May 2022", regular)
    assert (m["meeting_no"], m["match_method"]) == (106, "title")


def test_disagreeing_clues_are_flagged(regular):
    text = "Good morning. This was the 100th meeting of the Monetary Policy Committee."
    m = match_regular(text, "MPC Press Release – May 2022", regular)
    assert m["meeting_no"] == 106
    assert m["conflict"] is True


def test_february_statement_falls_back_to_the_emergency_meeting(regular, emergency):
    text = "Monetary Policy Committee February 6, 2014 E ST. 1957 1. At an emergency meeting"
    m = match_document(
        text, "MPC Press Release – February 2014", "press_release", regular, emergency
    )
    assert (m["meeting_no"], m["meeting_type"]) == (59, "emergency")


def test_july_2014_statement_matches_the_added_meeting(regular, emergency):
    text = "Monetary Policy Committee Press Release July 9, 2014 You are welcome. Its 60th meeting"
    m = match_document(text, "MPC Press Release – July 2014", "press_release", regular, emergency)
    assert (m["meeting_no"], m["meeting_type"]) == (60, "regular")


def test_emergency_release_without_a_rate_row_stays_unplaced(regular, emergency):
    text = (
        "PUBLIC BANK OF GHANA MONETARY POLICY COMMITTEE PRESS RELEASE July 18, 2025 The Committee"
    )
    m = match_document(
        text, "Emergency MPC Press Release – July 18, 2025", "emergency", regular, emergency
    )
    assert m["meeting_no"] is None
    assert m["match_method"] == "no_rate_row"


# ---- choosing one document per meeting ----


def doc(doc_id, *, kind="press_release", text_hash="a", n_chars=1000, title_ok=True):
    return {
        "doc_id": doc_id, "doc_type": kind, "role": "candidate", "meeting_type": "regular",
        "meeting_no": 105, "text_hash": text_hash, "n_chars": n_chars, "title_ok": title_ok,
        "duplicate_of": "", "text_identical": "",
    }  # fmt: skip


def roles(docs):
    return {d["doc_id"]: d["role"] for d in docs}


def test_identical_copies_become_duplicates_and_the_lower_id_wins():
    docs = [doc("20"), doc("10")]
    choose_primaries(docs)
    assert roles(docs) == {"10": "primary", "20": "duplicate"}


def test_the_copy_with_the_right_title_wins_over_a_lower_id():
    docs = [doc("10", title_ok=False), doc("20")]
    choose_primaries(docs)
    assert roles(docs) == {"10": "duplicate", "20": "primary"}


def test_near_identical_text_counts_as_a_duplicate():
    docs = [doc("1", text_hash="a", n_chars=4575), doc("2", text_hash="b", n_chars=4571)]
    choose_primaries(docs)
    assert roles(docs) == {"1": "primary", "2": "duplicate"}


def test_clearly_different_text_is_a_conflicting_version():
    docs = [doc("1", text_hash="a", n_chars=21000), doc("2", text_hash="b", n_chars=9000)]
    choose_primaries(docs)
    assert roles(docs) == {"1": "primary", "2": "conflicting_version"}


def test_highlights_never_replace_the_full_statement():
    docs = [doc("1", kind="highlights", text_hash="h", n_chars=9000), doc("2")]
    choose_primaries(docs)
    assert roles(docs) == {"1": "summary_of_primary", "2": "primary"}