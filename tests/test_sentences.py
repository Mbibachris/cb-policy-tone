import pytest

from cb_policy_tone.sentences import (
    build_paragraphs,
    clean_lines,
    drop_letterhead_debris,
    drop_repeated_lines,
    is_decision_sentence,
    is_noise_line,
    looks_like_heading,
    section_hint,
    split_sentences,
    split_statement,
    strip_letterhead,
)

OLD_LETTERHEAD = """BANK OF GHANA
K O F G H
Press Release
N A A N B A
Monetary Policy Committee May 23, 2003 E ST. 1957
1. Ladies and Gentlemen, welcome to the briefing."""


# ---- cleaning ----


def test_old_letterhead_is_removed():
    out = strip_letterhead(OLD_LETTERHEAD)
    assert out.strip().startswith("1. Ladies and Gentlemen")


def test_letter_spaced_letterhead_without_a_day_is_removed():
    text = (
        "r e N s K s O F R G e H l ease A A B N A Bank of Ghana E S T . 1 9 5 7 Monetary "
        "Policy Committee Press Release APRIL 2014 Let me welcome you all."
    )
    assert strip_letterhead(text).strip() == "Let me welcome you all."


def test_text_without_a_letterhead_is_untouched():
    text = "Good morning, Ladies and Gentlemen of the Media, and welcome to the briefing."
    assert strip_letterhead(text) == text


def test_a_real_sentence_before_committee_and_a_month_is_not_mistaken_for_a_letterhead():
    text = "Welcome all. The Monetary Policy Committee May 2022 meeting started on time."
    assert strip_letterhead(text) == text


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("11", True),
        ("Page 3 of 10", True),
        ("K O F G H", True),
        ("", True),
        ("The Committee decided to keep the rate.", False),
        ("1. Inflation fell.", False),
    ],
)
def test_is_noise_line(line, expected):
    assert is_noise_line(line) is expected


def test_repeated_header_lines_are_dropped_but_repeated_sentence_ends_are_kept():
    lines = [
        "Bank of Ghana", "First sentence runs on to", "per cent.",
        "Bank of Ghana", "Second sentence runs on to", "per cent.",
        "Bank of Ghana", "Third sentence runs on to", "per cent.",
    ]  # fmt: skip
    out = drop_repeated_lines(lines)
    assert "Bank of Ghana" not in out
    assert out.count("per cent.") == 3


def test_clean_lines_removes_page_numbers_and_noise():
    text = OLD_LETTERHEAD + "\nInflation fell in January.\n11\nIt rose again in February.\n"
    lines = clean_lines(text)
    assert "11" not in lines
    assert all("K O F G H" not in line for line in lines)


# ---- headings and paragraphs ----


def test_heading_between_two_sentences_is_recognised():
    assert looks_like_heading("Global Developments", "It ended well.", "1. The global economy")


@pytest.mark.parametrize(
    ("line", "prev", "nxt"),
    [
        ("Global Developments", "It continued to", "1. The global economy"),  # mid-sentence
        ("Global Developments", "It ended well.", "and then continued"),  # next line is lowercase
        ("the global economy grew", "It ended well.", "Next sentence."),  # lowercase start
        ("The global economy grew.", "It ended well.", "Next sentence."),  # full sentence
        ("Global Developments", "It ended well.", None),  # nothing follows
    ],
)
def test_lines_that_are_not_headings(line, prev, nxt):
    assert not looks_like_heading(line, prev, nxt)


def test_consecutive_headings_are_combined():
    lines = [
        "The committee met.",
        "Additional Policy Measures",
        "A. Budget Financing",
        "12. The pandemic strained the budget.",
    ]
    paragraphs = build_paragraphs(lines)
    assert paragraphs[-1] == (
        "Additional Policy Measures / Budget Financing",
        "The pandemic strained the budget.",
    )


def test_heading_on_the_same_line_as_a_numbered_paragraph_is_split_off():
    lines = ["Global Developments 1. The global economy continued its gradual recovery."]
    assert build_paragraphs(lines) == [
        ("Global Developments", "The global economy continued its gradual recovery.")
    ]


def test_numbers_and_bullets_start_new_paragraphs_and_wrapped_lines_are_joined():
    lines = [
        "1. Inflation fell in January on a year-on-",
        "year basis.",
        "2. The rate was unchanged",
        "at 14.5 percent.",
        "• The 91-day bill rose.",
        "• The 1-year note fell.",
    ]
    texts = [p for _, p in build_paragraphs(lines)]
    assert texts == [
        "Inflation fell in January on a year-on-year basis.",
        "The rate was unchanged at 14.5 percent.",
        "The 91-day bill rose.",
        "The 1-year note fell.",
    ]


# ---- sentence splitting ----


def test_splits_after_a_percentage_but_not_inside_a_decimal():
    out = split_sentences("Inflation rose to 14.5 percent. The Committee met in March.")
    assert out == ["Inflation rose to 14.5 percent.", "The Committee met in March."]


@pytest.mark.parametrize(
    "text",
    [
        "The U.S. Treasury also introduced various measures.",
        "Dr. Ernest A. Addison spoke to the press.",
        "The Bank will provide GH¢5.5 billion to the sector.",
        "Growth slowed, e.g. in mining, during the quarter.",
    ],
)
def test_abbreviations_and_amounts_do_not_end_sentences(text):
    assert split_sentences(text) == [text]


def test_leftover_paragraph_numbers_inside_text_are_removed():
    out = split_sentences("Output fell in March. 12. The Committee noted the decline.")
    assert out == ["Output fell in March.", "The Committee noted the decline."]


def test_quotes_and_brackets_after_a_full_stop_still_split():
    out = split_sentences("He said the risks were balanced.) The Committee agreed.")
    assert out == ["He said the risks were balanced.)", "The Committee agreed."]


# ---- decision sentences ----


@pytest.mark.parametrize(
    "text",
    [
        "The Committee decided to maintain the Bank of Ghana Prime Rate at 18.5 percent.",
        (
            "In view of these developments, the Committee decided to increase the policy rate by "
            "100 basis points to 19 percent."
        ),
        "This is consistent with the decision of the Committee to keep the policy rate unchanged.",
    ],
)
def test_decision_sentences_are_flagged(text):
    assert is_decision_sentence(text)


@pytest.mark.parametrize(
    "text",
    [
        "The U.S. Fed cut its policy rate by 150 basis points in March 2020.",
        "The ECB decided to keep its policy rate unchanged.",
        "The increase in the policy rate slowed the pace of depreciation.",
        "The Committee decided to raise the cash reserve requirement to 11 percent.",
        "Inflation rose to 14.5 percent in March.",
    ],
)
def test_other_sentences_are_not_flagged(text):
    assert not is_decision_sentence(text)


# ---- section hints ----


@pytest.mark.parametrize(
    ("heading", "hint"),
    [
        ("Global Developments", "global"),
        ("Inflation Outlook and Analysis", "inflation"),
        ("Real Sector Developments", "growth"),
        ("Fiscal Developments", "fiscal"),
        ("Budget Financing", "fiscal"),
        ("Banking Sector Developments", "financial"),
        ("Monetary and Financial Developments", "financial"),
        ("Additional Policy Measures", ""),
        ("", ""),
    ],
)
def test_section_hint(heading, hint):
    assert section_hint(heading) == hint


# ---- whole statements ----


STATEMENT = """Bank of Ghana Monetary Policy Committee Press Release May 31, 2021
Good morning, Ladies and Gentlemen of the Media, welcome to this Press Conference.
Global Developments
1. The global economy continued its gradual recovery, despite some moderation.
Growth is expected to pick up.
Decision
2. In view of these developments, the Committee decided to increase the policy rate by 100
basis points to 19 percent.
Thank you for your attention.
14
"""


def test_a_whole_statement_is_split_with_headings_and_decision_flags():
    rows = split_statement(STATEMENT)
    assert [r["position"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["section_heading"] for r in rows] == [
        "",
        "Global Developments",
        "Global Developments",
        "Decision",
        "Decision",
    ]
    assert [r["is_decision"] for r in rows] == [False, False, False, True, False]
    assert rows[3]["section_hint"] == "decision"
    assert all(r["text"][0].isupper() for r in rows)


def test_short_sentences_are_flagged():
    rows = split_statement("Thank you.\nThe Committee met on Monday to review developments.\n")
    assert [r["is_short"] for r in rows] == [True, False]

# ---- fixes from the first run on the real statements ----


def test_cid_glyph_and_middle_dot_bullets_start_new_paragraphs():
    lines = [
        "Rates eased as follows:",
        "(cid:131) The 91-day bill rose to 18.5 per cent.",
        "· The 1-year note fell.",
    ]
    texts = [p for _, p in build_paragraphs(lines)]
    assert texts == [
        "Rates eased as follows:",
        "The 91-day bill rose to 18.5 per cent.",
        "The 1-year note fell.",
    ]


def test_bullets_in_the_middle_of_a_line_split_the_text():
    text = "Data indicate the following: · On a narrow basis the deficit widened. · Revenue rose."
    assert split_sentences(text) == [
        "Data indicate the following:",
        "On a narrow basis the deficit widened.",
        "Revenue rose.",
    ]


@pytest.mark.parametrize(
    "line", ["BANK OF GHANA", "MONETARY POLICY COMMITTEE PRESS RELEASE", "PUBLIC"]
)
def test_letterhead_words_alone_on_a_line_are_noise(line):
    assert is_noise_line(line)


@pytest.mark.parametrize(
    ("heading", "hint"),
    [
        ("Informational Note", "admin"),
        ("Information Note", "admin"),
        ("Summary and Outlook", "decision"),
    ],
)
def test_more_section_hints(heading, hint):
    assert section_hint(heading) == hint


def test_lowercase_starts_are_flagged_as_fragments():
    rows = split_statement("• increased by 32.6 percent, well above the level a year earlier.\n")
    assert [r["is_fragment"] for r in rows] == [True]

# ---- second round of fixes ----


@pytest.mark.parametrize("line", ["KOFGH", "E 7", "ST .195", "THANK YOU", "END.", "N"])
def test_short_all_capital_fragments_are_noise(line):
    assert is_noise_line(line)


def test_letterhead_leftovers_in_the_first_lines_are_dropped():
    text = (
        "F\nFor Immediate Release Date: 22 May 2017\nN\nKOFGH\nEST.1957\n"
        "Global Developments\n1. The global economy grew.\n"
    )
    assert clean_lines(text) == ["Global Developments", "1. The global economy grew."]


def test_the_same_words_later_in_the_text_are_kept():
    lines = ["Line"] * 15 + ["The Bank of Ghana raised its forecast."]
    assert drop_letterhead_debris(lines)[-1] == "The Bank of Ghana raised its forecast."


@pytest.mark.parametrize(
    "text",
    [
        (
            "The Committee judged the risks to be balanced and therefore maintained the monetary "
            "policy rate at 21 percent."
        ),
        "The Committee maintained the policy rate at 14.5 percent.",
        "The MPC therefore raised the policy rate by 100 basis points to 19 percent.",
    ],
)
def test_decisions_without_a_decision_word_are_flagged(text):
    assert is_decision_sentence(text)


@pytest.mark.parametrize(
    "text",
    [
        "The Committee noted that the Fed raised its policy rate in March.",
        "Average yields rose therefore the policy rate corridor widened.",
    ],
)
def test_reports_about_other_things_are_still_not_flagged(text):
    assert not is_decision_sentence(text)    


# ---- third round of fixes ----


def test_a_first_body_line_that_names_the_bank_is_not_mistaken_for_letterhead():
    lines = ["The Monetary Policy Committee of the Bank of Ghana has", "undertaken a review."]
    assert drop_letterhead_debris(lines) == lines


@pytest.mark.parametrize(
    ("line", "is_debris"),
    [
        ("B A Bank of Ghana", True),
        ("A Bank of Ghana official said", False),
        ("Press Statement No 121", True),
        ("EST.1957", True),
        ("The Committee met on Monday.", False),
    ],
)
def test_letterhead_debris_rule(line, is_debris):
    assert (line in drop_letterhead_debris([line])) is not is_debris


@pytest.mark.parametrize(
    "text",
    [
        (
            "In the circumstances, the Monetary Policy Committee has decided to reduce the rate "
            "by 50 basis points from 18.5 percent to 18.0percent."
        ),
        "Increase the monetary policy rate from 19 percent to 21 percent to keep policy tight.",
        "The Committee decided to maintain the current tight policy stance.",
    ],
)
def test_more_decision_phrasings_are_flagged(text):
    assert is_decision_sentence(text)


def test_setting_a_corridor_is_not_the_rate_decision():
    text = "Set the interest rate corridor at 300 basis points around the monetary policy rate."
    assert not is_decision_sentence(text)