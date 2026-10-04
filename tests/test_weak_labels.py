import re
from types import SimpleNamespace

import pandas as pd
import pytest

from cb_policy_tone.weak_labels import (
    ToneRule,
    is_labellable,
    label_sentences,
    load_tone_rules,
    load_topic_rules,
    rule_fires,
    score_tone,
    score_topic,
)


@pytest.fixture(scope="module")
def tone_rules():
    return load_tone_rules()


@pytest.fixture(scope="module")
def topic_rules():
    return load_topic_rules()


# ---- the rule files themselves ----


def test_rule_files_load_and_ids_are_unique(tone_rules, topic_rules):
    assert len({r.rule_id for r in tone_rules}) == len(tone_rules) >= 30
    assert len(topic_rules) >= 30


def test_every_judgment_call_in_the_codebook_has_a_rule(tone_rules):
    calls = {r.call for r in tone_rules if r.call}
    assert {"1", "2", "3", "4", "5", "6"} <= calls


def test_no_rule_exists_for_market_yields_or_bank_soundness(tone_rules):
    """Calls 7 and 8 default to neutral, so nothing may fire on these sentences."""
    for text in ["Treasury bill yields rose across the curve.", "Non-performing loans declined."]:
        assert score_tone(text, tone_rules)["label"] == "neutral"


# ---- how a rule fires ----


def make_rule(**overrides):
    base = {
        "rule_id": "X", "label": "hawkish", "call": "", "subject": re.compile("inflation", re.IGNORECASE),
        "move": re.compile(r"\brose\b", re.IGNORECASE), "require": None, "window": 3, "note": "",
    }  # fmt: skip
    base.update(overrides)
    return ToneRule(**base)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Inflation rose in March.", True),
        ("Prices rose, and inflation was discussed.", True),  # move before subject, 1 word apart
        ("Inflation in the first quarter of the year rose.", False),  # 6 words apart
        ("Growth rose in March.", False),  # no subject
    ],
)
def test_subject_and_move_must_be_close(text, expected):
    assert rule_fires(make_rule(), text) is expected


def test_a_negated_move_does_not_fire():
    assert not rule_fires(make_rule(), "Inflation did not rose.")  # 'not' just before the move
    assert not rule_fires(make_rule(), "There was no sign inflation rose.")


def test_require_pattern_must_also_match():
    rule = make_rule(
        subject=None,
        move=re.compile(r"upside risks", re.IGNORECASE),
        require=re.compile("inflation", re.IGNORECASE),
    )
    assert rule_fires(rule, "There are upside risks to inflation.")
    assert not rule_fires(rule, "There are upside risks to growth.")


def test_equal_signals_in_both_directions_give_neutral(tone_rules):
    text = "Inflation rose in March, but the global economy is expected to slow."
    result = score_tone(text, tone_rules)
    assert result["hawk"] and result["dove"]
    assert result["label"] == "neutral"


# ---- tone on realistic sentences (real ones from the corpus and illustrative ones) ----

TONE_CASES = [
    ("The Committee notes that challenges in the Eurozone continue to pose threats to the global economic outlook.", "dovish"),
    ("GDP growth is forecasted to decline to 5.0 percent in a baseline scenario.", "dovish"),
    ("In the worst case scenario, GDP growth estimates could be halved to about 2.5 percent in 2020.", "dovish"),
    ("The cedi has appreciated significantly, gaining 42.6 percent against the US dollar.", "dovish"),
    ("Gross international reserves improved to US$11.1 billion, equivalent to 4.8 months of import cover.", "dovish"),
    ("There were higher receipts from gold, cocoa beans and crude oil exports as commodity prices increased.", "dovish"),
    ("Export receipts from gold amounted to US$2.7 billion, while crude oil was US$1.2 billion.", "neutral"),
    ("Inflation rose further above the upper band of the target.", "hawkish"),
    ("The Committee remains vigilant to upside risks to the inflation outlook.", "hawkish"),
    ("Pressures on the cedi intensified during the quarter.", "hawkish"),
    ("Pressures on the cedi eased in the third quarter.", "dovish"),
    ("The US dollar strengthened against the cedi.", "hawkish"),
    ("Output in the mining sector contracted in the second quarter.", "dovish"),
    ("Real GDP growth was 6.0 percent in 2019, up from 5.8 percent.", "neutral"),
    ("Strong growth added to demand pressures, and excess demand persisted.", "hawkish"),
    ("Inflation fell to 11.6 percent in January.", "dovish"),
    ("Inflation remains above the target band.", "hawkish"),
    ("Inflation expectations remain anchored.", "dovish"),
    ("Inflationary pressures have eased.", "dovish"),
    ("The disinflation process continues.", "dovish"),
    ("Headline inflation was 22.8 percent in March.", "neutral"),
    ("The Committee did not see inflation rising.", "neutral"),
    ("Broad money growth accelerated, adding to liquidity pressures.", "hawkish"),
    ("Private sector credit growth slowed in the quarter.", "dovish"),
    ("Lending rates fell.", "neutral"),
    ("The fiscal deficit widened beyond the target.", "hawkish"),
    ("The overall budget deficit narrowed compared with the same period last year.", "dovish"),
    ("Revenue fell short of the target by 12.6 percent.", "hawkish"),
    ("Government borrowing from the central bank increased.", "hawkish"),
    ("Oil prices increased sharply in the quarter.", "hawkish"),
    ("The Fed raised its policy rate by 75 basis points.", "hawkish"),
    ("Central banks in advanced economies cut rates and eased financial conditions.", "dovish"),
    ("The global economy is expected to slow in 2024.", "dovish"),
    ("There is room to ease the policy stance.", "dovish"),
    ("The Committee will not hesitate to act to keep inflation within the target band.", "hawkish"),
    ("The Bank will continue to monitor developments and take appropriate action.", "neutral"),
    ("Good morning, Ladies and Gentlemen of the Media.", "neutral"),
]  # fmt: skip


@pytest.mark.parametrize(("text", "expected"), TONE_CASES)
def test_tone_on_realistic_sentences(tone_rules, text, expected):
    assert score_tone(text, tone_rules)["label"] == expected


# ---- topic ----

TOPIC_CASES = [
    ("The Committee notes that challenges in the Eurozone continue to pose threats to the global economic outlook.", "global"),
    ("GDP growth is forecasted to decline to 5.0 percent in a baseline scenario.", "growth"),
    ("The cedi has appreciated significantly against the US dollar.", "external"),
    ("Export receipts from gold amounted to US$2.7 billion, while crude oil was US$1.2 billion.", "external"),
    ("The fiscal deficit widened beyond the target.", "fiscal"),
    ("Treasury bill yields rose across the curve.", "financial"),
    ("Inflation fell to 11.6 percent in January.", "inflation"),
    ("The Fed raised its policy rate by 75 basis points.", "global"),
    ("Central banks in advanced economies cut rates and eased financial conditions.", "global"),
    ("The Committee will keep the cash reserve requirement under review.", "policy"),
    ("Thank you for coming.", "other"),
    ("Good morning, Ladies and Gentlemen of the Media.", "other"),
]  # fmt: skip


@pytest.mark.parametrize(("text", "expected"), TOPIC_CASES)
def test_topic_on_realistic_sentences(topic_rules, text, expected):
    assert score_topic(text, "", topic_rules)["topic"] == expected


def test_heading_hint_is_used_only_when_no_keyword_matches(topic_rules):
    vague = "The outcome was in line with expectations."
    assert score_topic(vague, "fiscal", topic_rules) == {
        "topic": "fiscal", "source": "heading", "hint_agrees": None,
    }  # fmt: skip
    assert score_topic(vague, "", topic_rules)["topic"] == ""


def test_keywords_beat_the_heading_when_they_disagree(topic_rules):
    result = score_topic("The cedi depreciated against the dollar.", "global", topic_rules)
    assert result["topic"] == "external"
    assert result["hint_agrees"] is False


# ---- the labellable pool ----


def row(**overrides):
    base = {
        "is_decision": False,
        "is_fragment": False,
        "is_short": False,
        "n_words": 20,
        "section_hint": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"is_decision": True}, False),
        ({"is_fragment": True}, False),
        ({"is_short": True}, False),
        ({"n_words": 81}, False),
        ({"n_words": 80}, True),
        ({"section_hint": "admin"}, False),
        ({"section_hint": "global"}, True),
    ],
)
def test_is_labellable(overrides, expected):
    assert is_labellable(row(**overrides)) is expected


def test_label_sentences_labels_only_the_labellable_pool(tone_rules, topic_rules):
    sentences = pd.DataFrame(
        {
            "sentence_id": ["a-001", "a-002"],
            "text": [
                "Inflation fell to 11.6 percent in January.",
                "The Committee decided to keep the policy rate.",
            ],
            "is_decision": [False, True],
            "is_fragment": [False, False],
            "is_short": [False, False],
            "n_words": [8, 8],
            "section_hint": ["", ""],
        }
    )
    out = label_sentences(sentences, tone_rules, topic_rules).set_index("sentence_id")
    assert out.loc["a-001", "tone_weak"] == "dovish"
    assert out.loc["a-001", "topic_weak"] == "inflation"
    assert bool(out.loc["a-002", "labellable"]) is False
    assert pd.isna(out.loc["a-002", "tone_weak"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Inflation rose in March, but growth slowed.", "neutral"),
        ("Growth slowed while inflation rose.", "neutral"),
        ("Inflation fell, but remains above the target band.", "neutral"),
    ],
)
def test_a_movement_word_belongs_to_the_nearest_subject(tone_rules, text, expected):
    assert score_tone(text, tone_rules)["label"] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The cedi depreciated, adding to inflation.", "external"),
        ("Inflation fell as the cedi stabilised.", "inflation"),
        (
            (
                "However, there are upside risks to the inflation outlook, including second round "
                "effects of exchange rate depreciation and utility tariff adjustments."
            ),
            "inflation",
        ),
        ("The Committee remains vigilant to upside risks to the inflation outlook.", "policy"),
        ("Let me welcome you to the Press briefing of the 51st meeting of the Committee.", "other"),
    ],
)
def test_topic_follows_the_first_strong_keyword(topic_rules, text, expected):
    assert score_topic(text, "", topic_rules)["topic"] == expected


def test_summed_weights_can_pick_the_wrong_topic(topic_rules):
    """The older method; kept so the report can compare both against the headings."""
    text = "Inflation fell as the cedi stabilised."
    assert score_topic(text, "", topic_rules, method="sum")["topic"] == "external"


def test_a_summary_and_outlook_heading_says_nothing_about_the_topic(topic_rules):
    vague = "The outcome was in line with expectations."
    assert score_topic(vague, "decision", topic_rules)["topic"] == ""
    assert score_topic(vague, "admin", topic_rules)["topic"] == ""
