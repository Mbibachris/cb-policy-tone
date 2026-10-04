# Codebook: tone and topic labels for Bank of Ghana MPC statements

**Version 0.1 (draft).** Author: Christopher Mbiba. Project: `cb-policy-tone`.
Status: eight judgment calls (marked **[CALL n]** in section 3) are defaults awaiting confirmation. Any change to them is recorded in the change log at the end, and labels made under an older version are re-checked.

---

## 1. Purpose

Every sentence in the corpus gets two labels:

- **Tone**: does the sentence point towards *tighter* monetary policy (hawkish), *looser* monetary policy (dovish), or neither (neutral)?
- **Topic**: which area of the economy or policy is the sentence about?

The tone label is the target of the fine-tuned classifier. The topic label lets the tone index be split by what the Committee is talking about. These rules are written down so that the labels can be audited, repeated by someone else, and checked against the annotator's own later judgment.

The approach follows the logic of hawkish/dovish annotation for FOMC text (Shah, Paturi and Chava, 2023, "Trillion Dollar Words") and of dictionary-based tone measures for central-bank text. Both the rules below and the Bank's inflation-targeting framework (a medium-term target of 8 percent, plus or minus 2) shape the direction rules.

## 2. What is labelled

**Unit.** One sentence, as produced by `sentences.py`. The annotator sees the sentence and the section heading it sits under (if any). The previous and next sentences are *not* shown, because the classifier will also see one sentence at a time.

**Not labelled (excluded from the gold set and from training):**

| Excluded | Why |
|---|---|
| Decision sentences (`is_decision` is true) | They state the rate decision itself. Using them would make the tone index and the later validation against rate decisions circular. |
| Fragments (`is_fragment`), very short sentences (`is_short`), sentences over 80 words | Broken by text extraction, or too short to carry a signal. |
| Sentences under an "Information Note" heading (`section_hint` = admin) | Scheduling information about the next meeting. |

**Labelled neutral, topic `other`:** greetings, thank-yous, introductions of speakers, procedural statements ("The Committee met last week"), and anything with no economic content.

## 3. Tone

### 3.1 The principle

**Tone is about the policy implication of a sentence, not about whether the news is good or bad.**

- Rising inflation is bad news and is **hawkish**, because it pushes towards tightening.
- Falling inflation is good news and is **dovish**, because it makes room to ease.
- Weak growth is bad news and is **dovish**, because it argues for support.

Ask: *if this sentence were the only thing a reader knew, would they expect the next rate move to be up, down, or neither?*

### 3.2 Decision order

1. Is the sentence boilerplate or procedural? If yes: **neutral**, topic `other`. Stop.
2. Does it describe a development, a risk, or guidance about policy? If it is only background with no direction, it is **neutral**.
3. Find the direction using the table in 3.3.
4. If two signals of similar weight point in opposite directions, the sentence is **neutral**. If one clearly dominates, label the dominant one.

### 3.3 Direction rules

| Variable | Moves towards **hawkish** | Moves towards **dovish** | Note |
|---|---|---|---|
| Inflation (headline, core, food, non-food) | rising; above the target band; stays elevated | falling; back inside or below the band | within the band and stable is neutral |
| Inflation expectations, price pressures | rising; second-round effects; wage pressure | anchored; easing pressures | |
| Risks to inflation | "upside risks" | "downside risks" | "balanced risks" is neutral |
| Economic activity, growth | see **[CALL 2]** | weak; slowing; contraction; "downside risks to growth"; falling confidence | asymmetric by default |
| Exchange rate (cedi) | depreciation; pressure on the cedi; pass-through to prices **[CALL 1]** | appreciation; stability | |
| External position | | stronger reserves **[CALL 3]**; higher export receipts; narrower current-account deficit | |
| Fiscal position | deficit overshoot; revenue shortfall; higher government borrowing or central-bank financing **[CALL 4]** | consolidation; deficit on target | |
| Money and credit | liquidity overhang; fast monetary growth **[CALL 5]** | weak or contracting credit | asymmetric by default |
| Commodity prices | higher oil or fuel prices (imported inflation) **[CALL 6]** | lower oil prices; higher gold or cocoa prices (export relief) | |
| Market interest rates, yields | neutral by default **[CALL 7]** | neutral by default | outcomes, not signals |
| Banking-sector soundness | neutral by default **[CALL 8]** | neutral by default | |
| Global economy | rising global inflation; tightening by major central banks; strong dollar | global slowdown; easing by major central banks; falling global commodity prices | an event abroad is judged by its effect on Ghana |
| Forward guidance | "further tightening", "remain vigilant", "stay the course" | "scope to ease", "accommodative", "support growth" | "data dependent" alone is neutral |

### 3.4 The eight judgment calls

| # | Question | Default | Alternative |
|---|---|---|---|
| 1 | Is cedi depreciation hawkish? | **Yes**: pass-through to inflation | Neutral unless the sentence links it to prices |
| 2 | Is strong growth hawkish? | **Neutral** unless the sentence says demand is above potential or adds price pressure. Weak growth is dovish. | Strong growth hawkish (symmetric) |
| 3 | Do improved reserves count as dovish? | **Yes**: less pressure to defend the cedi | Neutral |
| 4 | Is fiscal slippage hawkish? | **Yes**: inflationary pressure and fiscal dominance risk. Consolidation is dovish. | Neutral unless tied to inflation |
| 5 | Is fast credit or money growth hawkish? | **Hawkish only if framed as liquidity or inflation pressure.** Weak credit is dovish. | Symmetric |
| 6 | Oil and cocoa/gold prices | **Higher oil is hawkish; higher gold and cocoa receipts are dovish** | Neutral |
| 7 | Rising or falling T-bill yields and interbank rates | **Neutral**: market outcomes, not the Committee's signal | Rising yields hawkish |
| 8 | Bank soundness (NPLs, capital) | **Neutral**; dovish only when the sentence calls for caution or support | Always neutral |

### 3.5 Levels, changes and expectations

- Judge the **direction of the news** for flows ("inflation fell").
- Judge the **level against the target** for stocks ("inflation remains above the target band" is hawkish).
- A forecast counts the same as an outturn.
- "Slower than expected" or "higher than projected" is judged by the direction of the surprise.
- Judge the main clause. A subordinate clause only changes the label when it reverses the main message ("although inflation fell, the Committee remains concerned about…" is hawkish).

## 4. Topic

One primary topic per sentence.

| Topic | Covers |
|---|---|
| `inflation` | prices, inflation expectations, price pressures, the target band |
| `growth` | GDP, activity indicators, PMI, confidence, sector output (mining, agriculture), employment |
| `external` | the cedi, balance of payments, trade, reserves, remittances |
| `fiscal` | budget, revenue, expenditure, deficit, public debt, government borrowing and financing |
| `financial` | money and credit aggregates, interest rates and yields, bank soundness, financial markets |
| `global` | the world economy, other countries, other central banks, global commodity prices and financial conditions |
| `policy` | the Bank's stance and guidance, policy measures (reserve requirements, liquidity operations), reasoning about the policy outlook |
| `other` | greetings, procedure, scheduling, thank-yous, and anything without economic content |

**When a sentence touches several topics:**

1. If it describes a Bank action, stance, or guidance: `policy`.
2. If its subject is another country or a world aggregate: `global`, even when Ghana is mentioned.
3. Otherwise take the topic of the **grammatical subject** ("The cedi depreciated, adding to inflation" is `external`).
4. If the subject is vague, use the topic of the **first stated driver**; if still tied, `inflation`.

The section heading is a hint, not a rule. A sentence under "Global Developments" that is about the cedi is `external`.

## 5. Edge cases

| Case | Rule |
|---|---|
| "Inflation fell, but remains above the target band." | Hawkish if the second clause is the main message ("remains above"). Neutral if both clauses carry equal weight. |
| A list item that completes a sentence from the previous line | Fragments are excluded. Do not guess. |
| Other central banks ("The Fed cut its policy rate") | Topic `global`. Tone: easing abroad is dovish for Ghana by default; tightening abroad is hawkish. |
| Past-tense history ("Inflation was 33 percent in 2001") | Neutral unless the sentence draws a direction from it. |
| Statements about statistics or methodology | Neutral, topic of the variable. |
| Numbers only, no direction ("The cedi traded at 14.5 to the dollar") | Neutral. |
| Quoted official targets ("The target is 8 percent, plus or minus 2") | Neutral, topic `inflation`. |
| Doubtful between hawkish and neutral | Neutral. Mark `unsure` in the notes column. |
| Doubtful between two topics | Apply the precedence rules in section 4 and mark `unsure`. |

## 6. Annotation procedure

1. **Blind.** The annotation file shows only: `sentence_id`, section heading, sentence text. The lexicon's label is not shown.
2. **Sample.** About 400 sentences, drawn only from the labellable pool (section 2), stratified by era (2003-2009, 2010-2016, 2017-2026), by topic hint where one exists, and by lexicon class (computed but hidden). At least 100 come from 2022-2026 so the model can be tested out of regime.
3. **Split.** About 100 sentences form **gold-dev**, used to tune the lexicon and to correct weak labels in training. About 300 form **gold-test** and are never used for training or tuning.
4. **Reliability.** A random 10 percent of the gold set is re-labelled after at least seven days, without looking at the first labels. Cohen's kappa between the two passes is reported in the README. With one annotator this is the only available check on consistency.
5. **Disagreements with the lexicon** are reviewed only after labelling is finished, never during.
6. **Changes to this codebook** are logged below. If a rule changes, the affected labels are re-checked and the version recorded with the dataset.

## 7. Examples

Sentences marked (real) are from the corpus. Those marked (illustrative) are written for this codebook and are not quotes.

| Sentence | Tone | Topic | Why |
|---|---|---|---|
| "The Committee notes that challenges in the Eurozone continue to pose threats to the global economic outlook." (real) | dovish | global | Global slowdown risk |
| "There are substantial downside risks to growth due to the austerity measures being implemented in parts of the zone." (real) | dovish | global | Subject is another region; downside risk to growth |
| "GDP growth is forecasted to decline to 5.0 percent in a baseline scenario." (real) | dovish | growth | Weakening outlook |
| "The cedi has appreciated significantly, gaining 42.6 percent, year-to-date, against the US dollar, supported by strong foreign exchange…" (real) | dovish | external | Appreciation, less pass-through (call 1) |
| "The year 2002 ended with inflation at 15.2 percent, down from 21.3 per cent in December 2001, and up from 12.9 per cent in September 2002." (real) | neutral | inflation | Two offsetting directions |
| "Thank you for coming." (real) | neutral | other | No content |
| "Inflation rose further above the upper band of the target." (illustrative) | hawkish | inflation | Rising and above target |
| "The Committee remains vigilant to upside risks to the inflation outlook." (illustrative) | hawkish | policy | Guidance with upside risk |
| "Pressures on the cedi intensified during the quarter." (illustrative) | hawkish | external | Depreciation pressure (call 1) |
| "Output in the mining sector contracted in the second quarter." (illustrative) | dovish | growth | Weak activity |
| "Treasury bill yields rose across the curve." (illustrative) | neutral | financial | Market outcome (call 7) |

## 8. Known limitations

- **One annotator.** Tone is subjective. Reliability is checked only against the annotator's own later labels.
- **The direction rules are economic judgments**, not facts. Calls 1 to 8 are the contestable ones, and the README will report how results change if they are flipped.
- **Text extraction noise** (glued words, dropped table rows) affects some sentences and cannot be fixed from the PDFs.
- **Sentence-level labelling ignores context.** Some sentences are only hawkish or dovish in light of the sentence before them.

## 9. Change log

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-10-04 | First draft. Eight judgment calls set to defaults. |
