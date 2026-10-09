---
title: Bank of Ghana Policy Tone
emoji: 🏦
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
short_description: Hawkish or dovish? Tone of Bank of Ghana MPC statements
models:
  - Kobichris/cb-policy-tone
---

# Bank of Ghana policy tone

Classifies sentences from Bank of Ghana Monetary Policy Committee statements as **hawkish**,
**neutral** or **dovish**, and shows a per-meeting tone index (2003–2026) next to the policy rate
and inflation. The model runs in your browser with transformers.js.

- Model: FinBERT fine-tuned on rule-based (lexicon) labels. Version 0: accuracy against
  hand-labelled sentences is still pending.
- Finding so far: tone is associated with the next rate decision in-sample, but does not improve
  out-of-sample forecasts beyond past decisions and inflation.
- Research prototype, not financial advice.

Code, method and limitations: https://github.com/Mbibachris/cb-policy-tone
