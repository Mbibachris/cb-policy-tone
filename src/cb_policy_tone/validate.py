"""Does the tone of a statement help predict the NEXT rate decision?

Reads : data/processed/tone_index.csv, policy_rates.csv, statements.csv (and the extracted texts)
        data/external/inflation_overrides.csv   (optional manual fixes: meeting_no, infl)

SPECIFICATION (fixed before this was run on the real data; do not change it after seeing results)
-----------------------------------------------------------------------------------------------
Unit        : regular meetings that have a statement.
Outcome     : the direction of the NEXT regular meeting's decision: cut (-1), hold (0), hike (+1).
Model       : ordered probit.
Controls    : the direction of the decision at this meeting (momentum), and, where available, the
              inflation figure the statement itself reports (text-derived, no look-ahead).
Test        : add the standardised tone index to the controls and compare nested models with a
              likelihood-ratio test, AIC and McFadden's pseudo-R2. Also a pseudo out-of-sample
              check: fit up to 2018, score 2019 onwards.
Tone indices: model_net (the fine-tuned model) and lexicon_net, reported side by side.
Reading     : descriptive and correlational. About 115 observations, and a statement is not an
              exogenous event. No causal claims. A null result is a result.
The lexicon is NOT adjusted in response to anything this script prints.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

from cb_policy_tone.sentences import split_statement

INDEX = Path("data/processed/tone_index.csv")
RATES = Path("data/processed/policy_rates.csv")
STATEMENTS = Path("data/processed/statements.csv")
OVERRIDES = Path("data/external/inflation_overrides.csv")
INFLATION_OUT = Path("data/processed/inflation_from_statements.csv")
SPLIT_YEAR = 2018
TONE_VARIABLES = ["model_net", "lexicon_net"]

SKIP = re.compile(
    r"core|food|expectation|target|band|objective|forecast|project|outlook|risk|pressure|real ",
    re.IGNORECASE,
)
LEVEL = re.compile(
    r"\binflation\b.{0,160}?\b(?:rose|fell|declined|increased|decreased|eased|moderated|accelerated"
    r"|picked up|stood|stands|remained|remains|was|is|at|to|of)\s+"
    r"(?:marginally\s+|slightly\s+|further\s+|sharply\s+|steadily\s+|again\s+)?(?:at\s+|to\s+)?"
    r"(\d{1,2}(?:\.\d+)?)\s*(?:per\s*cent|percent|%)",
    re.IGNORECASE,
)


def extract_inflation(sentences: list[str]) -> float | None:
    """The first headline-inflation level stated in the opening sentences of a statement."""
    for text in sentences[:60]:
        if SKIP.search(text):
            continue
        m = LEVEL.search(text)
        if m and 0 < float(m.group(1)) < 100:
            return float(m.group(1))
    return None


def statement_inflation(statements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in statements.itertuples():
        text = Path(s.txt_file).read_text(encoding="utf-8")
        sentences = [r["text"] for r in split_statement(text)]
        rows.append(
            {
                "doc_id": s.doc_id,
                "meeting_no": s.meeting_no,
                "meeting_type": s.meeting_type,
                "infl": extract_inflation(sentences),
            }
        )
    return pd.DataFrame(rows)


def apply_overrides(inflation: pd.DataFrame, overrides: pd.DataFrame | None) -> pd.DataFrame:
    if overrides is None or overrides.empty:
        return inflation
    fixed = inflation.copy()
    fixed["meeting_no"] = fixed["meeting_no"].astype("Int64")
    fix = dict(
        zip(overrides["meeting_no"].astype(int), overrides["infl"].astype(float), strict=True)
    )
    mask = fixed["meeting_no"].isin(list(fix))
    fixed.loc[mask, "infl"] = fixed.loc[mask, "meeting_no"].map(fix).astype(float)
    return fixed


def build_frame(
    index: pd.DataFrame, rates: pd.DataFrame, inflation: pd.DataFrame | None
) -> pd.DataFrame:
    """One row per regular meeting with a statement: tone now, decision now, decision NEXT."""
    reg = rates[rates["meeting_type"] == "regular"].sort_values("meeting_no").copy()
    reg["dir"] = np.sign(reg["mpr_change_bp"].astype(float))
    reg["next_dir"] = reg["dir"].shift(-1)
    reg["year"] = pd.to_datetime(reg["meeting_end"]).dt.year
    tone = index[index["meeting_type"] == "regular"][
        ["meeting_no", "lexicon_net", "model_net", "n_scored"]
    ]
    frame = reg.merge(tone, on="meeting_no", how="inner")
    if inflation is not None:
        reg_infl = inflation[inflation["meeting_type"] == "regular"][["meeting_no", "infl"]]
        frame = frame.merge(reg_infl.astype({"meeting_no": "int64"}), on="meeting_no", how="left")
    frame["infl"] = frame.get("infl", np.nan)
    return frame.dropna(subset=["dir", "next_dir", "lexicon_net", "model_net"]).reset_index(
        drop=True
    )


def standardise(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        out[f"{col}_z"] = (out[col] - out[col].mean()) / out[col].std(ddof=0)
    return out


def fit_probit(frame: pd.DataFrame, cols: list[str]):
    y = frame["next_dir"].astype(int).map({-1: 0, 0: 1, 1: 2})
    y = pd.Series(pd.Categorical(y, categories=[0, 1, 2], ordered=True), index=frame.index)
    return OrderedModel(y, frame[cols], distr="probit").fit(method="bfgs", disp=False, maxiter=1000)


def null_loglik(frame: pd.DataFrame) -> float:
    counts = frame["next_dir"].value_counts()
    return float((counts * np.log(counts / counts.sum())).sum())


def out_of_sample_gain(frame: pd.DataFrame, base: list[str], tone: str) -> float | None:
    """Mean log-likelihood gain per meeting from adding tone, fit up to SPLIT_YEAR, scored after."""
    train, test = frame[frame["year"] <= SPLIT_YEAR], frame[frame["year"] > SPLIT_YEAR]
    if len(test) < 15 or train["next_dir"].nunique() < 3:
        return None
    actual = test["next_dir"].astype(int).map({-1: 0, 0: 1, 1: 2}).to_numpy()
    scores = []
    for cols in (base, [*base, tone]):
        res = fit_probit(train, cols)
        probs = np.asarray(res.model.predict(res.params, exog=test[cols].to_numpy()))
        scores.append(
            float(np.mean(np.log(np.clip(probs[np.arange(len(test)), actual], 1e-9, None))))
        )
    return scores[1] - scores[0]


def test_tone(frame: pd.DataFrame, tone: str, controls: list[str]) -> dict:
    needed = [*controls, tone, "next_dir"]
    sample = standardise(frame.dropna(subset=needed), [tone])
    z = f"{tone}_z"
    base = fit_probit(sample, controls)
    full = fit_probit(sample, [*controls, z])
    lr = 2 * (full.llf - base.llf)
    ll0 = null_loglik(sample)
    return {
        "n": len(sample),
        "coef": float(full.params[z]),
        "se": float(full.bse[z]),
        "p_coef": float(full.pvalues[z]),
        "lr": float(lr),
        "p_lr": float(stats.chi2.sf(lr, df=1)),
        "aic_base": float(base.aic),
        "aic_full": float(full.aic),
        "r2_base": 1 - float(base.llf) / ll0,
        "r2_full": 1 - float(full.llf) / ll0,
        "oos_gain": out_of_sample_gain(sample, controls, z),
        "full": full,
        "sample": sample,
        "z": z,
    }


def hike_probabilities(result: dict, controls: list[str]) -> dict[str, tuple[float, float, float]]:
    """P(cut), P(hold), P(hike) next meeting when tone is 1 sd below, at, and 1 sd above average."""
    full, sample, z = result["full"], result["sample"], result["z"]
    out = {}
    for label, shift in (("tone -1 sd", -1.0), ("tone average", 0.0), ("tone +1 sd", 1.0)):
        row = {c: float(sample[c].mean()) for c in controls}
        row[z] = shift
        probs = np.asarray(
            full.model.predict(full.params, exog=pd.DataFrame([row])[[*controls, z]].to_numpy())
        )
        out[label] = tuple(float(p) for p in probs[0])
    return out


def report(frame: pd.DataFrame) -> None:
    sys.stdout.reconfigure(errors="replace")
    has_infl = frame["infl"].notna().sum() >= 40
    print(f"Meetings in the analysis: {len(frame)}")
    print(
        "Next decision:",
        frame["next_dir"].map({-1: "cut", 0: "hold", 1: "hike"}).value_counts().to_dict(),
    )
    print(
        f"Meetings with a statement-reported inflation figure: {int(frame['infl'].notna().sum())}"
    )
    specs = [("momentum only", ["dir"])]
    if has_infl:
        specs.append(("momentum + inflation", ["dir", "infl"]))
    for label, controls in specs:
        print(f"\n=== Controls: {label} ===")
        for tone in TONE_VARIABLES:
            r = test_tone(frame, tone, controls)
            oos = f"{r['oos_gain']:+.3f}" if r["oos_gain"] is not None else "n/a"
            print(
                f"  {tone:12s} n={r['n']} | coef on tone (per sd) {r['coef']:+.3f} (se {r['se']:.3f}, p={r['p_coef']:.3f})"
                f" | LR={r['lr']:.2f} (p={r['p_lr']:.3f}) | AIC {r['aic_base']:.1f} -> {r['aic_full']:.1f}"
                f" | pseudo-R2 {r['r2_base']:.3f} -> {r['r2_full']:.3f} | out-of-sample log-lik gain per meeting {oos}"
            )
            probs = hike_probabilities(r, controls)
            for name, (cut, hold, hike) in probs.items():
                print(f"      {name:13s}: P(cut) {cut:.2f}  P(hold) {hold:.2f}  P(hike) {hike:.2f}")
    print(
        "\nReading: descriptive only. About a hundred meetings; no causal claim. See the module docstring."
    )


def main() -> None:
    index = pd.read_csv(INDEX)
    rates = pd.read_csv(RATES)
    statements = pd.read_csv(STATEMENTS)
    inflation = statement_inflation(statements)
    overrides = pd.read_csv(OVERRIDES) if OVERRIDES.exists() else None
    inflation = apply_overrides(inflation, overrides)
    inflation.to_csv(INFLATION_OUT, index=False)
    reg = inflation[inflation["meeting_type"] == "regular"].sort_values("meeting_no")
    print(
        f"Inflation figure extracted from {int(reg['infl'].notna().sum())} of {len(reg)} regular statements."
    )
    print("Check this series against what you know (meeting_no: inflation):")
    print(
        ", ".join(
            f"{int(n)}:{v:.1f}"
            for n, v in zip(reg["meeting_no"], reg["infl"], strict=True)
            if pd.notna(v)
        )
    )
    frame = build_frame(index, rates, inflation)
    report(frame)


if __name__ == "__main__":
    main()
