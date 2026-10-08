"""Headline figures for the README and the Space.

Reads : data/processed/tone_index.csv, policy_rates.csv, inflation_from_statements.csv (optional)
Writes: reports/figures/overlay.png            tone, policy rate and inflation over time
        reports/figures/tone_by_next_decision.png   tone before a cut / hold / hike

Tone is z-scored (mean 0, sd 1) so the two measures sit on one axis. The level of the index has
no meaning (the rules are asymmetric by design); only movements do. The figures show
co-movement, nothing causal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cb_policy_tone.validate import (
    INDEX,
    INFLATION_OUT,
    RATES,
    build_frame,
)

OUT_DIR = Path("reports/figures")
TONES = {"model_net": "Fine-tuned model", "lexicon_net": "Lexicon"}
COLOURS = {"model_net": "#1f4e79", "lexicon_net": "#c0792b"}
DECISIONS = {-1.0: "cut", 0.0: "hold", 1.0: "hike"}


def zscore(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    return (values - values.mean()) / sd if sd > 0 else values * 0.0


def meeting_table(index: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Regular meetings with their date, rate, rate change and (if present) tone."""
    reg = rates[rates["meeting_type"] == "regular"].copy()
    reg["date"] = pd.to_datetime(reg["meeting_end"])
    tone = index[index["meeting_type"] == "regular"][["meeting_no", *TONES]]
    return (
        reg.merge(tone, on="meeting_no", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )


def overlay(index: pd.DataFrame, rates: pd.DataFrame, inflation: pd.DataFrame | None):
    table = meeting_table(index, rates)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    for col, label in TONES.items():
        ok = table.dropna(subset=[col])
        ax.plot(
            ok["date"],
            zscore(ok[col]),
            marker="o",
            ms=3,
            lw=1.2,
            label=label,
            color=COLOURS[col],
        )
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Tone (z-score)\nhigher = more hawkish")
    ax.set_title("Bank of Ghana MPC statements: tone, policy rate and inflation")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    ax.step(table["date"], table["mpr"], where="post", color="black", lw=1.4)
    change = table["mpr_change_bp"].astype(float)
    hikes, cuts = table[change > 0], table[change < 0]
    ax.scatter(
        hikes["date"], hikes["mpr"], marker="^", color="#b22222", s=28, label="hike"
    )
    ax.scatter(
        cuts["date"], cuts["mpr"], marker="v", color="#2e7d32", s=28, label="cut"
    )
    ax.set_ylabel("Policy rate (%)")
    ax.legend(loc="upper left", frameon=False, ncol=2)

    ax = axes[2]
    if inflation is not None:
        infl = inflation[inflation["meeting_type"] == "regular"][
            ["meeting_no", "infl"]
        ].merge(table[["meeting_no", "date"]], on="meeting_no")
        infl = infl.dropna(subset=["infl"]).sort_values("date")
        ax.plot(infl["date"], infl["infl"], marker="o", ms=3, lw=1.2, color="#6a3d9a")
    ax.set_ylabel("Inflation quoted in\nstatement (%)")
    ax.set_xlabel("Meeting date")
    fig.tight_layout()
    return fig


def tone_by_next_decision(frame: pd.DataFrame):
    fig, axes = plt.subplots(1, len(TONES), figsize=(10, 4.5), sharey=False)
    rng = np.random.default_rng(0)
    for ax, (col, label) in zip(axes, TONES.items(), strict=True):
        z = zscore(frame[col])
        groups = [z[frame["next_dir"] == k].to_numpy() for k in DECISIONS]
        ax.boxplot(groups, tick_labels=list(DECISIONS.values()), showfliers=False)
        for i, values in enumerate(groups, start=1):
            ax.scatter(
                i + rng.uniform(-0.12, 0.12, len(values)),
                values,
                s=10,
                alpha=0.5,
                color=COLOURS[col],
            )
            if len(values):
                ax.scatter(i, values.mean(), marker="D", color="black", s=30, zorder=3)
            ax.annotate(
                f"n={len(values)}", (i, ax.get_ylim()[0]), ha="center", va="bottom"
            )
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(label)
        ax.set_xlabel("Decision at the NEXT meeting")
    axes[0].set_ylabel("Tone at this meeting (z-score)")
    fig.suptitle("Tone before a cut, hold or hike (diamond = mean)")
    fig.tight_layout()
    return fig


def main() -> None:
    index = pd.read_csv(INDEX)
    rates = pd.read_csv(RATES)
    inflation = pd.read_csv(INFLATION_OUT) if INFLATION_OUT.exists() else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = overlay(index, rates, inflation)
    fig.savefig(OUT_DIR / "overlay.png", dpi=150)
    plt.close(fig)

    frame = build_frame(index, rates, inflation)
    fig = tone_by_next_decision(frame)
    fig.savefig(OUT_DIR / "tone_by_next_decision.png", dpi=150)
    plt.close(fig)

    print(f"Wrote figures to {OUT_DIR}")
    print("Mean z-scored model tone before the next decision:")
    z = zscore(frame["model_net"])
    for k, name in DECISIONS.items():
        sel = z[frame["next_dir"] == k]
        print(
            f"  {name}: n={len(sel)}, mean {sel.mean():+.2f}"
            if len(sel)
            else f"  {name}: n=0"
        )


if __name__ == "__main__":
    main()
