"""Bank of Ghana policy tone: a demo of the cb-policy-tone model.

Tab 1 classifies sentences you paste as hawkish, neutral or dovish.
Tab 2 shows the per-meeting tone index next to the policy rate and inflation.
Tab 3 explains the method and its limits.

The model runs on CPU in the main process. The Space uses ZeroGPU hardware only because free
accounts cannot host CPU Gradio Spaces; the decorated function below is never called.
"""

import spaces  # noqa: I001  (must be imported before torch)

import os
import re
from collections.abc import Callable
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODEL_ID = os.environ.get("MODEL_ID", "Kobichris/cb-policy-tone")
DATA = Path(__file__).parent / "data"
if not DATA.exists():  # running locally from the project folder
    DATA = Path("data/processed")
MAX_LEN = 128
MAX_SENTENCES = 200
BATCH = 32
LABELS = ["hawkish", "neutral", "dovish"]
COLOURS = {"hawkish": "#b03a2e", "neutral": "#a6a6a6", "dovish": "#2e7d4f"}
REPO_URL = "https://github.com/Mbibachris/cb-policy-tone"

Predictor = Callable[[list[str]], pd.DataFrame]


@spaces.GPU(duration=1)
def _noop() -> None:
    """ZeroGPU needs at least one decorated function. Never called: inference runs on CPU."""


# ---------------------------------------------------------------- sentences and the model

ABBREVIATIONS = ["U.S.", "U.K.", "e.g.", "i.e.", "No.", "Mr.", "Mrs.", "Dr.", "St.", "vs.", "etc."]
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"“])")


def split_sentences(text: str) -> list[str]:
    """Split pasted text into sentences (simple rules; good enough for MPC prose)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    for abbr in ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", "<DOT>"))
    return [p.replace("<DOT>", ".").strip() for p in SPLIT.split(text) if p.strip()]


def load_predictor(model_id: str = MODEL_ID) -> Predictor:
    """Load the fine-tuned model once. Returns a function: sentences -> probability table."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    names = [model.config.id2label[i] for i in range(model.config.num_labels)]

    @torch.no_grad()
    def predict(sentences: list[str]) -> pd.DataFrame:
        parts = []
        for start in range(0, len(sentences), BATCH):
            enc = tokenizer(
                sentences[start : start + BATCH],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            parts.append(torch.softmax(model(**enc).logits.float(), dim=-1).numpy())
        return pd.DataFrame(np.vstack(parts), columns=names)[LABELS]

    return predict


def analyse(text: str, predict: Predictor) -> tuple[list, pd.DataFrame, str]:
    """Classify each sentence. Returns highlighted text, a probability table and a summary."""
    sentences = split_sentences(text)
    empty = pd.DataFrame(columns=["#", "tone", *[f"P({lab})" for lab in LABELS], "sentence"])
    if not sentences:
        return [], empty, "Paste one or more sentences and press **Classify**."
    cut = len(sentences) > MAX_SENTENCES
    sentences = sentences[:MAX_SENTENCES]

    probs = predict(sentences)
    tone = probs.idxmax(axis=1)
    highlighted = []
    for sentence, label in zip(sentences, tone, strict=True):
        highlighted += [(sentence, label), (" ", None)]

    table = pd.DataFrame({"#": range(1, len(sentences) + 1), "tone": tone})
    for lab in LABELS:
        table[f"P({lab})"] = probs[lab].round(3)
    table["sentence"] = sentences

    net = float((probs["hawkish"] - probs["dovish"]).mean())
    counts = tone.value_counts()
    lean = "hawkish" if net > 0.05 else "dovish" if net < -0.05 else "balanced"
    summary = (
        f"**{len(sentences)} sentences**: "
        + ", ".join(f"{counts.get(lab, 0)} {lab}" for lab in LABELS)
        + f"\n\n**Net tone {net:+.2f}** on a scale from −1 (all dovish) to +1 (all hawkish), "
        f"so this text leans **{lean}**."
        "\n\nThe published index leaves out the sentence that announces the rate decision, so a "
        "full statement scored here is not exactly comparable with it."
    )
    if cut:
        summary += f"\n\nOnly the first {MAX_SENTENCES} sentences were scored."
    return highlighted, table, summary


# ---------------------------------------------------------------- the tone index


def zscore(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    return (values - values.mean()) / sd if sd > 0 else values * 0.0


def meeting_table(folder: Path = DATA) -> pd.DataFrame | None:
    """One row per regular meeting: date, rate, decision, inflation and both tone measures."""
    infl_name = (
        "inflation.csv" if (folder / "inflation.csv").exists() else "inflation_from_statements.csv"
    )
    paths = [folder / f for f in ("tone_index.csv", "policy_rates.csv", infl_name)]
    if not all(p.exists() for p in paths):
        return None
    index, rates, infl = (pd.read_csv(p) for p in paths)
    reg = rates[rates["meeting_type"] == "regular"].copy()
    reg["date"] = pd.to_datetime(reg["meeting_end"])
    tone = index[index["meeting_type"] == "regular"][["meeting_no", "model_net", "lexicon_net"]]
    infl = infl[infl["meeting_type"] == "regular"][["meeting_no", "infl"]]
    table = reg.merge(tone, on="meeting_no", how="left").merge(infl, on="meeting_no", how="left")
    table["model_z"] = zscore(table["model_net"])
    table["lexicon_z"] = zscore(table["lexicon_net"])
    return table.sort_values("date").reset_index(drop=True)


def index_figure(table: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Tone of the statement (z-score, higher = more hawkish)",
            "Monetary Policy Rate (%)",
            "Inflation quoted in the statement (%)",
        ),
    )
    hover = (
        "Meeting "
        + table["meeting_no"].astype(int).astype(str)
        + "<br>"
        + table["date"].dt.strftime("%d %b %Y")
        + "<br>Decision: "
        + table["decision"].fillna("n/a").astype(str)
    )
    for col, name, colour in (
        ("model_z", "Fine-tuned model", "#1f4e79"),
        ("lexicon_z", "Lexicon", "#c0792b"),
    ):
        ok = table[col].notna()
        fig.add_trace(
            go.Scatter(
                x=table.loc[ok, "date"],
                y=table.loc[ok, col],
                name=name,
                mode="lines+markers",
                marker={"size": 4},
                line={"color": colour, "width": 1.5},
                text=hover[ok],
                hovertemplate="%{text}<br>Tone %{y:.2f}<extra>" + name + "</extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_hline(y=0, line={"color": "#999", "width": 0.8}, row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=table["date"],
            y=table["mpr"],
            name="Policy rate",
            line={"shape": "hv", "color": "#222"},
            text=hover,
            hovertemplate="%{text}<br>Rate %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    change = table["mpr_change_bp"].astype(float)
    for mask, symbol, colour, name in (
        (change > 0, "triangle-up", COLOURS["hawkish"], "Hike"),
        (change < 0, "triangle-down", COLOURS["dovish"], "Cut"),
    ):
        fig.add_trace(
            go.Scatter(
                x=table.loc[mask, "date"],
                y=table.loc[mask, "mpr"],
                mode="markers",
                name=name,
                marker={"symbol": symbol, "color": colour, "size": 8},
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
    ok = table["infl"].notna()
    fig.add_trace(
        go.Scatter(
            x=table.loc[ok, "date"],
            y=table.loc[ok, "infl"],
            name="Inflation",
            mode="lines+markers",
            marker={"size": 4},
            line={"color": "#6a3d9a", "width": 1.5},
            text=hover[ok],
            hovertemplate="%{text}<br>Inflation %{y:.1f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        height=760,
        margin={"l": 50, "r": 20, "t": 40, "b": 30},
        legend={"orientation": "h", "y": -0.05},
        hovermode="closest",
        template="plotly_white",
    )
    return fig


def meetings_for_display(table: pd.DataFrame) -> pd.DataFrame:
    out = table.sort_values("date", ascending=False)[
        ["date", "meeting_no", "mpr", "decision", "model_z", "lexicon_z", "infl"]
    ].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["meeting_no"] = out["meeting_no"].astype(int)
    out = out.round({"model_z": 2, "lexicon_z": 2, "infl": 1})
    return out.rename(
        columns={
            "meeting_no": "meeting",
            "mpr": "policy rate %",
            "model_z": "tone (model, z)",
            "lexicon_z": "tone (lexicon, z)",
            "infl": "inflation %",
        }
    )


# ---------------------------------------------------------------- page

INTRO = f"""
# Bank of Ghana policy tone
Is a Monetary Policy Committee statement **hawkish** (leaning towards tighter policy and higher
rates), **dovish** (leaning towards easier policy and lower rates) or **neutral**? This demo uses
a FinBERT model fine-tuned on sentences from Bank of Ghana MPC statements, 2003–2026.
Research prototype (v0), not financial advice. Code and data notes: [GitHub]({REPO_URL}).
"""

EXAMPLES = [
    (
        "Inflation pressures have intensified, driven by exchange rate depreciation and upward "
        "adjustments in utility tariffs. The Committee judged that the risks to the inflation "
        "outlook are tilted to the upside."
    ),
    (
        "Headline inflation has declined for the sixth consecutive month and is expected to "
        "return to the medium-term target band earlier than projected. Growth has slowed, and "
        "private sector credit remains weak."
    ),
    (
        "The banking sector remains liquid, well capitalised and profitable. Total assets grew "
        "by 18.2 percent year-on-year."
    ),
]

ABOUT = f"""
## How it works
1. **Data.** 121 MPC statements scraped from the Bank of Ghana website, matched to 132 regular
   meetings (2002–2026), split into about 10,000 sentences.
2. **Labels.** A rule-based lexicon, written from a published codebook, labels every sentence.
   Its rules depend on *what* is moving: rising inflation is hawkish, rising growth risks are
   dovish. Sentences that announce the rate decision are left out.
3. **Model.** `ProsusAI/finbert` fine-tuned on those rule labels (train ≤ 2018, validation
   2019–2021, test 2022–2026).
4. **Index.** For each statement: the average of P(hawkish) − P(dovish) over its sentences,
   shown as a z-score. Only movements over time carry meaning, not the level.

## What it can and cannot tell you
- In an ordered probit, a more hawkish statement is followed by a higher chance of a hike at
  the next meeting, controlling for the latest rate moves and inflation (116 meetings).
- It does **not** improve forecasts of 2019–2026 decisions out of sample. Tone tracks policy;
  it does not predict it better than past decisions and inflation already do.
- The model learned from rule-based labels, so the two lines in the chart agree by construction.
  Accuracy against hand-labelled sentences is **pending** (a 400-sentence blind gold set is
  being labelled).
- A sentence about rising inflation reads as hawkish even when inflation is low (as in 2026).
- Inflation is the figure quoted in each statement, hand-checked against the text; it is not an
  official CPI series.

Full method, corrections log and limitations: [GitHub]({REPO_URL}).
"""


def build_demo(predict: Predictor, folder: Path = DATA) -> gr.Blocks:
    table = meeting_table(folder)

    def classify(text: str) -> tuple[list, pd.DataFrame, str]:
        """Classify each sentence of the text as hawkish, neutral or dovish.

        Args:
            text: one or more sentences, for example a paragraph of an MPC statement.
        """
        return analyse(text, predict)

    with gr.Blocks(title="Bank of Ghana policy tone") as demo:
        gr.Markdown(INTRO)
        with gr.Tab("Classify text"):
            text = gr.Textbox(
                label="Text", lines=6, placeholder="Paste a paragraph or a whole MPC statement…"
            )
            button = gr.Button("Classify", variant="primary")
            summary = gr.Markdown()
            highlighted = gr.HighlightedText(
                label="Tone of each sentence",
                color_map=COLOURS,
                show_legend=True,
                combine_adjacent=False,
            )
            probs = gr.Dataframe(
                label="Probabilities",
                wrap=True,
                column_widths=["5%", "11%", "11%", "11%", "11%", "51%"],
            )
            gr.Examples(EXAMPLES, inputs=text)
            button.click(
                classify, inputs=text, outputs=[highlighted, probs, summary], api_name="classify"
            )
            text.submit(
                classify, inputs=text, outputs=[highlighted, probs, summary], api_name=False
            )
        with gr.Tab("Tone index"):
            if table is None:
                gr.Markdown("The index data is not available in this build.")
            else:
                gr.Markdown(
                    f"{int(table['model_z'].notna().sum())} regular meetings with a statement "
                    "online. Hover over a point for the meeting and its decision."
                )
                gr.Plot(index_figure(table))
                gr.Dataframe(meetings_for_display(table), label="Meetings", max_height=400)
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    build_demo(load_predictor()).launch(theme=gr.themes.Soft())
