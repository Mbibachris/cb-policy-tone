"""Publish the demo page in app/ to a free (static) Hugging Face Space.

Run from the project folder:
  python scripts/deploy_space.py            create or update the Space (private)
  python scripts/deploy_space.py --public   the same, and make the Space public

The page runs the model in the visitor's browser with transformers.js, so it needs no server.
That only works once the model repo is public and has its ONNX files (notebook 05).
The tone index data is rebuilt from data/processed each time and uploaded as data.json.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi

SPACE = "Kobichris/policy-tone"
PAGE_FILES = ["index.html", "README.md"]
APP = Path("app")
INDEX = Path("data/processed/tone_index.csv")
RATES = Path("data/processed/policy_rates.csv")
INFLATION = Path("data/processed/inflation_from_statements.csv")


def zscore(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    return (values - values.mean()) / sd if sd > 0 else values * 0.0


def meeting_rows(index: pd.DataFrame, rates: pd.DataFrame, inflation: pd.DataFrame) -> list[dict]:
    """One record per regular meeting, oldest first, with z-scored tone."""
    reg = rates[rates["meeting_type"] == "regular"].copy()
    tone = index[index["meeting_type"] == "regular"][["meeting_no", "model_net", "lexicon_net"]]
    infl = inflation[inflation["meeting_type"] == "regular"][["meeting_no", "infl"]]
    table = reg.merge(tone, on="meeting_no", how="left").merge(infl, on="meeting_no", how="left")
    table["model_z"] = zscore(table["model_net"])
    table["lexicon_z"] = zscore(table["lexicon_net"])
    table = table.sort_values("meeting_end")

    def num(value, digits):
        return None if pd.isna(value) else round(float(value), digits)

    return [
        {
            "date": str(r.meeting_end)[:10],
            "meeting": int(r.meeting_no),
            "mpr": num(r.mpr, 2),
            "change": num(r.mpr_change_bp, 0),
            "decision": None if pd.isna(r.decision) else str(r.decision),
            "model_z": num(r.model_z, 3),
            "lexicon_z": num(r.lexicon_z, 3),
            "infl": num(r.infl, 1),
        }
        for r in table.itertuples()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="make the Space public")
    args = parser.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:  # noqa: BLE001
        print("You are not logged in to Hugging Face. Run:  hf auth login   and then run this again.")
        sys.exit(1)
    print(f"Logged in as {user}")

    if not api.repo_exists(SPACE, repo_type="space"):
        api.create_repo(SPACE, repo_type="space", space_sdk="static", private=True)
        print(f"Created the Space {SPACE} (private).")

    rows = meeting_rows(pd.read_csv(INDEX), pd.read_csv(RATES), pd.read_csv(INFLATION))
    operations = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(APP / f)) for f in PAGE_FILES]
    operations.append(
        CommitOperationAdd(path_in_repo="data.json", path_or_fileobj=json.dumps(rows).encode())
    )
    api.create_commit(SPACE, repo_type="space", operations=operations, commit_message="Update page")
    print(f"Uploaded index.html, README.md and data.json ({len(rows)} meetings).")

    if args.public:
        api.update_repo_settings(SPACE, repo_type="space", private=False)
        print("The Space is now public.")
    print(f"Open https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    main()
