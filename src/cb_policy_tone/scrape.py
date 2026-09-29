"""Download Bank of Ghana MPC press-release PDFs and record a provenance manifest."""

from __future__ import annotations

import argparse
import csv
import html
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API = "https://www.bog.gov.gh/wp-json/wp/v2/mpc_press_release"
HEADERS = {"User-Agent": "cb-policy-tone research (mbibachris@gmail.com)"}
DELAY_SECONDS = 2.0
PDF_DIR = Path("data/raw/pdf")
MANIFEST = Path("data/manifest.csv")

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]  # fmt: skip
TITLE_RE = re.compile(r"(" + "|".join(MONTHS) + r")\s+(\d{4})", re.IGNORECASE)

FIELDS = [
    "id", "title", "meeting_year", "meeting_month", "feed_date", "slug",
    "page_url", "pdf_url", "n_pdf_links", "file", "bytes", "status", "retrieved_on",
]  # fmt: skip


def get(url: str, retries: int = 3, **kwargs) -> requests.Response | None:
    """Polite GET: pause before every request, retry on network errors."""
    for attempt in range(1, retries + 1):
        time.sleep(DELAY_SECONDS)
        try:
            r = requests.get(url, headers=HEADERS, timeout=60, **kwargs)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"  attempt {attempt} failed: {type(e).__name__}")
            time.sleep(5)
        except requests.HTTPError as e:
            print(f"  HTTP error: {e}")
            return None
    return None


def list_posts() -> list[dict]:
    """All press-release posts from the WordPress REST feed."""
    posts, page = [], 1
    while True:
        params = {"per_page": 100, "page": page, "_fields": "id,slug,link,title,date"}
        r = get(API, params=params)
        if r is None:
            raise RuntimeError(f"Could not fetch page {page} of the post list")
        posts.extend(r.json())
        if page >= int(r.headers.get("X-WP-TotalPages", 1)):
            return posts
        page += 1


def parse_title(title: str) -> tuple[int | None, int | None]:
    """'MPC Press Release - May 2018' -> (2018, 5). The feed's own dates are unreliable."""
    m = TITLE_RE.search(title)
    if not m:
        return None, None
    return int(m.group(2)), MONTHS.index(m.group(1).lower()) + 1


def find_pdf_urls(page_html: str) -> list[str]:
    """PDF links on a press-release page, in page order, without duplicates."""
    soup = BeautifulSoup(page_html, "html.parser")
    urls: list[str] = []
    for tag in soup.find_all(["a", "iframe", "embed", "object"]):
        u = tag.get("href") or tag.get("src") or tag.get("data")
        if u and ".pdf" in u.lower() and u not in urls:
            urls.append(u)
    return urls


def fetch_statement(post: dict) -> dict:
    """Download one statement's PDF and return its manifest row."""
    title = html.unescape(post["title"]["rendered"])
    year, month = parse_title(title)
    row = {
        "id": str(post["id"]),
        "title": title,
        "meeting_year": year or "",
        "meeting_month": month or "",
        "feed_date": post["date"][:10],
        "slug": post["slug"],
        "page_url": post["link"],
        "pdf_url": "",
        "n_pdf_links": 0,
        "file": "",
        "bytes": 0,
        "status": "",
        "retrieved_on": datetime.now(tz=timezone.utc).date().isoformat(),
    }

    page = get(post["link"])
    if page is None:
        row["status"] = "page_failed"
        return row

    urls = find_pdf_urls(page.text)
    row["n_pdf_links"] = len(urls)
    if not urls:
        row["status"] = "no_pdf"
        return row
    row["pdf_url"] = urls[0]

    pdf = get(urls[0])
    if pdf is None:
        row["status"] = "download_failed"
        return row
    if not pdf.content.startswith(b"%PDF"):
        row["status"] = "not_a_pdf"
        return row

    stem = f"{year}-{month:02d}_{row['id']}" if year and month else f"unknown_{row['id']}"
    path = PDF_DIR / f"{stem}.pdf"
    path.write_bytes(pdf.content)
    row["file"] = path.as_posix()
    row["bytes"] = len(pdf.content)
    row["status"] = "ok"
    return row


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def save_manifest(rows: dict[str, dict]) -> None:
    ordered = sorted(
        rows.values(),
        key=lambda r: (int(r["meeting_year"] or 0), int(r["meeting_month"] or 0), r["id"]),
    )
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only process N posts (testing)")
    args = parser.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    posts = list_posts()
    print(f"Found {len(posts)} press-release posts")
    if args.limit:
        posts = posts[: args.limit]

    for i, post in enumerate(posts, start=1):
        pid = str(post["id"])
        done = manifest.get(pid)
        if done and done["status"] == "ok" and Path(done["file"]).exists():
            continue  # already downloaded
        print(f"[{i}/{len(posts)}] {html.unescape(post['title']['rendered'])}")
        manifest[pid] = fetch_statement(post)
        save_manifest(manifest)  # save after every post so a crash loses nothing

    print("\nStatus counts:", dict(Counter(r["status"] for r in manifest.values())))
    print(f"Manifest: {MANIFEST}   PDFs: {PDF_DIR}")


if __name__ == "__main__":
    main()