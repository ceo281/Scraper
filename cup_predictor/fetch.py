"""Fetch Melbourne Cup race pages from Wikipedia with caching and backoff.

Wikimedia rate-limits aggressively, so every page is cached on disk and
responses are only accepted if they are valid parse-API JSON (error pages
are HTML and are rejected rather than cached).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "MelbourneCupPredictor/0.1 (github.com/ceo281/scraper)"}
CACHE = Path(__file__).resolve().parent / ".cache"


def _valid(text: str) -> bool:
    if not text.startswith("{"):
        return False
    try:
        return "parse" in json.loads(text)
    except ValueError:
        return False


def fetch_year(year: int, refresh: bool = False, retries: int = 6) -> str:
    """Return the rendered HTML of '<year> Melbourne Cup'."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{year}.json"
    if path.exists() and not refresh and _valid(path.read_text()):
        return json.loads(path.read_text())["parse"]["text"]
    params = dict(action="parse", page=f"{year}_Melbourne_Cup", prop="text",
                  redirects=1, format="json", formatversion=2)
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=30)
            if r.ok and _valid(r.text):
                path.write_text(r.text)
                return json.loads(r.text)["parse"]["text"]
        except requests.RequestException:
            pass
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"could not fetch {year} Melbourne Cup page")


def fetch_range(first: int, last: int, refresh: bool = False) -> dict[int, str]:
    out = {}
    for y in range(first, last + 1):
        cached = (CACHE / f"{y}.json").exists()
        try:
            out[y] = fetch_year(y, refresh)
        except RuntimeError as e:
            print(f"warning: {e}")
        if not cached:
            time.sleep(2)  # be polite between live requests
    return out
