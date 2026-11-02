"""Turn Wikipedia Melbourne Cup pages into one normalised runner table.

Wikipedia's tables change shape from year to year (column order, colspans
for flag icons, footnote markers, "kg" suffixes), so columns are matched by
header keyword and every cell is cleaned explicitly.
"""
from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup

# Countries whose breeding suffix / flag marks a northern-hemisphere-bred horse.
NH = {
    "GB", "IRE", "FR", "GER", "USA", "JPN", "ITY", "UAE", "CAN", "SPA",
    "United Kingdom", "Great Britain", "Ireland", "France", "Germany",
    "United States", "Japan", "Italy", "United Arab Emirates", "Canada",
}
SH = {"AUS", "NZ", "Australia", "New Zealand", "SAF", "South Africa"}

FIELD_KEYS = {
    "horse": ("horse",),
    "trainer": ("trainer",),
    "jockey": ("jockey",),
    "weight": ("weight",),
    "barrier": ("barrier",),
    "place": ("placing", "place"),
    "number": ("number", "no.", "saddle"),
}


def _clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text)          # footnotes [a], [5]
    text = text.replace("†", "").replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def _cell_text(td) -> str:
    for br in td.find_all("br"):
        br.replace_with(" & ")
    parts = [p.strip() for p in _clean(td.get_text(" ")).split("&")]
    return " & ".join(p for p in parts if p)


def _expanded_headers(row) -> list[str]:
    heads = []
    for th in row.find_all(["th", "td"]):
        h = _clean(th.get_text(" ")).lower()
        heads.extend([h] * int(th.get("colspan", 1) or 1))
    return heads


def _match(header: str) -> str | None:
    for field, keys in FIELD_KEYS.items():
        if any(header.startswith(k) for k in keys):
            return field
    return None


def parse_place(text: str) -> float:
    m = re.match(r"^(\d+)", text.strip())
    return float(m.group(1)) if m else float("nan")   # DNF, Fell, Scratched...


def parse_float(text: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", text.replace(",", "."))
    return float(m.group(0)) if m else float("nan")


def _origin(horse_cells: list) -> tuple[str, str | None]:
    """Return (horse name, breeding origin) from the horse cell(s)."""
    name, origin = "", None
    for td in horse_cells:
        img = td.select_one(".flagicon img")
        if img is not None and img.get("alt"):
            origin = img["alt"]
        txt = _cell_text(td)
        if txt:
            name = txt
    m = re.search(r"\(([A-Za-z]{2,4})\)\s*$", name)
    if m:
        origin = {"AUST": "AUS"}.get(m.group(1).upper(), m.group(1).upper())
        name = name[: m.start()].strip()
    return name, origin


def bred_north(origin: str | None) -> float:
    if origin is None:
        return float("nan")
    if origin in NH or origin.upper() in NH:
        return 1.0
    if origin in SH or origin.upper() in SH:
        return 0.0
    return float("nan")


def parse_year(year: int, html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    best = None
    for table in soup.select("table.wikitable"):
        rows = table.find_all("tr")
        if len(rows) < 11:
            continue
        heads = _expanded_headers(rows[0])
        fields = [_match(h) for h in heads]
        if "horse" in fields and ("trainer" in fields or "jockey" in fields):
            best = (rows, fields)
            break
    if best is None:
        return pd.DataFrame()
    rows, fields = best
    records = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        expanded = []
        for c in cells:
            expanded.extend([c] * int(c.get("colspan", 1) or 1))
        if len(expanded) < len(fields) - 1:
            continue
        bucket: dict[str, list] = {}
        for f, c in zip(fields, expanded):
            if f:
                bucket.setdefault(f, [])
                if c not in bucket[f]:
                    bucket[f].append(c)
        name, origin = _origin(bucket.get("horse", []))
        name = name.strip(" &")
        if not name:
            continue
        rec = {
            "year": year,
            "horse": name,
            "origin": origin,
            "bred_north": bred_north(origin),
            "trainer": " & ".join(_cell_text(c) for c in bucket.get("trainer", []) if _cell_text(c)),
            "jockey": " & ".join(_cell_text(c) for c in bucket.get("jockey", []) if _cell_text(c)),
            "weight": parse_float(_cell_text(bucket["weight"][-1])) if "weight" in bucket else float("nan"),
            "barrier": parse_float(_cell_text(bucket["barrier"][-1])) if "barrier" in bucket else float("nan"),
            "place": parse_place(_cell_text(bucket["place"][-1])) if "place" in bucket else float("nan"),
        }
        records.append(rec)
    df = pd.DataFrame(records)
    if not df.empty:
        # Pre-2000 tables list placegetters in finishing order with no number.
        if df["place"].isna().all() and year < 2026:
            df["place"] = range(1, len(df) + 1)
        df["field_size"] = len(df)
        # A malformed wiki cell can drop one barrier; if exactly one runner is
        # missing and exactly one stall number is unused, that is its barrier.
        miss = df["barrier"].isna()
        if miss.sum() == 1 and df["barrier"].notna().any():
            unused = set(range(1, len(df) + 1)) - set(df["barrier"].dropna().astype(int))
            if len(unused) == 1:
                df.loc[miss, "barrier"] = float(unused.pop())
        # Wikipedia convention: Australian-bred horses carry no suffix. Only
        # trust that where the page marks origins systematically (>= half the
        # field); otherwise blank the whole year. A feature that is blank for
        # a whole race cancels out of the model instead of feeding it errors
        # (e.g. 2013 marks one horse, which would brand Fiorente (IRE) as AUS).
        if df["origin"].notna().mean() >= 0.5:
            df["bred_north"] = df["bred_north"].fillna(0.0)
        else:
            df["bred_north"] = float("nan")
    return df


def build_dataset(pages: dict[int, str]) -> pd.DataFrame:
    frames = [parse_year(y, h) for y, h in sorted(pages.items())]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)
