"""Primary source adapters.

Every source here is a real, public, bulk-accessible feed. None of this is
scraping rendered HTML: each one is either a bulk download or a documented API,
which is what makes the corpus reproducible and diffable rather than brittle.

  US  uscode.house.gov     Title 26 as bulk XML, released per public law
      ecfr.gov API         Treasury Regulations, versioned with a change API
      irs.gov IRB          Revenue Rulings, Procedures and Notices

  AU  legislation.gov.au   Consolidated Acts with point-in-time compilations
      ato.gov.au legal db  Rulings (TR), Determinations (TD), Guidelines (PCG)

The point-in-time capability is the reason these specific endpoints were
chosen. A question about a prior tax year needs the law as it stood in that
year, not the current consolidation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

USER_AGENT = "taxsys-corpus-ingest/0.1 (compliance research; contact repository owner)"
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 4


class SourceUnavailable(RuntimeError):
    """Raised when a primary source cannot be reached.

    Deliberately fatal. A partial corpus that silently omits a statute is more
    dangerous than no corpus, because the retriever cannot distinguish 'this
    rule does not exist' from 'this rule failed to download'.
    """


@dataclass
class SourceDocument:
    """One fetched primary source document, before parsing."""

    source_id: str
    jurisdiction: str
    kind: str
    citation: str
    title: str
    url: str
    raw: bytes
    fetched_at: str
    effective_from: str = "1900-01-01"
    effective_to: str | None = None
    compilation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA of the source text as fetched.

        This is the freshness mechanism for the whole system. Every derived
        rule record stores the hash of the span it came from. When a re-fetch
        produces a different hash, every rule anchored to it is invalidated.
        """
        return hashlib.sha256(self.raw).hexdigest()

    def save(self, root: Path) -> Path:
        target = root / self.jurisdiction.lower() / "raw" / f"{self.source_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_id": self.source_id,
            "jurisdiction": self.jurisdiction,
            "kind": self.kind,
            "citation": self.citation,
            "title": self.title,
            "url": self.url,
            "fetched_at": self.fetched_at,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "compilation_id": self.compilation_id,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "raw_b64": self.raw.decode("utf-8", errors="replace"),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target


def http_get(url: str, *, timeout: int = DEFAULT_TIMEOUT, accept: str = "*/*") -> bytes:
    """Fetch with backoff. Raises SourceUnavailable rather than returning partial data."""
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise SourceUnavailable(f"{url} returned HTTP {response.status}")
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last = exc
            wait = 2 ** (attempt + 1)
            log.warning("fetch failed (%s), attempt %d/%d, retrying in %ds", exc, attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)
    raise SourceUnavailable(
        f"Could not reach {url} after {MAX_RETRIES} attempts: {last}. "
        f"Ingestion aborted rather than building a corpus with a silent gap."
    )


# ----------------------------------------------------------------- US sources


class USCodeSource:
    """IRC Title 26, bulk XML from the Office of the Law Revision Counsel.

    Release points are keyed by public law number, which gives genuine
    point-in-time versions: xml_usc26@119-23.zip is Title 26 as it stood at
    Public Law 119-23.
    """

    BASE = "https://uscode.house.gov/download/releasepoints/us/pl"
    RELEASE_INDEX = "https://uscode.house.gov/download/download.shtml"

    def __init__(self, congress: int = 119, law: int = 23) -> None:
        self.congress = congress
        self.law = law

    @property
    def url(self) -> str:
        return f"{self.BASE}/{self.congress}/{self.law}/xml_usc26@{self.congress}-{self.law}.zip"

    def fetch(self) -> SourceDocument:
        log.info("fetching IRC Title 26 at PL %d-%d", self.congress, self.law)
        raw = http_get(self.url, accept="application/zip")
        return SourceDocument(
            source_id=f"us-irc-title26-pl{self.congress}-{self.law}",
            jurisdiction="US",
            kind="statute",
            citation="26 U.S.C.",
            title=f"Internal Revenue Code, Title 26, at Public Law {self.congress}-{self.law}",
            url=self.url,
            raw=raw,
            fetched_at=date.today().isoformat(),
            compilation_id=f"PL{self.congress}-{self.law}",
            metadata={"format": "zip/xml", "release_point": f"{self.congress}-{self.law}"},
        )


class ECFRSource:
    """Treasury Regulations, 26 CFR, via the eCFR versioner API.

    The versioner exposes a date parameter, so a regulation can be retrieved as
    it stood on any given day, and /versions returns every amendment date.
    """

    API = "https://www.ecfr.gov/api/versioner/v1"

    def __init__(self, part: str, on_date: str | None = None) -> None:
        self.part = part
        self.on_date = on_date or date.today().isoformat()

    @property
    def url(self) -> str:
        return f"{self.API}/full/{self.on_date}/title-26.xml?part={self.part}"

    def versions_url(self) -> str:
        return f"{self.API}/versions/title-26.json?part={self.part}"

    def amendment_dates(self) -> list[str]:
        """Every date this part changed. Drives point-in-time ingestion."""
        payload = json.loads(http_get(self.versions_url(), accept="application/json"))
        return sorted({v["date"] for v in payload.get("content_versions", []) if v.get("date")})

    def fetch(self) -> SourceDocument:
        log.info("fetching 26 CFR part %s as at %s", self.part, self.on_date)
        raw = http_get(self.url, accept="application/xml")
        return SourceDocument(
            source_id=f"us-cfr26-{self.part}-{self.on_date}",
            jurisdiction="US",
            kind="regulation",
            citation=f"26 CFR {self.part}",
            title=f"Treasury Regulations, 26 CFR part {self.part}, as at {self.on_date}",
            url=self.url,
            raw=raw,
            fetched_at=date.today().isoformat(),
            effective_from=self.on_date,
            compilation_id=self.on_date,
            metadata={"format": "xml", "part": self.part},
        )


class IRBSource:
    """Internal Revenue Bulletin: Revenue Rulings, Procedures and Notices.

    This is where the indexed dollar figures actually live. IRC 63 states the
    standard deduction exists; the amount for a given year is in a Revenue
    Procedure. Any system that ingests only statute misses every number.
    """

    BASE = "https://www.irs.gov/pub/irs-irbs"

    def __init__(self, year: int, bulletin: int) -> None:
        self.year = year
        self.bulletin = bulletin

    @property
    def url(self) -> str:
        return f"{self.BASE}/irb{str(self.year)[2:]}-{self.bulletin:02d}.pdf"

    def fetch(self) -> SourceDocument:
        raw = http_get(self.url, accept="application/pdf")
        return SourceDocument(
            source_id=f"us-irb-{self.year}-{self.bulletin:02d}",
            jurisdiction="US",
            kind="ruling",
            citation=f"I.R.B. {self.year}-{self.bulletin}",
            title=f"Internal Revenue Bulletin {self.year}-{self.bulletin}",
            url=self.url,
            raw=raw,
            fetched_at=date.today().isoformat(),
            effective_from=f"{self.year}-01-01",
            metadata={"format": "pdf"},
        )


# ----------------------------------------------------------------- AU sources


class FRLSource:
    """Federal Register of Legislation, the authoritative source for Commonwealth Acts.

    Registered compilations are the point-in-time mechanism: each compilation
    carries the date range it was in force for, so FY2024-25 questions can be
    answered against the compilation in force during FY2024-25.
    """

    BASE = "https://www.legislation.gov.au"

    # Register ids for the Acts this system actually reasons about.
    ACTS = {
        "ITAA1997": ("C2004A05138", "Income Tax Assessment Act 1997"),
        "ITAA1936": ("C2004A07138", "Income Tax Assessment Act 1936"),
        "GSTACT":   ("C2004A00446", "A New Tax System (Goods and Services Tax) Act 1999"),
        "TAA1953":  ("C2004A07235", "Taxation Administration Act 1953"),
        "MLA1986":  ("C2004A03385", "Medicare Levy Act 1986"),
        "FBTAA1986":("C2004A03280", "Fringe Benefits Tax Assessment Act 1986"),
    }

    def __init__(self, act_key: str, compilation: str | None = None) -> None:
        if act_key not in self.ACTS:
            raise KeyError(f"Unknown Act {act_key!r}. Known: {', '.join(sorted(self.ACTS))}")
        self.act_key = act_key
        self.register_id, self.act_title = self.ACTS[act_key]
        self.compilation = compilation or "latest"

    @property
    def url(self) -> str:
        return f"{self.BASE}/{self.register_id}/{self.compilation}/text"

    def compilations_url(self) -> str:
        return f"{self.BASE}/{self.register_id}/compilations"

    def fetch(self) -> SourceDocument:
        log.info("fetching %s compilation %s", self.act_title, self.compilation)
        raw = http_get(self.url, accept="text/html,application/xhtml+xml")
        return SourceDocument(
            source_id=f"au-{self.act_key.lower()}-{self.compilation}",
            jurisdiction="AU",
            kind="statute",
            citation=self.act_title,
            title=f"{self.act_title} ({self.compilation} compilation)",
            url=self.url,
            raw=raw,
            fetched_at=date.today().isoformat(),
            compilation_id=self.compilation,
            metadata={"register_id": self.register_id, "act_key": self.act_key},
        )


class ATOLegalSource:
    """ATO Legal Database: Rulings, Determinations and Practical Compliance Guidelines.

    These carry most of the applied rules. The Act says a deduction must not be
    private or domestic; TR 2023/x and the PCGs say what that means for a home
    office, a uniform, or a car. Statute alone cannot answer a user's question.
    """

    BASE = "https://www.ato.gov.au/law/view/document"

    KINDS = {"TR": "ruling", "TD": "determination", "PCG": "guideline", "ATOID": "determination", "LCR": "ruling"}

    def __init__(self, doc_type: str, identifier: str) -> None:
        self.doc_type = doc_type.upper()
        self.identifier = identifier

    @property
    def url(self) -> str:
        return f"{self.BASE}?docid={self.doc_type}/{self.identifier}/NAT/ATO/00001"

    def fetch(self) -> SourceDocument:
        raw = http_get(self.url, accept="text/html")
        return SourceDocument(
            source_id=f"au-{self.doc_type.lower()}-{self.identifier.lower()}",
            jurisdiction="AU",
            kind=self.KINDS.get(self.doc_type, "guideline"),
            citation=f"{self.doc_type} {self.identifier}",
            title=f"{self.doc_type} {self.identifier}",
            url=self.url,
            raw=raw,
            fetched_at=date.today().isoformat(),
            metadata={"doc_type": self.doc_type},
        )


def default_manifest() -> list[dict[str, Any]]:
    """The documents this system needs to answer the questions it claims to answer."""
    return [
        {"source": "uscode", "args": {"congress": 119, "law": 23}},
        *[{"source": "ecfr", "args": {"part": part}} for part in ("1", "301")],
        *[{"source": "frl", "args": {"act_key": key}} for key in FRLSource.ACTS],
        *[
            {"source": "ato", "args": {"doc_type": dt, "identifier": ident}}
            for dt, ident in [
                ("PCG", "PCG20231"), ("TR", "TR20244"), ("TR", "TR9712"),
                ("TR", "TR985"), ("TR", "TR20214"), ("TD", "TD20243"),
            ]
        ],
    ]
