"""The cited corpus, and the binding between rules and source text.

This module is the answer to two questions that decide whether the system is
credible at all:

  How is it up to date?   Every rule record stores the hash of the exact source
                          span it was derived from. Re-ingestion re-hashes. A
                          changed hash invalidates the rule automatically and
                          the system refuses to serve it until a human re-reads
                          the amended text.

  How is it credible?     An answer carries the actual words of the provision,
                          not a paraphrase of them. The rule says what it means;
                          the source span proves it says that.

A rule whose anchor is broken is worse than no rule, so a broken anchor is a
refusal, never a silent fallback to the stale text.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = REPO_ROOT / "corpus"

# A position cannot rest on a plain-language guide alone. IRS publications and
# ATO web guidance explain the law but are not the law, so they are retrievable
# and quotable but can never be the sole authority for an answer.
AUTHORITY_RANK = {
    "statute": 100, "regulation": 90, "case": 80,
    "ruling": 70, "determination": 65, "guideline": 60,
    "publication": 30, "guide": 20,
}
BINDING_KINDS = {"statute", "regulation", "case", "ruling", "determination"}


class BrokenAnchorError(RuntimeError):
    """A rule points at source text that has changed or vanished."""


@dataclass
class Chunk:
    """One addressable span of primary source text, effective-dated."""

    id: str
    jurisdiction: str
    kind: str
    citation: str
    heading: str
    text: str
    source_url: str = ""
    compilation_id: str = ""
    effective_from: str = "1900-01-01"
    effective_to: str | None = None
    span_hash: str = ""
    superseded_by: str | None = None
    fetched_at: str = ""

    def __post_init__(self) -> None:
        if not self.span_hash:
            self.span_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @property
    def authority_rank(self) -> int:
        return AUTHORITY_RANK.get(self.kind, 10)

    @property
    def is_binding_authority(self) -> bool:
        return self.kind in BINDING_KINDS

    def in_force_on(self, when: date) -> bool:
        if when < date.fromisoformat(self.effective_from):
            return False
        if self.effective_to and when > date.fromisoformat(self.effective_to):
            return False
        return True

    def in_force_for_year(self, jurisdiction: str, year_label: str) -> bool:
        """Date scoping. A FY2024-25 question gets the law as it stood then."""
        from .periods import period_for

        try:
            period = period_for(jurisdiction, year_label)
        except (ValueError, KeyError):
            return True
        return self.in_force_on(period.end)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Anchor:
    """Binds one rule pack record to the source span it was derived from.

    The hash is the whole point. Without it a citation is decoration.
    """

    rule_id: str
    chunk_id: str
    citation: str
    span_hash: str
    bound_at: str
    bound_by: str = "unreviewed"
    note: str = ""

    def verify(self, corpus: "Corpus") -> tuple[bool, str]:
        chunk = corpus.get(self.chunk_id)
        if chunk is None:
            return False, (
                f"Source span {self.chunk_id} is no longer in the corpus. "
                f"Rule {self.rule_id} cannot be served until it is re-anchored."
            )
        if chunk.span_hash != self.span_hash:
            return False, (
                f"{self.citation} has changed since rule {self.rule_id} was written "
                f"(anchored to {self.span_hash}, source is now {chunk.span_hash}). "
                f"The rule is suspended pending human review of the amended text."
            )
        return True, "Anchor intact."

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Corpus:
    """In-memory corpus. The interface is what matters; swap the store for
    pgvector or Elasticsearch at scale without touching callers."""

    def __init__(self, chunks: Iterable[Chunk] | None = None) -> None:
        self.chunks: list[Chunk] = list(chunks or [])
        self._by_id = {c.id: c for c in self.chunks}
        self.anchors: dict[str, Anchor] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
        self._by_id[chunk.id] = chunk

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def bind(self, anchor: Anchor) -> None:
        self.anchors[anchor.rule_id] = anchor

    # ------------------------------------------------------- invalidation

    def reconcile(self) -> dict[str, list[str]]:
        """Re-verify every anchor after an ingest. This is the freshness run.

        Returns rules that remain valid and rules now suspended, with reasons.
        """
        intact: list[str] = []
        broken: list[str] = []
        reasons: list[str] = []
        for rule_id, anchor in sorted(self.anchors.items()):
            ok, reason = anchor.verify(self)
            if ok:
                intact.append(rule_id)
            else:
                broken.append(rule_id)
                reasons.append(reason)
        return {"intact": intact, "suspended": broken, "reasons": reasons}

    def is_servable(self, rule_id: str) -> tuple[bool, str]:
        """Called before any answer uses a rule. A broken anchor refuses."""
        anchor = self.anchors.get(rule_id)
        if anchor is None:
            return False, (
                f"Rule {rule_id} is not bound to any source text. "
                f"It cannot be used in an answer until it is anchored to a provision."
            )
        return anchor.verify(self)

    # ------------------------------------------------------------ storage

    def save(self, root: Path | None = None) -> None:
        root = root or CORPUS_ROOT
        by_jurisdiction: dict[str, list[Chunk]] = {}
        for chunk in self.chunks:
            by_jurisdiction.setdefault(chunk.jurisdiction.lower(), []).append(chunk)
        for jurisdiction, chunks in by_jurisdiction.items():
            target = root / jurisdiction / "chunks.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps({"chunks": [c.as_dict() for c in chunks]}, indent=2), encoding="utf-8"
            )
        if self.anchors:
            anchors_path = root / "anchors.json"
            anchors_path.parent.mkdir(parents=True, exist_ok=True)
            anchors_path.write_text(
                json.dumps({"anchors": [a.as_dict() for a in self.anchors.values()]}, indent=2),
                encoding="utf-8",
            )

    @classmethod
    def load(cls, jurisdiction: str | None = None, root: Path | None = None) -> "Corpus":
        root = root or CORPUS_ROOT
        corpus = cls()
        if not root.is_dir():
            return corpus
        dirs = [root / jurisdiction.lower()] if jurisdiction else [p for p in sorted(root.iterdir()) if p.is_dir()]
        for directory in dirs:
            path = directory / "chunks.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for raw in payload.get("chunks", []):
                corpus.add(Chunk(**raw))
        anchors_path = root / "anchors.json"
        if anchors_path.exists():
            payload = json.loads(anchors_path.read_text(encoding="utf-8"))
            for raw in payload.get("anchors", []):
                corpus.bind(Anchor(**raw))
        return corpus
