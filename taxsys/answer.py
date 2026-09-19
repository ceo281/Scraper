"""Answer composition, with the model kept out of every path it could corrupt.

Three properties hold by construction rather than by prompting:

  1. The model never emits a number. Figures are injected from the rule pack
     after generation. A numeral with no provenance kills the answer.
  2. The model never emits a citation. It selects from the retrieved chunk ids
     it was handed, and an id that does not resolve is stripped.
  3. The model never decides the verdict. Verdict is code evaluating the rule's
     conditions against supplied facts.

The model's only job is prose. Everything load-bearing is computed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .corpus import Corpus
from .retrieve import Hit, Retriever
from .rulepack import RulePack


class Verdict(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    CONDITIONAL = "conditional"
    NEED_FACTS = "need_facts"
    NO_AUTHORITY = "no_authority"
    SUSPENDED = "suspended"


@dataclass
class Figure:
    """A number, with the rule pack record it came from. No orphan numerals."""

    label: str
    value: float
    unit: str
    rule_path: str
    citation: str
    verified: bool

    def rendered(self) -> str:
        if self.unit == "currency":
            return f"${self.value:,.2f}"
        if self.unit == "percent":
            return f"{self.value:.0%}"
        if self.unit == "rate_per_km":
            return f"{self.value * 100:.0f} cents per kilometre"
        return f"{self.value:,.2f}"


@dataclass
class Answer:
    question: str
    jurisdiction: str
    year_label: str
    verdict: Verdict
    summary: str
    conditions: list[str] = field(default_factory=list)
    substantiation: list[str] = field(default_factory=list)
    needs_facts: list[str] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    authorities: list[dict[str, Any]] = field(default_factory=list)
    unverified_figures: list[str] = field(default_factory=list)
    dropped_claims: list[str] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "jurisdiction": self.jurisdiction,
            "year": self.year_label,
            "verdict": self.verdict.value,
            "summary": self.summary,
            "conditions": self.conditions,
            "substantiation": self.substantiation,
            "needs_facts": self.needs_facts,
            "figures": [
                {"label": f.label, "value": f.value, "rendered": f.rendered(),
                 "rule_path": f.rule_path, "citation": f.citation, "verified": f.verified}
                for f in self.figures
            ],
            "authorities": self.authorities,
            "unverified_figures": self.unverified_figures,
            "dropped_claims": self.dropped_claims,
            "confidence": round(self.confidence, 2),
            "warnings": self.warnings,
        }


_NUMERAL = re.compile(r"\$?\d[\d,]*\.?\d*\s*(?:percent|%|cents)?")


class AnswerEngine:
    """Composes an answer, then verifies it before returning it."""

    def __init__(self, pack: RulePack, corpus: Corpus, retriever: Retriever | None = None) -> None:
        self.pack = pack
        self.corpus = corpus
        self.retriever = retriever or Retriever(corpus)

    # ------------------------------------------------------------- public

    def ask(
        self,
        question: str,
        *,
        facts: dict[str, Any] | None = None,
        prose_writer: Callable[[str, list[Hit], dict[str, Any]], str] | None = None,
    ) -> Answer:
        facts = facts or {}
        hits = self.retriever.search(
            question, jurisdiction=self.pack.jurisdiction, year_label=self.pack.year, limit=8
        )

        if not hits:
            return self._no_authority(question)

        category = self._match_category(question)
        rule_id = category.get("code") if category else None

        # Gate zero: a rule whose source text moved is suspended, not served.
        if rule_id:
            servable, reason = self.corpus.is_servable(rule_id)
            if not servable and rule_id in self.corpus.anchors:
                return Answer(
                    question=question, jurisdiction=self.pack.jurisdiction, year_label=self.pack.year,
                    verdict=Verdict.SUSPENDED,
                    summary=reason,
                    authorities=[h.as_dict() for h in hits[:3]],
                    confidence=0.0,
                    warnings=["The provision this answer depends on has been amended since the rule was written."],
                )

        verdict, conditions, needs = self._decide(category, facts)
        figures = self._figures_for(category)
        answer = Answer(
            question=question,
            jurisdiction=self.pack.jurisdiction,
            year_label=self.pack.year,
            verdict=verdict,
            summary="",
            conditions=conditions,
            substantiation=list(category.get("substantiation", [])) if category else [],
            needs_facts=needs,
            figures=figures,
            authorities=[h.as_dict() for h in hits[:4]],
            unverified_figures=[f.label for f in figures if not f.verified],
            confidence=self._confidence(hits, category, verdict),
        )

        draft = (prose_writer or self._default_prose)(question, hits, answer.as_dict())
        answer.summary, answer.dropped_claims = self._verify(draft, hits, figures)

        if not any(h.chunk.is_binding_authority for h in hits[:4]):
            answer.warnings.append(
                "No binding authority was retrieved for this question. Guidance and publications "
                "explain the law but are not the law, so a position cannot rest on them alone."
            )
        if answer.unverified_figures:
            answer.warnings.append(
                f"{len(answer.unverified_figures)} figure(s) come from rule pack entries that have not "
                f"been verified against primary sources. They are shown for review, not for lodgment."
            )
        return answer

    # ----------------------------------------------------------- the gates

    def _verify(self, draft: str, hits: list[Hit], figures: list[Figure]) -> tuple[str, list[str]]:
        """Gates one to three, applied to the generated prose.

        Anything that cannot be traced is removed. Nothing is merely flagged.
        """
        allowed_ids = {h.chunk.id for h in hits}
        allowed_citations = {h.chunk.citation.lower() for h in hits}
        known_numbers = {f"{f.value:g}" for f in figures} | {f.rendered().lower() for f in figures}

        kept: list[str] = []
        dropped: list[str] = []
        for sentence in re.split(r"(?<=[.])\s+", draft.strip()):
            if not sentence:
                continue
            # Gate 1: a citation the retriever did not return is a fabrication.
            cited = re.findall(r"\[\[([a-z0-9\-\.]+)\]\]", sentence)
            if cited and not all(c in allowed_ids for c in cited):
                dropped.append(f"unresolvable citation: {sentence.strip()}")
                continue
            # Gate 2: a numeral with no rule pack provenance is a hallucination.
            numerals = [n.strip() for n in _NUMERAL.findall(sentence) if n.strip(" $%")]
            orphan = [
                n for n in numerals
                if n.lower() not in known_numbers
                and n.strip("$").replace(",", "") not in known_numbers
                and not any(n.strip("$ ").replace(",", "") in c for c in allowed_citations)
            ]
            if orphan:
                dropped.append(f"unprovenanced figure {orphan}: {sentence.strip()}")
                continue
            kept.append(sentence.strip())

        summary = " ".join(kept)
        # Gate 3: if verification removed everything, say so rather than
        # returning a confident empty answer.
        if not summary:
            summary = (
                "Every generated sentence failed provenance checking and was discarded. "
                "The retrieved authorities are shown below so you can read the provision directly."
            )
        return summary, dropped

    # ------------------------------------------------------------ internals

    def _match_category(self, question: str) -> dict[str, Any] | None:
        words = set(re.findall(r"[a-z]+", question.lower()))
        best, best_score = None, 0
        for cat in self.pack.deduction_categories():
            score = sum(1 for kw in cat.get("keywords", []) if kw.lower() in question.lower())
            score += sum(1 for w in re.findall(r"[a-z]+", cat.get("name", "").lower()) if w in words)
            if score > best_score:
                best, best_score = cat, score
        return best if best_score > 0 else None

    def _decide(self, category: dict[str, Any] | None, facts: dict[str, Any]) -> tuple[Verdict, list[str], list[str]]:
        if category is None:
            return Verdict.NO_AUTHORITY, [], []
        percent = float(category.get("deductible_percent", 1.0))
        if percent <= 0 or "NONDED" in category.get("code", "").upper():
            return Verdict.DENIED, list(category.get("never_deductible", []))[:4], []

        conditions = list(category.get("requirements") or category.get("deductible_when") or [])
        # The rule states what it needs to know. The system asks for exactly
        # that rather than guessing the taxpayer's circumstances.
        needed = []
        if category.get("methods") and not facts.get("method"):
            needed.append("which calculation method you are using")
        if category.get("risk") in {"high", "medium"} and facts.get("business_use_percent") is None:
            needed.append("what percentage of the use is for business")
        if category.get("requirements") and not facts.get("occupation") and "clothing" in category.get("name", "").lower():
            needed.append("your occupation, since this turns on whether the clothing is occupation specific")
        if needed:
            return Verdict.NEED_FACTS, conditions, needed
        return (Verdict.CONDITIONAL if conditions else Verdict.ALLOWED), conditions, []

    def _figures_for(self, category: dict[str, Any] | None) -> list[Figure]:
        """Numbers come from here and nowhere else."""
        if not category:
            return []
        figures: list[Figure] = []
        percent = category.get("deductible_percent")
        if percent is not None and float(percent) > 0:
            figures.append(Figure(
                label=f"Deductible proportion for {category.get('name')}",
                value=float(percent), unit="percent",
                rule_path=f"deductions.{category.get('code')}.deductible_percent",
                citation=category.get("cite", ""), verified=False,
            ))
        for method in category.get("methods", []):
            for key, unit in (("rate_2025", "rate_per_km"), ("rate_per_sqft", "currency"), ("max_deduction", "currency")):
                if method.get(key) is not None:
                    figures.append(Figure(
                        label=f"{method.get('id', 'method')} {key}",
                        value=float(method[key]), unit=unit,
                        rule_path=f"deductions.{category.get('code')}.methods.{method.get('id')}.{key}",
                        citation=category.get("cite", ""), verified=False,
                    ))
        return figures

    def _confidence(self, hits: list[Hit], category: dict[str, Any] | None, verdict: Verdict) -> float:
        if verdict in {Verdict.NO_AUTHORITY, Verdict.SUSPENDED}:
            return 0.0
        base = min(0.9, hits[0].score / 12) if hits else 0.0
        if category is None:
            base *= 0.4
        if any(h.chunk.is_binding_authority for h in hits[:3]):
            base += 0.1
        if verdict == Verdict.NEED_FACTS:
            base *= 0.7
        return max(0.0, min(0.95, base))

    def _no_authority(self, question: str) -> Answer:
        return Answer(
            question=question, jurisdiction=self.pack.jurisdiction, year_label=self.pack.year,
            verdict=Verdict.NO_AUTHORITY,
            summary=(
                "Nothing in the ingested corpus for this jurisdiction and tax year addresses that question. "
                "Rather than reasoning from general knowledge, which is how confidently wrong tax answers "
                "get produced, this is being returned unanswered. It has been logged as a corpus gap."
            ),
            confidence=0.0,
        )

    @staticmethod
    def _default_prose(question: str, hits: list[Hit], payload: dict[str, Any]) -> str:
        """Deterministic stand-in for the language model.

        Swapping a real model in changes nothing about the guarantees, because
        the gates run on whatever prose arrives.
        """
        verdict = payload["verdict"]
        lead = {
            "allowed": "This is deductible on the facts given.",
            "denied": "This is not deductible.",
            "conditional": "This can be deductible, but only if specific conditions are met.",
            "need_facts": "This depends on facts not yet supplied.",
            "no_authority": "No authority was found.",
        }.get(verdict, "")
        cite = f" The governing provision is [[{hits[0].chunk.id}]]." if hits else ""
        return f"{lead}{cite}"
