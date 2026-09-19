"""Expense classification against a jurisdiction's deduction rules.

Deterministic and explainable by design. Every result carries the rule that
produced it and the authority behind that rule, so a classification can be
argued with rather than merely trusted. A language model may be layered on top
to propose a category, but it never decides one: it proposes, this engine
adjudicates against the rule pack, and the rule pack cites the statute.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .rulepack import RulePack

# Categories that exist to say "no". Matching one is a finding, not a failure.
_DENIAL_MARKERS = ("NONDED", "ENTERTAINMENT", "FINES", "COMMUTING", "POLITICAL")


@dataclass
class ExpenseLine:
    """One spend item, however it arrived: receipt, bank feed, or manual entry."""

    description: str
    amount: float
    date: str
    merchant: str = ""
    note: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    business_use_percent: float | None = None
    id: str | None = None

    @property
    def haystack(self) -> str:
        return f"{self.merchant} {self.description} {self.note}".lower()


@dataclass
class Classification:
    """The engine's answer for one expense line."""

    line: ExpenseLine
    code: str | None
    name: str
    schedule_or_label: str
    deductible_percent: float
    confidence: float
    reasoning: list[str]
    citation: str
    source_url: str | None
    risk: str
    deductible: bool
    matched_keywords: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    needs_human_review: bool = False
    apportionment_required: bool = False

    @property
    def apportionment_unknown(self) -> bool:
        """Mixed-use category with no business percentage supplied yet."""
        return self.apportionment_required and self.line.business_use_percent is None

    @property
    def ceiling_amount(self) -> float:
        """The most this line could ever be worth, at 100% business use."""
        if not self.deductible:
            return 0.0
        return round(self.line.amount * self.deductible_percent, 2)

    @property
    def claimable_amount(self) -> float | None:
        """None means not yet determinable, which is not the same as zero.

        Returning the full amount when business use is unknown would silently
        overstate the claim, so a mixed-use line stays None until apportioned.
        """
        if not self.deductible:
            return 0.0
        if self.apportionment_unknown:
            return None
        business = self.line.business_use_percent
        share = 1.0 if business is None else max(0.0, min(business, 100.0)) / 100.0
        return round(self.line.amount * self.deductible_percent * share, 2)

    @property
    def status(self) -> str:
        """The single word shown on the line in the UI."""
        if not self.deductible:
            return "not_deductible"
        if self.missing_evidence or self.apportionment_unknown:
            return "incomplete"
        if self.needs_human_review:
            return "review"
        return "ready"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.line.id,
            "merchant": self.line.merchant,
            "description": self.line.description,
            "date": self.line.date,
            "amount": round(self.line.amount, 2),
            "code": self.code,
            "name": self.name,
            "schedule_or_label": self.schedule_or_label,
            "deductible": self.deductible,
            "deductible_percent": self.deductible_percent,
            "business_use_percent": self.line.business_use_percent,
            "claimable_amount": self.claimable_amount,
            "ceiling_amount": self.ceiling_amount,
            "apportionment_required": self.apportionment_required,
            "apportionment_unknown": self.apportionment_unknown,
            "status": self.status,
            "confidence": round(self.confidence, 2),
            "citation": self.citation,
            "source_url": self.source_url,
            "risk": self.risk,
            "reasoning": self.reasoning,
            "matched_keywords": self.matched_keywords,
            "missing_evidence": self.missing_evidence,
            "warnings": self.warnings,
            "needs_human_review": self.needs_human_review,
        }


class Classifier:
    def __init__(self, pack: RulePack) -> None:
        self.pack = pack
        self._categories = pack.deduction_categories()

    # ------------------------------------------------------------------ main

    def classify(self, line: ExpenseLine) -> Classification:
        scored = sorted(
            (self._score(line, cat) for cat in self._categories),
            key=lambda pair: pair[0],
            reverse=True,
        )
        best_score, best_cat, hits = scored[0] if scored else (0.0, None, [])

        if best_cat is None or best_score <= 0:
            return self._unclassified(line)

        code = best_cat.get("code", "")
        is_denial = any(marker in code.upper() for marker in _DENIAL_MARKERS)
        percent = float(best_cat.get("deductible_percent", 0.0 if is_denial else 1.0))
        deductible = percent > 0

        reasoning = self._explain(best_cat, hits, deductible, percent)
        missing = self._missing_evidence(line, best_cat)
        warnings = self._warnings(line, best_cat, deductible)

        confidence = min(0.95, best_score / 4.0)
        # An unapportioned mixed-use claim is the most common audit trigger in
        # both jurisdictions, so never present one as high confidence.
        if deductible and line.business_use_percent is None and best_cat.get("risk") == "high":
            confidence = min(confidence, 0.55)

        return Classification(
            line=line,
            code=code,
            name=best_cat.get("name", "Unnamed category"),
            schedule_or_label=self._label(best_cat),
            deductible_percent=percent,
            confidence=confidence,
            reasoning=reasoning,
            citation=best_cat.get("cite", "No citation on record"),
            source_url=best_cat.get("source_url"),
            risk=best_cat.get("risk", "unknown"),
            deductible=deductible,
            matched_keywords=hits,
            missing_evidence=missing,
            warnings=warnings,
            needs_human_review=bool(missing) or confidence < 0.5 or best_cat.get("risk") == "high",
            apportionment_required=self._needs_apportionment(best_cat),
        )

    def classify_all(self, lines: list[ExpenseLine]) -> list[Classification]:
        return [self.classify(line) for line in lines]

    # -------------------------------------------------------------- scoring

    def _score(self, line: ExpenseLine, cat: dict[str, Any]) -> tuple[float, dict[str, Any], list[str]]:
        haystack = line.haystack
        hits: list[str] = []
        score = 0.0
        for keyword in cat.get("keywords", []):
            if re.search(rf"\b{re.escape(keyword.lower())}", haystack):
                hits.append(keyword)
                # A merchant-name match is stronger evidence than a loose
                # mention buried in a free-text note.
                score += 2.0 if keyword.lower() in line.merchant.lower() else 1.0
        for sub in cat.get("subcategories", []):
            if sub.get("id", "").replace("_", " ") in haystack:
                hits.append(sub["id"])
                score += 1.5
        # Denial categories win ties. Telling someone a claim fails is more
        # useful than quietly filing it under something that passes.
        if hits and any(marker in cat.get("code", "").upper() for marker in _DENIAL_MARKERS):
            score += 0.5
        return score, cat, hits

    # ---------------------------------------------------------- explanation

    def _explain(self, cat: dict[str, Any], hits: list[str], deductible: bool, percent: float) -> list[str]:
        out: list[str] = []
        if hits:
            out.append(f"Matched on {', '.join(sorted(set(hits))[:4])}.")
        if not deductible:
            rule = cat.get("rule") or "This category is not deductible."
            out.append(rule)
            for item in cat.get("never_deductible", [])[:3]:
                out.append(f"Not deductible: {item}")
        else:
            if percent < 1.0:
                out.append(f"Deductible at {percent:.0%} of the amount, not in full.")
            for item in cat.get("deductible_when", [])[:3]:
                out.append(f"Deductible when: {item}")
            if cat.get("rule"):
                out.append(cat["rule"])
        if cat.get("risk_note"):
            out.append(cat["risk_note"])
        return out

    def _missing_evidence(self, line: ExpenseLine, cat: dict[str, Any]) -> list[str]:
        """What a photograph of a receipt cannot prove."""
        required = cat.get("substantiation")
        if isinstance(required, str) or required is None:
            required = []
        # The line already carries some of what the rules ask for. Reporting
        # "date" missing on a record that has a date destroys trust in the
        # whole list.
        known = dict(line.evidence)
        if line.date:
            known.setdefault("date", True)
        if line.amount:
            known.setdefault("amount", True)
        if line.merchant:
            known.setdefault("place", True)
        missing = [item for item in required if not known.get(item)]
        if cat.get("methods") and not line.evidence.get("method"):
            missing.append("which calculation method you are using")
        if self._needs_apportionment(cat) and line.business_use_percent is None:
            missing.append("business use percentage")
        return missing

    @staticmethod
    def _needs_apportionment(cat: dict[str, Any]) -> bool:
        """Categories where a private portion is realistically possible.

        A denied category needs no apportionment, and neither does a category
        that cannot have a private component in the first place.
        """
        if float(cat.get("deductible_percent", 1.0)) <= 0:
            return False
        if cat.get("apportionment_required") is not None:
            return bool(cat["apportionment_required"])
        return cat.get("risk") in {"high", "medium"}

    def _warnings(self, line: ExpenseLine, cat: dict[str, Any], deductible: bool) -> list[str]:
        out: list[str] = []
        if deductible and line.business_use_percent == 100 and cat.get("risk") == "high":
            out.append(
                "Claimed at 100% business use in a high-risk category. "
                "Both the ATO and the IRS treat an unapportioned mixed-use claim as a review trigger."
            )
        if cat.get("cgt_warning"):
            out.append(cat["cgt_warning"])
        for sub in cat.get("subcategories", []):
            if sub.get("cgt_warning") and sub.get("id", "") in line.haystack:
                out.append(sub["cgt_warning"])
        threshold = (self.pack.deductions.get("general_test") or {}).get("employee_note")
        if threshold and self.pack.jurisdiction == "US" and "employee" in line.note.lower():
            out.append(threshold)
        return out

    def _label(self, cat: dict[str, Any]) -> str:
        if self.pack.jurisdiction == "AU":
            return cat.get("label", "")
        schedule = cat.get("schedule")
        line_no = cat.get("line")
        if schedule and line_no:
            return f"Schedule {schedule}, line {line_no}"
        return f"Schedule {schedule}" if schedule else ""

    def _unclassified(self, line: ExpenseLine) -> Classification:
        return Classification(
            line=line,
            code=None,
            name="Unclassified",
            schedule_or_label="",
            deductible_percent=0.0,
            confidence=0.0,
            reasoning=[
                "No rule in the pack matched this description.",
                "This system will not guess a category. Add a note describing the "
                "business purpose, or assign a category manually.",
            ],
            citation=self.pack.deductions.get("general_test", {}).get("cite", ""),
            source_url=None,
            risk="unknown",
            deductible=False,
            needs_human_review=True,
        )
