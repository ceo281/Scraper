"""Rule pack loading, with a verification gate.

The central design decision of this system: no tax figure is ever written into
code. Rates, thresholds, caps and tests live in rulepacks/<jurisdiction>/*.json,
each carrying a `cite` (the statutory authority), a `source_url`, and a
`verified` flag that defaults to false.

Nothing that has not been verified by a credentialed professional may be used to
produce a figure presented as final. `RulePack.assert_usable` enforces that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
RULEPACK_ROOT = REPO_ROOT / "rulepacks"

JURISDICTIONS = ("AU", "US")


class RulePackError(RuntimeError):
    """Raised when a rule pack is missing, malformed, or used outside its scope."""


class UnverifiedRulePackError(RulePackError):
    """Raised when unverified figures are used in a context requiring final numbers."""


@dataclass(frozen=True)
class Citation:
    """A pointer back to the authority for a figure, carried with the figure."""

    text: str
    url: str | None = None
    verified: bool = False

    def __str__(self) -> str:
        return self.text


@dataclass
class RulePack:
    """One jurisdiction's rules for one tax year, plus its jurisdiction-wide files."""

    jurisdiction: str
    year: str
    meta: dict[str, Any]
    rates: dict[str, Any]
    deductions: dict[str, Any]
    positions: dict[str, Any]
    indirect: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, jurisdiction: str, year: str) -> "RulePack":
        jurisdiction = jurisdiction.upper()
        if jurisdiction not in JURISDICTIONS:
            raise RulePackError(
                f"Unsupported jurisdiction {jurisdiction!r}. "
                f"This system covers {', '.join(JURISDICTIONS)} only."
            )
        base = RULEPACK_ROOT / jurisdiction.lower()
        if not base.is_dir():
            raise RulePackError(f"No rule pack directory at {base}")

        rates_name = f"rates_{year.replace('-', '_')}.json"
        rates_path = base / rates_name
        if not rates_path.exists():
            available = sorted(p.stem.replace("rates_", "") for p in base.glob("rates_*.json"))
            raise RulePackError(
                f"No rates on record for {jurisdiction} {year}. "
                f"Years available: {', '.join(available) or 'none'}. "
                f"This system will not substitute another year's figures."
            )

        indirect_path = base / "gst_bas.json"
        return cls(
            jurisdiction=jurisdiction,
            year=year,
            meta=_read(base / "meta.json"),
            rates=_read(rates_path),
            deductions=_read(base / "deductions.json"),
            positions=_read(base / "positions.json"),
            indirect=_read(indirect_path) if indirect_path.exists() else {},
        )

    @classmethod
    def available_years(cls, jurisdiction: str) -> list[str]:
        base = RULEPACK_ROOT / jurisdiction.lower()
        if not base.is_dir():
            return []
        return sorted(p.stem[len("rates_"):].replace("_", "-") for p in base.glob("rates_*.json"))

    # ----------------------------------------------------------- verification

    @property
    def is_verified(self) -> bool:
        return self.meta.get("verification", {}).get("status") == "verified"

    def unverified_entries(self) -> list[str]:
        """Every path in the loaded pack still carrying verified=false."""
        found: list[str] = []
        for name, doc in (
            ("meta", self.meta),
            (f"rates_{self.year}", self.rates),
            ("deductions", self.deductions),
            ("positions", self.positions),
            ("indirect", self.indirect),
        ):
            if not doc:
                continue
            for path in _walk_unverified(doc, name):
                found.append(path)
        return found

    def assert_usable(self, *, allow_unverified: bool = False) -> None:
        """Gate. Call before presenting any figure as final."""
        if self.is_verified or allow_unverified:
            return
        raise UnverifiedRulePackError(
            f"The {self.jurisdiction} {self.year} rule pack has not been verified "
            f"against primary sources. {len(self.unverified_entries())} entries are "
            f"still flagged unverified. Figures may be computed for review but must "
            f"not be presented as final or lodged. "
            f"Sign-off required by: {self.meta.get('verification', {}).get('signoff_required_by', 'a credentialed professional')}."
        )

    # ------------------------------------------------------------- accessors

    def cite_for(self, *path: str) -> Citation:
        """Pull the citation attached to a figure, so answers can show their authority."""
        node: Any = self.rates
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return Citation(text="No citation on record", verified=False)
            node = node[key]
        if isinstance(node, dict):
            return Citation(
                text=node.get("cite", "No citation on record"),
                url=node.get("source_url"),
                verified=bool(node.get("verified", False)),
            )
        return Citation(text="No citation on record", verified=False)

    def deduction_categories(self) -> list[dict[str, Any]]:
        return list(self.deductions.get("categories", []))

    def category(self, code: str) -> dict[str, Any] | None:
        for cat in self.deduction_categories():
            if cat.get("code") == code:
                return cat
        return None

    def strategies(self) -> list[dict[str, Any]]:
        return list(self.positions.get("strategies", []))

    def blocked_patterns(self) -> set[str]:
        return {p["id"] for p in self.positions.get("blocked_patterns", [])}

    def anti_avoidance(self) -> list[dict[str, Any]]:
        return list(self.positions.get("anti_avoidance", []))


# ------------------------------------------------------------------ internals


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RulePackError(f"Rule pack file missing: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise RulePackError(f"Malformed rule pack at {path}: {exc}") from exc


def _walk_unverified(node: Any, prefix: str) -> Iterator[str]:
    if isinstance(node, dict):
        if node.get("verified") is False:
            yield prefix
        if node.get("verification", {}).get("status") == "unverified":
            yield f"{prefix}.verification"
        for key, value in node.items():
            if key in {"verified", "verification"}:
                continue
            yield from _walk_unverified(value, f"{prefix}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_unverified(value, f"{prefix}[{index}]")
