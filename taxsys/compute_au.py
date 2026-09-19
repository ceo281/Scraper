"""Australian individual income tax computation.

Every figure used here is read from the rule pack. If a value is missing from
the pack this module raises rather than substituting a default, because a
plausible wrong number is worse than a visible gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rulepack import RulePack, RulePackError


@dataclass
class AUTaxpayer:
    assessable_income: float = 0.0
    business_income: float = 0.0
    deductions: float = 0.0
    is_resident: bool = True
    has_private_hospital_cover: bool = True
    has_spouse: bool = False
    dependants: int = 0
    help_debt: float = 0.0
    reportable_fringe_benefits: float = 0.0
    reportable_super_contributions: float = 0.0
    payg_withheld: float = 0.0
    payg_instalments_paid: float = 0.0


@dataclass
class LineItem:
    label: str
    amount: float
    cite: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "amount": round(self.amount, 2), "cite": self.cite, "note": self.note}


@dataclass
class AUAssessment:
    year: str
    taxable_income: float
    items: list[LineItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verified: bool = False

    @property
    def total_payable(self) -> float:
        return round(sum(i.amount for i in self.items), 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction": "AU",
            "year": self.year,
            "taxable_income": round(self.taxable_income, 2),
            "items": [i.as_dict() for i in self.items],
            "total": self.total_payable,
            "warnings": self.warnings,
            "verified": self.verified,
        }


def _require(node: dict[str, Any], key: str, where: str) -> Any:
    if key not in node:
        raise RulePackError(f"Rule pack is missing {where}.{key}. Refusing to assume a value.")
    return node[key]


def tax_on_income(brackets: list[dict[str, Any]], income: float) -> float:
    """Progressive tax using the pack's stated base amounts."""
    if income <= 0:
        return 0.0
    payable = 0.0
    for bracket in brackets:
        lower = max(0.0, float(bracket["from"]) - 1) if bracket["from"] else 0.0
        upper = bracket["to"]
        if upper is not None and income > float(upper):
            continue
        if income <= lower:
            continue
        payable = float(bracket.get("base", 0.0)) + (income - lower) * float(bracket["rate"])
        return round(payable, 2)
    top = brackets[-1]
    lower = float(top["from"]) - 1
    return round(float(top.get("base", 0.0)) + (income - lower) * float(top["rate"]), 2)


def medicare_levy(cfg: dict[str, Any], taxable_income: float, taxpayer: AUTaxpayer) -> tuple[float, str]:
    rate = float(_require(cfg, "rate", "medicare_levy"))
    if taxpayer.has_spouse or taxpayer.dependants:
        lower = float(cfg["family_threshold_lower"]) + float(cfg.get("family_dependant_increment", 0)) * taxpayer.dependants
    else:
        lower = float(cfg["single_threshold_lower"])
    upper = float(cfg["single_threshold_upper"]) if not (taxpayer.has_spouse or taxpayer.dependants) else lower * 1.25
    shade = float(cfg.get("shade_in_rate", 0.10))

    if taxable_income <= lower:
        return 0.0, f"Below the levy threshold of ${lower:,.0f}."
    if taxable_income >= upper:
        return round(taxable_income * rate, 2), f"Full {rate:.0%} levy."
    return round(min(taxable_income * rate, (taxable_income - lower) * shade), 2), (
        f"Shade-in band between ${lower:,.0f} and ${upper:,.0f}."
    )


def medicare_levy_surcharge(cfg: dict[str, Any], income_for_mls: float, taxpayer: AUTaxpayer) -> tuple[float, str]:
    if taxpayer.has_private_hospital_cover:
        return 0.0, "Not applicable, private hospital cover held for the full year."
    tiers = cfg["families_tiers"] if (taxpayer.has_spouse or taxpayer.dependants) else cfg["singles_tiers"]
    for tier in tiers:
        upper = tier["to"]
        if upper is None or income_for_mls <= float(upper):
            rate = float(tier["rate"])
            if rate == 0:
                return 0.0, "Below the surcharge threshold."
            return round(income_for_mls * rate, 2), (
                f"Surcharge at {rate:.2%}. Taking out private hospital cover would usually cost less than this."
            )
    return 0.0, ""


def low_income_tax_offset(cfg: dict[str, Any], taxable_income: float) -> tuple[float, str]:
    for band in cfg.get("taper", []):
        upper = band["to"]
        if upper is None or taxable_income <= float(upper):
            offset = float(band["offset"])
            rate = float(band.get("withdrawal_rate", 0.0))
            if rate:
                lower = max(0.0, float(band["from"]) - 1)
                offset = max(0.0, offset - (taxable_income - lower) * rate)
            return round(offset, 2), "Non-refundable. It can reduce tax to zero but never produce a refund on its own."
    return 0.0, "Income is above the offset cut-out."


def study_loan_repayment(cfg: dict[str, Any], repayment_income: float, has_debt: bool) -> tuple[float, str]:
    if not has_debt:
        return 0.0, ""
    threshold = float(cfg.get("threshold", 0))
    if repayment_income <= threshold:
        return 0.0, f"Repayment income is below the ${threshold:,.0f} threshold."
    if cfg.get("model") == "marginal":
        owed = 0.0
        for band in cfg.get("marginal_bands", []):
            lower = float(band["from"])
            upper = band["to"]
            if repayment_income <= lower:
                break
            top = repayment_income if upper is None else min(repayment_income, float(upper))
            owed += (top - lower) * float(band["rate"])
        return round(owed, 2), "Marginal model: the rate applies only to income above the threshold."
    return 0.0, "Repayment model not on record for this year."


def assess(pack: RulePack, taxpayer: AUTaxpayer, *, allow_unverified: bool = True) -> AUAssessment:
    if pack.jurisdiction != "AU":
        raise RulePackError(f"compute_au requires an AU rule pack, got {pack.jurisdiction}")
    pack.assert_usable(allow_unverified=allow_unverified)

    gross = taxpayer.assessable_income + taxpayer.business_income
    taxable_income = max(0.0, gross - taxpayer.deductions)

    result = AUAssessment(year=pack.year, taxable_income=taxable_income, verified=pack.is_verified)
    if not pack.is_verified:
        result.warnings.append(
            "Computed from an unverified rule pack. Every rate and threshold must be confirmed "
            "against primary sources by a registered tax agent before these figures are lodged or relied on."
        )

    brackets_cfg = pack.rates["income_tax_brackets"]
    base_tax = tax_on_income(brackets_cfg["brackets"], taxable_income)
    result.items.append(LineItem("Income tax", base_tax, brackets_cfg.get("cite", "")))

    levy, levy_note = medicare_levy(pack.rates["medicare_levy"], taxable_income, taxpayer)
    result.items.append(LineItem("Medicare levy", levy, pack.rates["medicare_levy"].get("cite", ""), levy_note))

    income_for_mls = taxable_income + taxpayer.reportable_fringe_benefits + taxpayer.reportable_super_contributions
    mls, mls_note = medicare_levy_surcharge(pack.rates["medicare_levy_surcharge"], income_for_mls, taxpayer)
    if mls or not taxpayer.has_private_hospital_cover:
        result.items.append(
            LineItem("Medicare levy surcharge", mls, pack.rates["medicare_levy_surcharge"].get("cite", ""), mls_note)
        )

    lito, lito_note = low_income_tax_offset(pack.rates["low_income_tax_offset"], taxable_income)
    lito = min(lito, base_tax)  # non-refundable, cannot exceed the tax it offsets
    if lito:
        result.items.append(
            LineItem("Low income tax offset", -lito, pack.rates["low_income_tax_offset"].get("cite", ""), lito_note)
        )

    repayment_income = taxable_income + taxpayer.reportable_fringe_benefits + taxpayer.reportable_super_contributions
    help_amount, help_note = study_loan_repayment(
        pack.rates.get("study_loan_repayment", {}), repayment_income, taxpayer.help_debt > 0
    )
    if help_amount:
        result.items.append(
            LineItem("Study and training loan repayment", help_amount,
                     pack.rates.get("study_loan_repayment", {}).get("cite", ""), help_note)
        )
        result.warnings.append(
            "The study loan repayment model changed for FY2025-26. Confirm the threshold and marginal bands before relying on this figure."
        )

    credits = taxpayer.payg_withheld + taxpayer.payg_instalments_paid
    if credits:
        result.items.append(LineItem("PAYG credits already paid", -credits, "TAA 1953 Sch 1", "Amounts withheld or paid during the year."))

    return result
