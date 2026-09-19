"""US federal individual income tax computation.

Federal only. State income tax is deliberately out of scope and is flagged as a
warning on every assessment rather than silently omitted, because for a
California or New York filer the state liability can exceed a third of the
federal number and its absence would make the total misleading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rulepack import RulePack, RulePackError

FILING_STATUSES = (
    "single",
    "married_filing_jointly",
    "married_filing_separately",
    "head_of_household",
)


@dataclass
class USTaxpayer:
    filing_status: str = "single"
    wages: float = 0.0
    federal_withheld: float = 0.0
    schedule_c_gross: float = 0.0
    schedule_c_expenses: float = 0.0
    other_ordinary_income: float = 0.0
    short_term_capital_gain: float = 0.0
    long_term_capital_gain: float = 0.0
    investment_income: float = 0.0
    itemized_deductions: float = 0.0
    state_local_taxes_paid: float = 0.0
    retirement_contributions: float = 0.0
    hsa_contributions: float = 0.0
    self_employed_health_premiums: float = 0.0
    estimated_payments: float = 0.0
    prior_year_total_tax: float = 0.0
    prior_year_agi: float = 0.0
    age_65_or_older: int = 0
    is_sstb: bool = False
    w2_wages_paid_by_business: float = 0.0

    @property
    def net_self_employment(self) -> float:
        return max(0.0, self.schedule_c_gross - self.schedule_c_expenses)


@dataclass
class LineItem:
    label: str
    amount: float
    cite: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "amount": round(self.amount, 2), "cite": self.cite, "note": self.note}


@dataclass
class USAssessment:
    year: str
    agi: float
    taxable_income: float
    deduction_used: str
    deduction_amount: float
    items: list[LineItem] = field(default_factory=list)
    adjustments: list[LineItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verified: bool = False
    safe_harbor: dict[str, Any] = field(default_factory=dict)

    @property
    def total_payable(self) -> float:
        return round(sum(i.amount for i in self.items), 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction": "US",
            "year": self.year,
            "agi": round(self.agi, 2),
            "taxable_income": round(self.taxable_income, 2),
            "deduction_used": self.deduction_used,
            "deduction_amount": round(self.deduction_amount, 2),
            "items": [i.as_dict() for i in self.items],
            "adjustments": [i.as_dict() for i in self.adjustments],
            "total": self.total_payable,
            "warnings": self.warnings,
            "verified": self.verified,
            "safe_harbor": self.safe_harbor,
        }


def tax_on_income(brackets: list[dict[str, Any]], income: float) -> float:
    """Marginal tax, summed band by band."""
    if income <= 0:
        return 0.0
    owed = 0.0
    for band in brackets:
        lower = float(band["from"])
        upper = band["to"]
        if income <= lower:
            break
        top = income if upper is None else min(income, float(upper))
        owed += (top - lower) * float(band["rate"])
    return round(owed, 2)


def self_employment_tax(cfg: dict[str, Any], net_se: float, wages: float, status: str) -> tuple[float, float, str]:
    """Returns (tax, deductible half, explanation)."""
    if net_se * float(cfg["net_earnings_factor"]) < float(cfg.get("minimum_net_earnings_to_owe", 400)):
        return 0.0, 0.0, "Net earnings below the filing threshold for self-employment tax."
    base = net_se * float(cfg["net_earnings_factor"])
    wage_base = float(cfg["social_security_wage_base"])
    ss_remaining = max(0.0, wage_base - wages)
    ss = min(base, ss_remaining) * float(cfg["social_security_rate"])
    medicare = base * float(cfg["medicare_rate"])

    threshold_key = {
        "married_filing_jointly": "additional_medicare_threshold_mfj",
        "married_filing_separately": "additional_medicare_threshold_mfs",
    }.get(status, "additional_medicare_threshold_single")
    threshold = float(cfg[threshold_key])
    additional = max(0.0, (base + wages) - threshold) * float(cfg["additional_medicare_rate"])

    total = round(ss + medicare + additional, 2)
    half = round((ss + medicare) / 2, 2)
    note = (
        f"Social Security stops at ${wage_base:,.0f} of combined wages and net earnings. "
        f"Half of the Social Security and Medicare portion (${half:,.2f}) is deductible above the line. "
        "The additional Medicare surtax is not deductible."
    )
    return total, half, note


def qbi_deduction(cfg: dict[str, Any], taxpayer: USTaxpayer, taxable_before_qbi: float) -> tuple[float, str]:
    qbi = taxpayer.net_self_employment
    if qbi <= 0:
        return 0.0, "No qualified business income."
    rate = float(cfg["rate"])
    joint = taxpayer.filing_status == "married_filing_jointly"
    threshold = float(cfg["threshold_mfj"] if joint else cfg["threshold_single"])
    phase_in = float(cfg["phase_in_range_mfj"] if joint else cfg["phase_in_range_single"])

    tentative = qbi * rate
    income_cap = max(0.0, (taxable_before_qbi - taxpayer.long_term_capital_gain)) * rate
    allowed = min(tentative, income_cap)

    if taxable_before_qbi <= threshold:
        return round(allowed, 2), (
            f"Taxable income is below the ${threshold:,.0f} threshold, so the full {rate:.0%} applies "
            "with no wage or property limitation."
        )

    over = taxable_before_qbi - threshold
    if taxpayer.is_sstb:
        if over >= phase_in:
            return 0.0, (
                f"This is a specified service trade or business and taxable income exceeds "
                f"${threshold + phase_in:,.0f}, so the deduction is fully phased out. "
                "The effective marginal rate inside the phase-out band can exceed 50%, which is worth modelling."
            )
        remaining = 1 - (over / phase_in)
        return round(allowed * remaining, 2), (
            f"Specified service trade or business inside the phase-out band. "
            f"{remaining:.0%} of the deduction survives."
        )

    wage_limit = taxpayer.w2_wages_paid_by_business * float(cfg["w2_wage_limit_rate"])
    if taxpayer.w2_wages_paid_by_business > 0 and over >= phase_in:
        allowed = min(allowed, wage_limit)
        return round(allowed, 2), (
            f"Above the phase-in range, so the deduction is capped at 50% of W-2 wages paid by the business "
            f"(${wage_limit:,.0f})."
        )
    if over >= phase_in and taxpayer.w2_wages_paid_by_business == 0:
        return 0.0, (
            "Above the phase-in range with no W-2 wages paid by the business, so the wage limitation "
            "reduces the deduction to zero. Paying reasonable W-2 wages may restore part of it."
        )
    return round(allowed, 2), "Inside the phase-in band. Confirm the wage and property limitation with a preparer."


def salt_allowed(cfg: dict[str, Any], paid: float, magi: float, status: str) -> tuple[float, str]:
    cap = float(cfg["cap_mfs"] if status == "married_filing_separately" else cfg["cap"])
    threshold = float(cfg["phasedown_magi_threshold"])
    floor = float(cfg["floor"])
    if magi > threshold:
        cap = max(floor, cap - (magi - threshold) * float(cfg["phasedown_rate"]))
    allowed = min(paid, cap)
    return round(allowed, 2), f"Capped at ${cap:,.0f} for this income level."


def estimated_tax_safe_harbor(cfg: dict[str, Any], taxpayer: USTaxpayer, current_year_tax: float) -> dict[str, Any]:
    high_threshold = float(
        cfg["high_agi_threshold_mfs"] if taxpayer.filing_status == "married_filing_separately" else cfg["high_agi_threshold"]
    )
    is_high = taxpayer.prior_year_agi > high_threshold
    rate = float(cfg["safe_harbor_prior_year_rate_high_agi"] if is_high else cfg["safe_harbor_prior_year_rate"])
    prior_route = round(taxpayer.prior_year_total_tax * rate, 2)
    current_route = round(current_year_tax * float(cfg["safe_harbor_current_year_rate"]), 2)
    required = min(prior_route, current_route) if taxpayer.prior_year_total_tax else current_route
    paid = taxpayer.estimated_payments + taxpayer.federal_withheld
    return {
        "cite": cfg.get("cite", "IRC 6654"),
        "prior_year_route": prior_route,
        "prior_year_rate": rate,
        "high_agi": is_high,
        "current_year_route": current_route,
        "required_to_avoid_penalty": required,
        "already_paid": round(paid, 2),
        "shortfall": round(max(0.0, required - paid), 2),
        "note": (
            f"Paying {rate:.0%} of last year's total tax avoids the underpayment penalty regardless of how much "
            "this year's income grows. That avoids the penalty, not the tax. The balance is still due on 15 April."
        ),
    }


def assess(pack: RulePack, taxpayer: USTaxpayer, *, allow_unverified: bool = True) -> USAssessment:
    if pack.jurisdiction != "US":
        raise RulePackError(f"compute_us requires a US rule pack, got {pack.jurisdiction}")
    if taxpayer.filing_status not in FILING_STATUSES:
        raise ValueError(f"Unknown filing status {taxpayer.filing_status!r}")
    pack.assert_usable(allow_unverified=allow_unverified)

    r = pack.rates
    se_tax, se_half, se_note = self_employment_tax(
        r["self_employment_tax"], taxpayer.net_self_employment, taxpayer.wages, taxpayer.filing_status
    )

    gross = (
        taxpayer.wages
        + taxpayer.net_self_employment
        + taxpayer.other_ordinary_income
        + taxpayer.short_term_capital_gain
        + taxpayer.long_term_capital_gain
        + taxpayer.investment_income
    )
    above_line = se_half + taxpayer.retirement_contributions + taxpayer.hsa_contributions + taxpayer.self_employed_health_premiums
    agi = max(0.0, gross - above_line)

    std_cfg = r["standard_deduction"]
    standard = float(std_cfg[taxpayer.filing_status])
    if taxpayer.age_65_or_older:
        extra_key = (
            "additional_age65_or_blind_married_each"
            if taxpayer.filing_status in {"married_filing_jointly", "married_filing_separately"}
            else "additional_age65_or_blind_unmarried"
        )
        standard += float(std_cfg[extra_key]) * taxpayer.age_65_or_older

    salt, salt_note = salt_allowed(r["salt_cap"], taxpayer.state_local_taxes_paid, agi, taxpayer.filing_status)
    itemized = taxpayer.itemized_deductions + salt
    use_itemized = itemized > standard
    deduction_amount = itemized if use_itemized else standard
    deduction_used = "itemized" if use_itemized else "standard"

    taxable_before_qbi = max(0.0, agi - deduction_amount)
    qbi, qbi_note = qbi_deduction(r["qbi_deduction"], taxpayer, taxable_before_qbi)
    taxable_income = max(0.0, taxable_before_qbi - qbi)

    result = USAssessment(
        year=pack.year,
        agi=agi,
        taxable_income=taxable_income,
        deduction_used=deduction_used,
        deduction_amount=deduction_amount,
        verified=pack.is_verified,
    )
    if not pack.is_verified:
        result.warnings.append(
            "Computed from an unverified rule pack. The 2025 figures reflect the One Big Beautiful Bill Act, "
            "which changed the standard deduction, SALT cap, section 179 limit and bonus depreciation mid-year. "
            "Confirm every figure against the enacted statute and controlling Revenue Procedure before filing."
        )
    result.warnings.append(
        "FEDERAL ONLY. State and local income tax is not modelled. In a high-tax state this can add "
        "a substantial further liability that is not shown anywhere in this figure."
    )

    ordinary_for_rates = max(0.0, taxable_income - taxpayer.long_term_capital_gain)
    brackets = r["ordinary_income_brackets"][taxpayer.filing_status]
    income_tax = tax_on_income(brackets, ordinary_for_rates)
    result.items.append(LineItem("Federal income tax on ordinary income", income_tax, r["ordinary_income_brackets"].get("cite", "")))

    if taxpayer.long_term_capital_gain > 0:
        cg_cfg = r["capital_gains"]
        cg_brackets = cg_cfg["long_term_brackets_mfj"] if taxpayer.filing_status == "married_filing_jointly" else cg_cfg["long_term_brackets_single"]
        stacked = tax_on_income(cg_brackets, taxable_income) - tax_on_income(cg_brackets, ordinary_for_rates)
        result.items.append(
            LineItem("Tax on long term capital gains", round(max(0.0, stacked), 2), cg_cfg.get("cite", ""),
                     "Long term gains stack on top of ordinary income and are taxed at preferential rates.")
        )

    if se_tax:
        result.items.append(LineItem("Self-employment tax", se_tax, r["self_employment_tax"].get("cite", ""), se_note))

    niit_cfg = r["net_investment_income_tax"]
    niit_key = {
        "married_filing_jointly": "threshold_mfj",
        "married_filing_separately": "threshold_mfs",
        "head_of_household": "threshold_hoh",
    }.get(taxpayer.filing_status, "threshold_single")
    net_investment = taxpayer.investment_income + taxpayer.long_term_capital_gain + taxpayer.short_term_capital_gain
    niit_base = min(net_investment, max(0.0, agi - float(niit_cfg[niit_key])))
    if niit_base > 0:
        result.items.append(
            LineItem("Net investment income tax", round(niit_base * float(niit_cfg["rate"]), 2), niit_cfg.get("cite", ""),
                     f"3.8% on the lesser of net investment income and the excess of MAGI over ${float(niit_cfg[niit_key]):,.0f}.")
        )

    credits = taxpayer.federal_withheld + taxpayer.estimated_payments
    tax_before_credits = sum(i.amount for i in result.items)
    if credits:
        result.items.append(LineItem("Withholding and estimated payments", -credits, "IRC 31, 6315"))

    result.safe_harbor = estimated_tax_safe_harbor(r["estimated_tax"], taxpayer, tax_before_credits)
    result.adjustments.append(
        LineItem(
            "Itemized deductions" if use_itemized else "Standard deduction",
            deduction_amount,
            std_cfg.get("cite", ""),
            salt_note if use_itemized else "Itemizing would not beat the standard deduction this year.",
        )
    )
    if se_half:
        result.adjustments.append(
            LineItem("Deductible half of self-employment tax", se_half,
                     r["self_employment_tax"].get("cite", ""), "Subtracted above the line in arriving at AGI.")
        )
    result.adjustments.append(
        LineItem("Qualified business income deduction", qbi, r["qbi_deduction"].get("cite", ""), qbi_note)
    )
    return result
