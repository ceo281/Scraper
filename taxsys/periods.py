"""Tax period and obligation calendars.

The two jurisdictions do not agree on when a year is, which is the first thing
that breaks a naive dual-country design:

  Australia  1 July to 30 June, with BAS quarters offset from calendar quarters.
  US         1 January to 31 December, with estimated tax "quarters" that are
             not quarters at all. Q2 covers two months and Q3 covers three.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

from .rulepack import RulePack

Jurisdiction = Literal["AU", "US"]


@dataclass(frozen=True)
class Obligation:
    """A dated filing or payment obligation."""

    code: str
    label: str
    period_start: date
    period_end: date
    due: date
    cite: str
    note: str = ""

    def days_until(self, today: date | None = None) -> int:
        return (self.due - (today or date.today())).days

    def status(self, today: date | None = None) -> str:
        days = self.days_until(today)
        if days < 0:
            return "overdue"
        if days <= 14:
            return "due_soon"
        return "upcoming"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "due": self.due.isoformat(),
            "cite": self.cite,
            "note": self.note,
            "days_until": self.days_until(),
            "status": self.status(),
        }


@dataclass(frozen=True)
class TaxPeriod:
    jurisdiction: str
    label: str
    start: date
    end: date

    def contains(self, when: date) -> bool:
        return self.start <= when <= self.end


# --------------------------------------------------------------------- years


def au_financial_year(label: str) -> TaxPeriod:
    """'2025-26' becomes 1 Jul 2025 to 30 Jun 2026."""
    try:
        start_year = int(label.split("-")[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Malformed AU financial year {label!r}, expected '2025-26'") from exc
    return TaxPeriod("AU", label, date(start_year, 7, 1), date(start_year + 1, 6, 30))


def us_tax_year(label: str) -> TaxPeriod:
    year = int(label)
    return TaxPeriod("US", label, date(year, 1, 1), date(year, 12, 31))


def period_for(jurisdiction: str, year_label: str) -> TaxPeriod:
    return au_financial_year(year_label) if jurisdiction.upper() == "AU" else us_tax_year(year_label)


def year_label_for(jurisdiction: str, when: date) -> str:
    """Which tax year does a given date fall in?"""
    if jurisdiction.upper() == "AU":
        start = when.year if when.month >= 7 else when.year - 1
        return f"{start}-{str(start + 1)[2:]}"
    return str(when.year)


# --------------------------------------------------------------- obligations


def _next_business_day(day: date) -> date:
    """Weekend rollover. Public holidays are jurisdiction and state specific and
    are deliberately not modelled here rather than modelled wrongly."""
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def au_obligations(pack: RulePack, *, via_agent: bool = False, gst_registered: bool = True) -> list[Obligation]:
    period = au_financial_year(pack.year)
    out: list[Obligation] = []

    if gst_registered and pack.indirect:
        cite = pack.indirect.get("cite", "TAA 1953 Sch 1")
        due_key = "due_via_agent" if via_agent else "due_self"
        for quarter in pack.indirect.get("bas_quarters", []):
            s_month, s_day = (int(x) for x in quarter["starts"].split("-"))
            e_month, e_day = (int(x) for x in quarter["ends"].split("-"))
            d_month, d_day = (int(x) for x in quarter[due_key].split("-"))
            # Jul to Dec sit in the first calendar year of the FY, Jan to Jun in the second.
            s_year = period.start.year if s_month >= 7 else period.end.year
            e_year = period.start.year if e_month >= 7 else period.end.year
            # A BAS is always due after its quarter closes. Rolling the year
            # forward when the due month precedes the quarter end is what keeps
            # Q2 (Oct-Dec, due Feb) and Q4 (Apr-Jun, due Jul) in the right years.
            d_year = e_year if (d_month, d_day) >= (e_month, e_day) else e_year + 1
            out.append(
                Obligation(
                    code=f"AU-BAS-Q{quarter['q']}",
                    label=f"Business Activity Statement, Q{quarter['q']} ({quarter['period']})",
                    period_start=date(s_year, s_month, s_day),
                    period_end=date(e_year, e_month, e_day),
                    due=_next_business_day(date(d_year, d_month, d_day)),
                    cite=cite,
                    note="Agent concession date applies only if you are on a registered agent's lodgment program."
                    if via_agent
                    else "",
                )
            )

    out.append(
        Obligation(
            code="AU-ITR",
            label=f"Individual income tax return, FY{pack.year}",
            period_start=period.start,
            period_end=period.end,
            due=_next_business_day(date(period.end.year, 10, 31)),
            cite="TAA 1953 s 161",
            note="31 October if self-lodging. Registered agents have a concessional program running to 15 May, "
                 "but you must be on the agent's client list before 31 October to use it.",
        )
    )
    return sorted(out, key=lambda o: o.due)


def us_obligations(pack: RulePack, *, pays_estimated: bool = True) -> list[Obligation]:
    period = us_tax_year(pack.year)
    year = period.start.year
    out: list[Obligation] = []
    est = pack.rates.get("estimated_tax", {})

    if pays_estimated:
        spans = {
            "Q1": (date(year, 1, 1), date(year, 3, 31)),
            "Q2": (date(year, 4, 1), date(year, 5, 31)),
            "Q3": (date(year, 6, 1), date(year, 8, 31)),
            "Q4": (date(year, 9, 1), date(year, 12, 31)),
        }
        for entry in est.get("due_dates", []):
            month, day = (int(x) for x in entry["due"].split("-"))
            due_year = year + int(entry.get("due_year_offset", 0))
            start, end = spans[entry["period"]]
            out.append(
                Obligation(
                    code=f"US-1040ES-{entry['period']}",
                    label=f"Estimated tax payment, {entry['period']} ({entry['covers']})",
                    period_start=start,
                    period_end=end,
                    due=_next_business_day(date(due_year, month, day)),
                    cite=est.get("cite", "IRC 6654"),
                    note="These are not equal calendar quarters. Q2 covers two months and Q3 covers three.",
                )
            )

    deadlines = pack.rates.get("filing_deadlines", {})
    month, day = (int(x) for x in deadlines.get("individual_return", "04-15").split("-"))
    out.append(
        Obligation(
            code="US-1040",
            label=f"Form 1040, tax year {pack.year}",
            period_start=period.start,
            period_end=period.end,
            due=_next_business_day(date(year + 1, month, day)),
            cite="IRC 6072",
            note=deadlines.get("note", ""),
        )
    )
    return sorted(out, key=lambda o: o.due)


def obligations_for(pack: RulePack, **kwargs: Any) -> list[Obligation]:
    if pack.jurisdiction == "AU":
        return au_obligations(pack, **{k: v for k, v in kwargs.items() if k in {"via_agent", "gst_registered"}})
    return us_obligations(pack, **{k: v for k, v in kwargs.items() if k in {"pays_estimated"}})
