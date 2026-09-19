"""The guarantees this system claims, asserted as tests.

If any of these fail, the product's core claim is false and it must not ship.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from taxsys.answer import AnswerEngine, Verdict
from taxsys.corpus import Anchor, Chunk, Corpus
from taxsys.ingest.parse import parse_au_act_html, parse_ecfr_xml, parse_usc_xml
from taxsys.retrieve import Retriever
from taxsys.rulepack import RulePack, UnverifiedRulePackError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def us_corpus() -> Corpus:
    corpus = Corpus()
    for section in parse_usc_xml((FIXTURES / "usc26_sample.xml").read_bytes()):
        corpus.add(Chunk(id=section.section_id, jurisdiction="US", kind="statute",
                         citation=section.citation, heading=section.heading, text=section.text,
                         span_hash=section.span_hash, effective_from="2018-01-01"))
    for section in parse_ecfr_xml((FIXTURES / "ecfr26_sample.xml").read_bytes()):
        corpus.add(Chunk(id=section.section_id, jurisdiction="US", kind="regulation",
                         citation=section.citation, heading=section.heading, text=section.text,
                         span_hash=section.span_hash, effective_from="2020-10-09"))
    return corpus


@pytest.fixture
def engine(us_corpus: Corpus) -> AnswerEngine:
    return AnswerEngine(RulePack.load("US", "2025"), us_corpus, Retriever(us_corpus))


# ------------------------------------------------- the anti-hallucination gates


def test_fabricated_figure_is_removed(engine: AnswerEngine) -> None:
    """A number with no rule pack provenance never reaches the user."""
    def writer(question, hits, payload):
        return "Business meals are 80 percent deductible for 2025."

    answer = engine.ask("client dinner food and beverages", prose_writer=writer,
                        facts={"business_use_percent": 100, "method": "actual"})
    assert "80 percent" not in answer.summary
    assert any("unprovenanced" in claim for claim in answer.dropped_claims)


def test_fabricated_citation_is_removed(engine: AnswerEngine) -> None:
    """A citation the retriever did not return is a fabrication and is stripped."""
    def writer(question, hits, payload):
        return "This is settled by [[usc-26-9999]]."

    answer = engine.ask("client dinner food and beverages", prose_writer=writer,
                        facts={"business_use_percent": 100, "method": "actual"})
    assert "usc-26-9999" not in answer.summary
    assert any("unresolvable citation" in claim for claim in answer.dropped_claims)


def test_every_surviving_citation_resolves(engine: AnswerEngine) -> None:
    import re

    answer = engine.ask("client dinner food and beverages",
                        facts={"business_use_percent": 100, "method": "actual"})
    for chunk_id in re.findall(r"\[\[([a-z0-9\-\.]+)\]\]", answer.summary):
        assert engine.corpus.get(chunk_id) is not None


def test_every_figure_carries_a_rule_path(engine: AnswerEngine) -> None:
    """No orphan numbers. Each traces to the rule pack record it came from."""
    answer = engine.ask("client dinner food and beverages",
                        facts={"business_use_percent": 100, "method": "actual"})
    for figure in answer.figures:
        assert figure.rule_path.startswith("deductions.")
        assert figure.citation


def test_unknown_question_refuses_rather_than_reasoning(engine: AnswerEngine) -> None:
    answer = engine.ask("Can I deduct my pet iguana as a security system?")
    assert answer.verdict is Verdict.NO_AUTHORITY
    assert answer.confidence == 0.0


# ------------------------------------------------------- freshness and anchors


def test_amended_source_suspends_the_rule() -> None:
    """The law moving must invalidate the rule automatically."""
    corpus = Corpus()
    sections = list(parse_au_act_html((FIXTURES / "itaa1997_sample.html").read_bytes(),
                                      act_citation="ITAA 1997", act_key="ITAA1997"))
    for section in sections:
        corpus.add(Chunk(id=section.section_id, jurisdiction="AU", kind="statute",
                         citation=section.citation, heading=section.heading, text=section.text,
                         span_hash=section.span_hash, effective_from="2024-07-01"))

    original = corpus.get("au-itaa1997-s8-1")
    corpus.bind(Anchor(rule_id="AU-GENERAL-DEDUCTION", chunk_id=original.id,
                       citation=original.citation, span_hash=original.span_hash,
                       bound_at="2026-09-19"))
    assert corpus.is_servable("AU-GENERAL-DEDUCTION")[0] is True

    amended = Chunk(id=original.id, jurisdiction="AU", kind="statute", citation=original.citation,
                    heading=original.heading, text=original.text + " (3) A further limitation applies.",
                    effective_from="2026-07-01")
    corpus._by_id[original.id] = amended

    servable, reason = corpus.is_servable("AU-GENERAL-DEDUCTION")
    assert servable is False
    assert "has changed" in reason
    assert corpus.reconcile()["suspended"] == ["AU-GENERAL-DEDUCTION"]


def test_unanchored_rule_cannot_be_served() -> None:
    corpus = Corpus()
    servable, reason = corpus.is_servable("AU-SOMETHING")
    assert servable is False
    assert "not bound to any source text" in reason


def test_out_of_force_law_is_filtered_not_ranked(us_corpus: Corpus) -> None:
    """A repealed provision must never be returned, however well it matches."""
    repealed = Chunk(id="usc-26-999", jurisdiction="US", kind="statute",
                     citation="26 U.S.C. 999", heading="Repealed meals provision",
                     text="ordinary necessary trade business food beverages entertainment meals deduction",
                     effective_from="1990-01-01", effective_to="2017-12-31")
    us_corpus.add(repealed)
    hits = Retriever(us_corpus).search("meals food beverages", jurisdiction="US", year_label="2025")
    assert "usc-26-999" not in {hit.chunk.id for hit in hits}


# ------------------------------------------------------- the verification gate


def test_unverified_pack_cannot_be_presented_as_final() -> None:
    pack = RulePack.load("AU", "2025-26")
    with pytest.raises(UnverifiedRulePackError):
        pack.assert_usable()


def test_missing_year_refuses_to_substitute_another() -> None:
    from taxsys.rulepack import RulePackError

    with pytest.raises(RulePackError, match="No rates on record"):
        RulePack.load("US", "2019")


# ------------------------------------------------------- computation guarantees


def test_mixed_use_line_is_withheld_not_overstated() -> None:
    """The failure mode that silently inflates a claim."""
    from taxsys.classify import Classifier, ExpenseLine

    classifier = Classifier(RulePack.load("US", "2025"))
    result = classifier.classify(ExpenseLine(merchant="Shell", description="fuel", amount=58.20, date="2026-09-09"))
    assert result.claimable_amount is None
    assert result.ceiling_amount == 58.20
    assert result.status == "incomplete"


def test_bas_due_dates_fall_after_their_quarter() -> None:
    """The year rollover bug: a BAS is always due after its quarter closes."""
    from taxsys.periods import obligations_for

    for obligation in obligations_for(RulePack.load("AU", "2025-26")):
        assert obligation.due > obligation.period_end, obligation.code


def test_au_brackets_match_published_figures() -> None:
    from taxsys.compute_au import tax_on_income

    brackets = RulePack.load("AU", "2025-26").rates["income_tax_brackets"]["brackets"]
    for income, expected in [(18200, 0.0), (45000, 4288.0), (135000, 31288.0), (190000, 51638.0)]:
        assert tax_on_income(brackets, income) == expected


def test_us_brackets_match_published_figures() -> None:
    from taxsys.compute_us import tax_on_income

    brackets = RulePack.load("US", "2025").rates["ordinary_income_brackets"]["single"]
    assert tax_on_income(brackets, 11925) == 1192.50
    assert tax_on_income(brackets, 100000) == 16914.00
