"""Parsers turning raw primary sources into effective-dated, hashed chunks.

Section granularity is deliberate. A chunk is one statutory section, one
regulation section, or one ruling paragraph group, because that is the unit a
citation points at. Chunking by fixed token window would produce spans that no
citation can address, which breaks the provenance guarantee the whole system
rests on.
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterator

from .sources import SourceDocument

USLM_NS = {"uslm": "http://schemas.gpo.gov/xml/uslm"}


@dataclass
class ParsedSection:
    """One addressable span of primary source text."""

    section_id: str
    citation: str
    heading: str
    text: str
    ordinal: int = 0

    @property
    def span_hash(self) -> str:
        """Hash of this exact span. A rule record stores this; when it changes,
        the rule is invalidated."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ------------------------------------------------------------------ US Code


def parse_usc_xml(xml_bytes: bytes, *, title: str = "26") -> Iterator[ParsedSection]:
    """Parse USLM XML from the Office of the Law Revision Counsel.

    Sections carry <num value="162"> and <heading>, with the operative text in
    nested <subsection>, <paragraph> and <content> elements.
    """
    root = ET.fromstring(xml_bytes)
    ordinal = 0
    for section in root.iter():
        if not section.tag.endswith("}section") and section.tag != "section":
            continue
        num_el = _find_child(section, "num")
        heading_el = _find_child(section, "heading")
        if num_el is None:
            continue
        number = (num_el.get("value") or _clean(num_el.text or "")).lstrip("§").strip().rstrip(".")
        if not number:
            continue
        body = _clean(" ".join(t for t in section.itertext()))
        heading = _clean(heading_el.text if heading_el is not None else "")
        if heading:
            body = body.replace(heading, "", 1).strip()
        ordinal += 1
        yield ParsedSection(
            section_id=f"usc-{title}-{number}",
            citation=f"{title} U.S.C. {number}",
            heading=heading,
            text=body,
            ordinal=ordinal,
        )


def _find_child(parent: ET.Element, localname: str) -> ET.Element | None:
    for child in parent:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == localname:
            return child
    return None


# ---------------------------------------------------------------- eCFR XML


def parse_ecfr_xml(xml_bytes: bytes, *, title: str = "26") -> Iterator[ParsedSection]:
    """Parse eCFR XML. Sections are DIV8 elements with TYPE="SECTION"."""
    root = ET.fromstring(xml_bytes)
    ordinal = 0
    for div in root.iter("DIV8"):
        if div.get("TYPE") != "SECTION":
            continue
        number = (div.get("N") or "").strip()
        if not number:
            continue
        head_el = div.find("HEAD")
        heading = _clean(head_el.text if head_el is not None else "")
        paragraphs = [_clean(" ".join(p.itertext())) for p in div.iter("P")]
        body = " ".join(p for p in paragraphs if p)
        ordinal += 1
        yield ParsedSection(
            section_id=f"cfr-{title}-{number}",
            citation=f"{title} CFR {number}",
            heading=heading.lstrip("§ ").strip(),
            text=body,
            ordinal=ordinal,
        )


# ------------------------------------------------- AU legislation and rulings


class _TextExtractor(HTMLParser):
    """Strip markup, keep section boundaries."""

    SKIP = {"script", "style", "nav", "header", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data)

    @property
    def text(self) -> str:
        return re.sub(r"\n{2,}", "\n", "".join(self.parts))


# "8-1  General deductions" or "s 8-1" style headings in the AU Acts.
_AU_SECTION_RE = re.compile(
    r"^\s*(?:Section\s+)?(\d+[A-Z]*(?:-\d+[A-Z]*)?)\s{1,6}([A-Z][^\n]{3,120})$",
    re.MULTILINE,
)


def parse_au_act_html(html_bytes: bytes, *, act_citation: str, act_key: str) -> Iterator[ParsedSection]:
    """Split a consolidated AU Act into sections.

    AU section numbering is hyphenated in the 1997 Act (8-1, 25-100, 900-115)
    and plain in the 1936 Act (177A, 262A), so both shapes are matched.
    """
    extractor = _TextExtractor()
    extractor.feed(html_bytes.decode("utf-8", errors="replace"))
    text = extractor.text

    matches = list(_AU_SECTION_RE.finditer(text))
    if not matches:
        yield ParsedSection(
            section_id=f"au-{act_key.lower()}-full",
            citation=act_citation,
            heading=act_citation,
            text=_clean(text),
        )
        return

    for ordinal, match in enumerate(matches, start=1):
        number, heading = match.group(1), _clean(match.group(2))
        start = match.end()
        end = matches[ordinal].start() if ordinal < len(matches) else len(text)
        body = _clean(text[start:end])
        if len(body) < 40:
            continue
        yield ParsedSection(
            section_id=f"au-{act_key.lower()}-s{number}",
            citation=f"{act_citation} s {number}",
            heading=heading,
            text=body,
            ordinal=ordinal,
        )


_ATO_PARA_RE = re.compile(r"^\s*(\d{1,3})\.\s+(.{40,})$", re.MULTILINE)


def parse_ato_ruling_html(html_bytes: bytes, *, citation: str, doc_id: str) -> Iterator[ParsedSection]:
    """Split an ATO ruling into numbered paragraphs.

    ATO rulings are cited by paragraph, so paragraph is the addressable unit.
    """
    extractor = _TextExtractor()
    extractor.feed(html_bytes.decode("utf-8", errors="replace"))
    text = extractor.text
    matches = list(_ATO_PARA_RE.finditer(text))
    if not matches:
        yield ParsedSection(section_id=f"{doc_id}-full", citation=citation, heading=citation, text=_clean(text))
        return
    for ordinal, match in enumerate(matches, start=1):
        para_no, body = match.group(1), _clean(match.group(2))
        yield ParsedSection(
            section_id=f"{doc_id}-p{para_no}",
            citation=f"{citation} para {para_no}",
            heading=f"{citation} paragraph {para_no}",
            text=body,
            ordinal=ordinal,
        )


PARSERS = {
    ("US", "statute"): parse_usc_xml,
    ("US", "regulation"): parse_ecfr_xml,
    ("AU", "statute"): parse_au_act_html,
    ("AU", "ruling"): parse_ato_ruling_html,
    ("AU", "determination"): parse_ato_ruling_html,
    ("AU", "guideline"): parse_ato_ruling_html,
}


def parse(document: SourceDocument) -> list[ParsedSection]:
    """Dispatch a fetched document to its parser."""
    key = (document.jurisdiction, document.kind)
    parser = PARSERS.get(key)
    if parser is None:
        raise ValueError(f"No parser registered for {key}")
    if key == ("AU", "statute"):
        return list(parser(document.raw, act_citation=document.citation, act_key=document.metadata.get("act_key", "act")))
    if document.jurisdiction == "AU":
        return list(parser(document.raw, citation=document.citation, doc_id=document.source_id))
    return list(parser(document.raw))
