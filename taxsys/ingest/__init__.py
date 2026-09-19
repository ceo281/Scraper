"""Ingestion of primary tax law sources.

sources.py fetches from the authoritative bulk feeds. parse.py splits each
document into citation-addressable, hashed spans. Nothing here interprets law:
it only moves text from a register into the corpus with its provenance intact.
"""
from .sources import (
    ATOLegalSource,
    ECFRSource,
    FRLSource,
    IRBSource,
    SourceDocument,
    SourceUnavailable,
    USCodeSource,
    default_manifest,
)
from .parse import ParsedSection, parse, parse_ato_ruling_html, parse_au_act_html, parse_ecfr_xml, parse_usc_xml

__all__ = [
    "ATOLegalSource", "ECFRSource", "FRLSource", "IRBSource", "SourceDocument",
    "SourceUnavailable", "USCodeSource", "default_manifest",
    "ParsedSection", "parse", "parse_ato_ruling_html", "parse_au_act_html",
    "parse_ecfr_xml", "parse_usc_xml",
]
