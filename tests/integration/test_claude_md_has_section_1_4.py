"""Tests that CLAUDE.md exposes the §1.4 Human ↔ AI Responsibility Contract.

The contract makes the architect/implementer boundary explicit: a heading,
a 3-column table with at least 10 rows, and an 'Operating rule' paragraph.
"""

import re
from pathlib import Path

CLAUDE_MD = Path(__file__).resolve().parents[2] / "CLAUDE.md"


def _read_claude_md() -> str:
    return CLAUDE_MD.read_text(encoding="utf-8")


def test_section_1_4_header_present() -> None:
    """§1.4 architect/implementer contract must be visible as a section heading."""
    text = _read_claude_md()
    assert re.search(r"§1\.4|Human ↔ AI Responsibility Contract", text), (
        "CLAUDE.md missing §1.4 / Human ↔ AI Responsibility Contract heading"
    )


def test_contract_table_present() -> None:
    """The contract table must list Concern / Human-decided / AI-delegated columns."""
    text = _read_claude_md()
    m = re.search(r"\|\s*Concern\s*\|\s*Human-decided.*?\|\s*AI-delegated.*?\|", text)
    assert m, "Contract table header (Concern | Human-decided | AI-delegated) not found"


def test_contract_table_has_at_least_10_rows() -> None:
    """The contract table must enumerate enough responsibility rows to be useful."""
    text = _read_claude_md()
    header_idx = text.find("| Concern")
    assert header_idx > 0
    body = text[header_idx:]
    rows = re.findall(r"^\|[^\n]+\|$", body, flags=re.MULTILINE)
    assert len(rows) >= 12, (
        f"Contract table has only {len(rows)} pipe-delimited lines; want ≥ 12 (header+sep+10 body rows)"
    )


def test_operating_rule_paragraph_after_table() -> None:
    """The 'Operating rule' explanation must follow the table to make the boundary executable."""
    text = _read_claude_md()
    assert "Operating rule" in text, "CLAUDE.md must include the 'Operating rule' paragraph"
