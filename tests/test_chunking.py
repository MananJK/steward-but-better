"""Tests for article-aware chunking and metadata enrichment in vector_index."""

from vector_index import (
    _chunk_text,
    _derive_category,
    _enrich_metadata,
    _extract_article,
    _split_sections,
)


def test_sections_split_at_markdown_headings():
    text = "Intro paragraph.\n\n# First Heading\nContent one.\n\n## Second Heading\nContent two."
    sections = _split_sections(text)
    assert len(sections) == 3
    assert sections[0].startswith("Intro")
    assert sections[1].startswith("# First Heading")
    assert sections[2].startswith("## Second Heading")


def test_sections_split_at_article_headings():
    text = "Preamble.\n33.3 Track limits text.\nArticle 54.3 More text."
    sections = _split_sections(text)
    assert len(sections) == 2
    assert sections[1].startswith("Article 54.3")


def test_chunking_respects_size_limit():
    text = "# Heading\n" + ("word " * 400)
    chunks = _chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 600 for c in chunks)


def test_chunking_preserves_short_sections_intact():
    text = "# A\nshort section\n\n# B\n" + ("filler " * 100)
    chunks = _chunk_text(text, chunk_size=400, chunk_overlap=40)
    joined = "\n\n".join(chunks)
    assert "short section" in joined


def test_empty_text_yields_no_chunks():
    assert _chunk_text("") == []
    assert _chunk_text("   \n  ") == []


def test_extract_article_isc_style():
    assert _extract_article("As per Article 33.3 of the regulations") == "Article 33.3"
    assert _extract_article("see Art. 12 herein") == "Article 12"


def test_extract_article_fia_section_heading():
    assert (
        _extract_article("## 54) INCIDENTS DURING THE RACE\n54.1 If a driver is reported...")
        == "Clause 54.1"
    )
    assert _extract_article("## 9) SAFETY CAR\nSome prose.") == "Section 9"


def test_extract_article_none_for_plain_text():
    assert _extract_article("Cars must not be released unsafely.") is None


def test_derive_category_detects_misfiled_technical_doc():
    assert (
        _derive_category("rules/driving_standards/fia_2025_technical_regulations.md")
        == "Technical Regulations"
    )
    assert _derive_category("rules/sporting_regulations/x.md") == "Sporting Regulations"
    assert _derive_category("rules/driving_standards/guidelines.md") == "Driving Standards"


def test_enrich_metadata_infers_year_from_source():
    meta = {"Year": "unknown", "Document Category": "Unknown", "source": "rules/sporting_regulations/fia_2024_issue7.md"}
    enriched = _enrich_metadata(meta, "Any text")
    assert enriched["Year"] == "2024"
    assert enriched["Document Category"] == "Sporting Regulations"
