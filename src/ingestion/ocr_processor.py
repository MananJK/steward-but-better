"""OCR processor — converts F1 rule PDFs to Markdown via Mistral OCR.

Scans each rule directory for PDFs, runs OCR, and writes .md files
while keeping the original folder structure intact.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import base64

from dotenv import load_dotenv
from mistralai import Mistral

# Pull env vars from the project-root .env
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


RULE_DIRECTORIES = (
    Path("rules/sporting_regulations"),
    Path("rules/driving_standards"),
    Path("rules/steward_standards"),
)
DEFAULT_OCR_MODEL = "mistral-ocr-latest"
MANIFEST_FILENAME = ".ocr_manifest.json"
FAILED_LOG_FILENAME = "failed_docs.log"


def _sha256_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Quick SHA-256 hash of a file for change detection."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    """Load the processing manifest so we can skip unchanged files."""
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_manifest(manifest_path: Path, data: Dict[str, Dict[str, str]]) -> None:
    """Write the manifest back to disk."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, sort_keys=True)


def _iter_pdf_files(input_root: Path) -> Iterable[Path]:
    """Walk the rule directories and yield every PDF found."""
    for relative_dir in RULE_DIRECTORIES:
        source_dir = input_root / relative_dir
        if not source_dir.exists():
            continue
        yield from source_dir.rglob("*.pdf")


def _extract_markdown_from_response(response: object) -> str:
    """Pull markdown text out of whichever response shape the SDK gives us."""
    if response is None:
        return ""

    # Simple case: top-level markdown string
    markdown_direct = getattr(response, "markdown", None)
    if isinstance(markdown_direct, str) and markdown_direct.strip():
        return markdown_direct

    # Multi-page response — stitch pages together
    pages_obj = getattr(response, "pages", None)
    if pages_obj:
        page_blocks = []
        for page in pages_obj:
            value = getattr(page, "markdown", None)
            if isinstance(value, str) and value.strip():
                page_blocks.append(value)
        if page_blocks:
            return "\n\n".join(page_blocks)

    # Fallback for dict-style responses (older SDK versions)
    if isinstance(response, dict):
        pages = response.get("pages", [])
        return "\n\n".join([p.get("markdown", "") for p in pages if isinstance(p, dict)])

    return ""


def _call_ocr_markdown(client: Mistral, pdf_path: Path, model: str) -> str:
    """Base64-encode a PDF and send it through Mistral OCR, return markdown."""
    with open(pdf_path, "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

    ocr_response = client.ocr.process(
        model=model,
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_base64}",
        },
    )
    
    return "\n\n".join(page.markdown for page in ocr_response.pages)


def process_f1_docs(
    input_root: str | Path,
    output_root: str | Path,
    model: str = DEFAULT_OCR_MODEL,
) -> None:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment. Check your .env file.")
    client = Mistral(api_key=api_key)

    input_root_path = Path(input_root).resolve()
    output_root_path = Path(output_root).resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root_path / MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    pdf_files = sorted(_iter_pdf_files(input_root_path))
    logger.info("Discovered %d PDF file(s). Starting Ingestion...", len(pdf_files))

    for pdf_path in pdf_files:
        relative_pdf = pdf_path.relative_to(input_root_path)
        output_md_path = (output_root_path / relative_pdf).with_suffix(".md")
        output_key = relative_pdf.as_posix()

        try:
            checksum = _sha256_file(pdf_path)
            if manifest.get(output_key, {}).get("checksum") == checksum and output_md_path.exists():
                logger.info("Skipping (unchanged): %s", relative_pdf)
                continue

            logger.info("Uploading and Processing: %s", relative_pdf)
            markdown = _call_ocr_markdown(client=client, pdf_path=pdf_path, model=model)
            
            output_md_path.parent.mkdir(parents=True, exist_ok=True)
            output_md_path.write_text(markdown, encoding="utf-8")

            manifest[output_key] = {"checksum": checksum}
            _save_manifest(manifest_path, manifest)
            logger.info("Successfully ingested: %s", relative_pdf)

        except Exception as exc:
            logger.error("Failed to process %s: %s", relative_pdf, exc)


if __name__ == "__main__":
    process_f1_docs(input_root=".", output_root="processed_rules")