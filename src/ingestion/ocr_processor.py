"""OCR processor for F1 rule documents using Mistral OCR.

This module scans the required rule directories for PDF files, converts them to
Markdown with Mistral OCR, and stores the output while preserving the input
folder structure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv
load_dotenv(override=True)
from mistralai import Mistral

RULE_DIRECTORIES = (
    Path("rules/sporting_regulations"),
    Path("rules/driving_standards"),
    Path("rules/steward_standards"),
)
DEFAULT_OCR_MODEL = "mistral-ocr-2512"
MANIFEST_FILENAME = ".ocr_manifest.json"
FAILED_LOG_FILENAME = "failed_docs.log"


def _sha256_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 checksum for a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    """Load manifest data if available, otherwise return an empty dictionary."""
    if not manifest_path.exists():
        return {}

    try:
        with manifest_path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except (json.JSONDecodeError, OSError):
        return {}

    return data if isinstance(data, dict) else {}


def _save_manifest(manifest_path: Path, data: Dict[str, Dict[str, str]]) -> None:
    """Persist manifest data to disk."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, sort_keys=True)


def _iter_pdf_files(input_root: Path) -> Iterable[Path]:
    """Yield all PDFs from the required rule directories."""
    for relative_dir in RULE_DIRECTORIES:
        source_dir = input_root / relative_dir
        if not source_dir.exists():
            continue
        yield from source_dir.rglob("*.pdf")


def _extract_markdown_from_response(response: object) -> str:
    """Extract markdown text from multiple possible SDK response shapes."""
    if response is None:
        return ""

    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        markdown = response.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            return markdown

        pages = response.get("pages")
        if isinstance(pages, list):
            page_blocks: List[str] = []
            for page in pages:
                if isinstance(page, dict):
                    value = page.get("markdown")
                    if isinstance(value, str) and value.strip():
                        page_blocks.append(value)
            if page_blocks:
                return "\n\n".join(page_blocks)

    markdown_direct = getattr(response, "markdown", None)
    if isinstance(markdown_direct, str) and markdown_direct.strip():
        return markdown_direct

    pages_obj = getattr(response, "pages", None)
    if pages_obj:
        page_blocks = []
        for page in pages_obj:
            value = getattr(page, "markdown", None)
            if isinstance(value, str) and value.strip():
                page_blocks.append(value)
        if page_blocks:
            return "\n\n".join(page_blocks)

    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return _extract_markdown_from_response(dumped)

    return ""


# def _call_ocr_markdown(client: Mistral, pdf_path: Path, model: str) -> str:
#     """Run OCR for a local PDF and return markdown text."""
#     with pdf_path.open("rb") as file_handle:
#         uploaded_file = client.files.upload(
#             file={
#                 "file_name": pdf_path.name,
#                 "content": file_handle,
#             },
#             purpose="ocr",
#         )

#     file_id = getattr(uploaded_file, "id", None)
#     if not file_id and isinstance(uploaded_file, dict):
#         file_id = uploaded_file.get("id")

#     if not file_id:
#         raise RuntimeError(f"Unable to resolve uploaded file id for {pdf_path}")

#     signed_url_response = client.files.get_signed_url(file_id=file_id)
#     document_url = getattr(signed_url_response, "url", None)
#     if not document_url and isinstance(signed_url_response, dict):
#         document_url = signed_url_response.get("url")

#     if not document_url:
#         raise RuntimeError(f"Unable to obtain signed URL for {pdf_path}")

#     ocr_kwargs = {
#         "model": model,
#         "document": {
#             "type": "document_url",
#             "document_url": document_url,
#         },
#         "output_format": "markdown",
#     }

#     try:
#         response = client.ocr.process(**ocr_kwargs)
#     except TypeError:
#         ocr_kwargs.pop("output_format", None)
#         response = client.ocr.process(**ocr_kwargs)

#     markdown = _extract_markdown_from_response(response).strip()
#     if not markdown:
#         raise RuntimeError(f"OCR returned empty markdown for {pdf_path}")

#     return markdown


def _call_ocr_markdown(client: Mistral, pdf_path: Path, model: str) -> str:
    """Run OCR for a local PDF with enhanced error handling for F1 docs."""
    # Use a direct upload-and-process flow to avoid signed URL timeouts
    with pdf_path.open("rb") as f:
        uploaded_file = client.files.upload(
            file={"file_name": pdf_path.name, "content": f},
            purpose="ocr"
        )
    
    # Mistral OCR 3 requires a slight delay for larger FIA PDFs
    import time
    time.sleep(1) 

    ocr_response = client.ocr.process(
        model=model,
        document={"type": "file_id", "file_id": uploaded_file.id}
    )
    
    return _extract_markdown_from_response(ocr_response)

def process_f1_docs(
    input_root: str | Path,
    output_root: str | Path,
    model: str = DEFAULT_OCR_MODEL,
) -> None:
    # FOOLPROOF INITIALIZATION: No .env, no shadowing.
    # Replace the string below with your FULL key from the Hacker Space dashboard.
    api_key = "F0C157B7-4661-427A-A134-54CEFA11B5A8" 
    client = Mistral(api_key=api_key)

    input_root_path = Path(input_root).resolve()
    output_root_path = Path(output_root).resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    failed_log_path = output_root_path / FAILED_LOG_FILENAME
    manifest_path = output_root_path / MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    pdf_files = sorted(_iter_pdf_files(input_root_path))
    logger.info("Discovered %d PDF file(s) in target rule directories.", len(pdf_files))

    for pdf_path in pdf_files:
        relative_pdf = pdf_path.relative_to(input_root_path)
        output_md_path = (output_root_path / relative_pdf).with_suffix(".md")
        output_key = relative_pdf.as_posix()

        try:
            checksum = _sha256_file(pdf_path)
            existing = manifest.get(output_key, {})
            if existing.get("checksum") == checksum and output_md_path.exists():
                skipped_count += 1
                logger.info("Skipping (unchanged): %s", relative_pdf)
                continue

            logger.info("Uploading and processing: %s", relative_pdf)
            markdown = _call_ocr_markdown(client=client, pdf_path=pdf_path, model=model)
            
            output_md_path.parent.mkdir(parents=True, exist_ok=True)
            output_md_path.write_text(markdown, encoding="utf-8")

            manifest[output_key] = {
                "checksum": checksum,
                "output_file": output_md_path.relative_to(output_root_path).as_posix(),
                "model": model,
            }
            _save_manifest(manifest_path, manifest)
            processed_count += 1
            logger.info("Processed: %s", relative_pdf)

        except Exception as exc:
            failed_count += 1
            logger.error("Failed %s: %s", relative_pdf, exc)
            with failed_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{relative_pdf}: {exc}\n")

    logger.info("Done. Processed: %d | Failed: %d", processed_count, failed_count)


if __name__ == "__main__":
    process_f1_docs(input_root=".", output_root="processed_rules")
