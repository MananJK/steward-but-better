"""Build a FAISS vector index from processed FIA markdown rule files."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RULES_DIR = Path("processed_rules")
DEFAULT_INDEX_DIR = Path("steward_vector_db")
INDEX_MANIFEST_FILE = "index_manifest.json"
YEAR_PATTERN = re.compile(r"^(19|20)\d{2}$")
IGNORED_CATEGORY_PARTS = {"rules", "processed_rules", "documents", "fia"}


def _discover_markdown_files(processed_rules_dir: Path) -> List[Path]:
    return sorted(path for path in processed_rules_dir.rglob("*.md") if path.is_file())


def _extract_metadata_from_path(file_path: Path, root_dir: Path) -> Dict[str, str]:
    relative = file_path.relative_to(root_dir)
    parts = relative.parts
    dir_parts = [part for part in parts[:-1] if part]

    year = "unknown"
    for part in dir_parts:
        if YEAR_PATTERN.match(part):
            year = part
            break

    category = "Unknown"
    for part in dir_parts:
        lowered = part.lower()
        if lowered in IGNORED_CATEGORY_PARTS:
            continue
        if YEAR_PATTERN.match(part):
            continue
        category = part.replace("_", " ").replace("-", " ").strip().title()
        if category:
            break

    return {
        "Year": year,
        "Document Category": category,
        "source": relative.as_posix(),
        "filename": file_path.name,
    }


def _chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> Iterable[str]:
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            preferred_break = text.rfind("\n", start + int(chunk_size * 0.6), end)
            if preferred_break != -1:
                end = preferred_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def _prepare_texts_and_metadata(
    markdown_files: List[Path], processed_rules_dir: Path
) -> Tuple[List[str], List[Dict[str, str]]]:
    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for file_path in markdown_files:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        base_metadata = _extract_metadata_from_path(file_path, processed_rules_dir)

        chunks = list(_chunk_text(content))
        for chunk_index, chunk in enumerate(chunks):
            metadata = dict(base_metadata)
            metadata["chunk_id"] = f"{base_metadata['source']}::chunk_{chunk_index}"
            texts.append(chunk)
            metadatas.append(metadata)

    return texts, metadatas


def build_vector_index(
    processed_rules_dir: str | Path = DEFAULT_RULES_DIR,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> Path:
    processed_rules_path = Path(processed_rules_dir).resolve()
    index_path = Path(index_dir).resolve()

    if not processed_rules_path.exists():
        raise FileNotFoundError(
            f"Processed rules directory not found: {processed_rules_path}"
        )

    markdown_files = _discover_markdown_files(processed_rules_path)
    if not markdown_files:
        raise FileNotFoundError(
            f"No markdown files found under: {processed_rules_path}"
        )

    logging.info("Found %d markdown files.", len(markdown_files))
    texts, metadatas = _prepare_texts_and_metadata(markdown_files, processed_rules_path)
    if not texts:
        raise ValueError("No chunkable text found in markdown files.")

    logging.info("Prepared %d text chunks for indexing.", len(texts))

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )

    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))

    manifest = {
        "embedding_model": embedding_model,
        "processed_rules_dir": str(processed_rules_path),
        "index_dir": str(index_path),
        "markdown_files": len(markdown_files),
        "chunks_indexed": len(texts),
    }
    (index_path / INDEX_MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logging.info("Saved FAISS index to %s", index_path)
    return index_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and persist a FAISS index for FIA rule markdown files."
    )
    parser.add_argument(
        "--processed-rules-dir",
        type=Path,
        default=DEFAULT_RULES_DIR,
        help="Directory containing processed markdown rules.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory where the FAISS index will be saved.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help="HuggingFace embedding model.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _build_arg_parser().parse_args()
    build_vector_index(
        processed_rules_dir=args.processed_rules_dir,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
    )
