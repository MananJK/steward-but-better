"""Build and validate a FAISS vector index for FIA rule documents.

The index file, its metadata JSON, and the manifest are written atomically as a
consistent set: all three are staged to temp files, the staged index is read
back and validated (vector count == text count), and only then are the files
moved into place. A mismatched index/metadata pair can therefore never exist on
disk, which is what `_load_vector_store` in steward_agent.py relies on.

Embeddings are computed locally with sentence-transformers (same model that
steward_agent.py uses at query time), so index vectors and query vectors are
always produced by the same model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_DIR = REPO_ROOT / "processed_rules"
DEFAULT_INDEX_FILE = Path(__file__).resolve().parent / "fia_rules.index"
DEFAULT_METADATA_FILE = Path(__file__).resolve().parent / "fia_rules_metadata.json"
DEFAULT_MANIFEST_FILE = Path(__file__).resolve().parent / "index_manifest.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
ARTICLE_HEADING_PATTERN = re.compile(
    r"^\s*(?:#+\s*)?(?:article|art\.?)\s+\d+(?:\.\d+)*\b", re.IGNORECASE
)
# FIA Sporting Regulations use "## 54) INCIDENTS" style section headings and
# "54.3" style clause numbers rather than ISC-style "Article 54.3".
ARTICLE_REFERENCE_PATTERN = re.compile(
    r"\b(?:article|art\.?)\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE
)
SECTION_HEADING_PATTERN = re.compile(r"^\s*(?:#+\s*)?(\d+)\)\s+\S", re.MULTILINE)
CLAUSE_NUMBER_PATTERN = re.compile(r"^\s*(\d+\.\d+)\s+[A-Za-z]", re.MULTILINE)

BATCH_SIZE = 64
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 80

logger = logging.getLogger(__name__)


def _discover_markdown_files(processed_rules_dir: Path) -> List[Path]:
    return sorted(path for path in processed_rules_dir.rglob("*.md") if path.is_file())


def _extract_metadata_from_path(file_path: Path, root_dir: Path) -> Dict[str, str]:
    relative = file_path.relative_to(root_dir)
    parts = relative.parts

    year = "unknown"
    for part in parts:
        match = YEAR_PATTERN.search(part)
        if match:
            year = match.group(0)
            break

    category = _derive_category(relative.as_posix())

    return {
        "Year": year,
        "Document Category": category,
        "source": relative.as_posix(),
        "filename": file_path.name,
    }


def _derive_category(source: str) -> str:
    """Derive an honest document category from the source path.

    The directory name is the primary signal, but filenames override it when a
    document is misfiled (e.g. the technical regulations sitting under
    driving_standards/).
    """
    lowered = source.lower()
    if "technical" in lowered:
        return "Technical Regulations"
    if "driving_standards" in lowered or "driving standards" in lowered:
        return "Driving Standards"
    if "steward_standards" in lowered or "steward standards" in lowered:
        return "Steward Standards"
    if "sporting" in lowered:
        return "Sporting Regulations"
    return "Unknown"


def _extract_article(text: str) -> str | None:
    """Return the strongest legal reference in a chunk.

    Preference order: explicit ISC-style "Article 33.3", then FIA section
    headings ("54) INCIDENTS"), then leading clause numbers ("54.3 ...").
    """
    match = ARTICLE_REFERENCE_PATTERN.search(text)
    if match:
        return f"Article {match.group(1)}"

    section_match = SECTION_HEADING_PATTERN.search(text)
    if section_match:
        section = section_match.group(1)
        clause = CLAUSE_NUMBER_PATTERN.search(text)
        if clause and clause.group(1).split(".")[0] == section:
            return f"Clause {clause.group(1)}"
        return f"Section {section}"

    clause_match = CLAUSE_NUMBER_PATTERN.search(text[:400])
    if clause_match:
        return f"Clause {clause_match.group(1)}"

    return None


def _enrich_metadata(metadata: Dict[str, str], text: str) -> Dict[str, str]:
    """Normalize metadata for retrieval-time filtering and citations."""
    enriched = dict(metadata)
    source = enriched.get("source", "")
    if enriched.get("Year", "unknown") == "unknown":
        match = YEAR_PATTERN.search(source)
        enriched["Year"] = match.group(0) if match else "unknown"
    enriched["Document Category"] = _derive_category(source)
    if not enriched.get("article"):
        article = _extract_article(text)
        if article:
            enriched["article"] = article
    return enriched


def _split_sections(text: str) -> List[str]:
    """Split markdown text into sections at markdown headings and article headings."""
    lines = text.splitlines()
    sections: List[str] = []
    current: List[str] = []

    for line in lines:
        is_heading = line.lstrip().startswith("#") or ARTICLE_HEADING_PATTERN.match(line)
        if is_heading and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return [section for section in sections if section]


def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Chunk text without crossing section/article boundaries when avoidable."""
    text = text.strip()
    if not text:
        return []

    sections = _split_sections(text)

    # Merge tiny sections (headers, spacers) into their neighbours so chunking
    # does not produce fragments with no semantic content.
    merged: List[str] = []
    for section in sections:
        if merged and (len(section) < MIN_CHUNK_CHARS or len(merged[-1]) < MIN_CHUNK_CHARS):
            merged[-1] = merged[-1] + "\n\n" + section
        else:
            merged.append(section)

    chunks: List[str] = []
    for section in merged:
        if len(section) <= chunk_size:
            chunks.append(section)
            continue

        start = 0
        length = len(section)
        while start < length:
            end = min(start + chunk_size, length)
            if end < length:
                preferred_break = section.rfind("\n", start + int(chunk_size * 0.6), end)
                if preferred_break != -1:
                    end = preferred_break
            chunk = section[start:end].strip()
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

        for chunk_index, chunk in enumerate(_chunk_text(content)):
            metadata = _enrich_metadata(base_metadata, chunk)
            metadata["chunk_id"] = f"{base_metadata['source']}::chunk_{chunk_index}"
            texts.append(chunk)
            metadatas.append(metadata)

    return texts, metadatas


def _embed_texts(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _atomic_write_set(
    index_file: Path,
    metadata_file: Path,
    manifest_file: Path,
    embeddings: np.ndarray,
    texts: List[str],
    metadatas: List[Dict[str, str]],
    source_description: str,
) -> None:
    """Write index + metadata + manifest as one validated, atomic set."""
    dimension = int(embeddings.shape[1])
    if len(texts) != embeddings.shape[0]:
        raise ValueError(
            f"Refusing to write index: {len(texts)} texts but "
            f"{embeddings.shape[0]} embeddings."
        )

    index_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=index_file.parent) as tmp_dir:
        tmp_index = Path(tmp_dir) / "index.tmp"
        tmp_metadata = Path(tmp_dir) / "metadata.tmp"
        tmp_manifest = Path(tmp_dir) / "manifest.tmp"

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(tmp_index))

        # Validate the staged index before anything is moved into place.
        staged = faiss.read_index(str(tmp_index))
        if staged.ntotal != len(texts) or staged.d != dimension:
            raise ValueError(
                f"Staged index failed validation: ntotal={staged.ntotal} "
                f"(expected {len(texts)}), d={staged.d} (expected {dimension})."
            )

        tmp_metadata.write_text(
            json.dumps({"texts": texts, "metadatas": metadatas}, ensure_ascii=False),
            encoding="utf-8",
        )

        manifest = {
            "embedding_model": EMBEDDING_MODEL,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source": source_description,
            "index_file": str(index_file.relative_to(REPO_ROOT)),
            "metadata_file": str(metadata_file.relative_to(REPO_ROOT)),
            "chunks_indexed": len(texts),
            "dimension": dimension,
        }
        tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        os.replace(tmp_index, index_file)
        os.replace(tmp_metadata, metadata_file)
        os.replace(tmp_manifest, manifest_file)

    logger.info(
        "Wrote validated index set: %d chunks, d=%d (%s)",
        len(texts),
        dimension,
        source_description,
    )


def build_vector_index(
    processed_rules_dir: str | Path = DEFAULT_RULES_DIR,
    index_file: str | Path = DEFAULT_INDEX_FILE,
) -> Path:
    """Build the index from processed markdown rule files."""
    processed_rules_path = Path(processed_rules_dir).resolve()
    index_path = Path(index_file).resolve()

    if not processed_rules_path.exists():
        raise FileNotFoundError(
            f"Processed rules directory not found: {processed_rules_path}. "
            "Run the OCR ingestion step (src/ingestion/ocr_processor.py) first."
        )

    markdown_files = _discover_markdown_files(processed_rules_path)
    if not markdown_files:
        raise FileNotFoundError(f"No markdown files found under: {processed_rules_path}")

    logger.info("Found %d markdown files.", len(markdown_files))
    texts, metadatas = _prepare_texts_and_metadata(markdown_files, processed_rules_path)
    if not texts:
        raise ValueError("No chunkable text found in markdown files.")

    logger.info("Prepared %d text chunks for indexing.", len(texts))
    embeddings = _embed_texts(texts)

    metadata_path = index_path.with_name(index_path.stem + "_metadata.json")
    manifest_path = index_path.parent / "index_manifest.json"
    _atomic_write_set(
        index_path, metadata_path, manifest_path, embeddings, texts, metadatas,
        source_description=str(processed_rules_path),
    )
    return index_path


def rebuild_from_metadata(
    metadata_file: str | Path = DEFAULT_METADATA_FILE,
    index_file: str | Path = DEFAULT_INDEX_FILE,
) -> Path:
    """Re-embed an existing metadata JSON and rebuild a consistent index.

    Recovery path for a mismatched index/metadata pair: chunk boundaries are
    preserved (texts are re-embedded as-is), metadata is re-normalized, and the
    result is written as a validated set.
    """
    metadata_path = Path(metadata_file).resolve()
    index_path = Path(index_file).resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    texts = payload.get("texts", [])
    raw_metadatas = payload.get("metadatas", [])
    if not texts or len(texts) != len(raw_metadatas):
        raise ValueError(
            f"Metadata file is unusable: {len(texts)} texts vs "
            f"{len(raw_metadatas)} metadatas."
        )

    logger.info("Rebuilding index from %d existing chunks.", len(texts))
    metadatas = [_enrich_metadata(m, t) for m, t in zip(raw_metadatas, texts)]
    embeddings = _embed_texts(texts)

    out_metadata_path = index_path.with_name(index_path.stem + "_metadata.json")
    manifest_path = index_path.parent / "index_manifest.json"
    _atomic_write_set(
        index_path, out_metadata_path, manifest_path, embeddings, texts, metadatas,
        source_description=f"rebuild_from_metadata:{metadata_path.name}",
    )
    return index_path


def validate_index(
    index_file: str | Path = DEFAULT_INDEX_FILE,
    metadata_file: str | Path = DEFAULT_METADATA_FILE,
) -> bool:
    """Check that the index and metadata describe the same chunk set."""
    index_path = Path(index_file)
    metadata_path = Path(metadata_file)

    if not index_path.exists() or not metadata_path.exists():
        logger.error("Index or metadata file missing: %s / %s", index_path, metadata_path)
        return False

    index = faiss.read_index(str(index_path))
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_texts = len(payload.get("texts", []))

    if index.ntotal != n_texts:
        logger.error(
            "Index/metadata mismatch: index has %d vectors but metadata has %d texts. "
            "Rebuild with: python src/brain/vector_index.py --rebuild-from-metadata",
            index.ntotal,
            n_texts,
        )
        return False

    logger.info("Index OK: %d vectors, d=%d, %d texts.", index.ntotal, index.d, n_texts)
    return True


def smoke_search(query: str, index_file: str | Path = DEFAULT_INDEX_FILE,
                 metadata_file: str | Path = DEFAULT_METADATA_FILE, k: int = 3) -> None:
    """Run a query against the index and print top chunks with metadata."""
    index_path = Path(index_file)
    payload = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
    texts = payload["texts"]
    metadatas = payload["metadatas"]

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = np.asarray(
        [model.encode(query, convert_to_numpy=True, normalize_embeddings=False)],
        dtype=np.float32,
    )

    index = faiss.read_index(str(index_path))
    k = min(k, index.ntotal)
    distances, indices = index.search(query_vector, k)

    print(f"\nQuery: '{query}' — top {k} chunks")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        meta = metadatas[idx]
        print(f"\n#{rank} (L2={dist:.3f}) [{meta.get('Year')}] "
              f"{meta.get('Document Category')} | {meta.get('article', '-')} "
              f"| {meta.get('source')}")
        print(f"   {texts[idx][:200].replace(chr(10), ' ')}...")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, rebuild, or validate the FAISS index for FIA rule documents."
    )
    parser.add_argument(
        "--processed-rules-dir", type=Path, default=DEFAULT_RULES_DIR,
        help="Directory containing processed markdown rules.",
    )
    parser.add_argument(
        "--index-file", type=Path, default=DEFAULT_INDEX_FILE,
        help="Path where the FAISS index will be written.",
    )
    parser.add_argument(
        "--rebuild-from-metadata", type=Path, default=None, metavar="METADATA_JSON",
        help="Re-embed an existing metadata JSON and rebuild the index from it.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate that the index and metadata files are consistent.",
    )
    parser.add_argument(
        "--search", type=str, default=None, metavar="QUERY",
        help="Run a smoke-search query against the index and print results.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_arg_parser().parse_args()

    if args.validate:
        ok = validate_index(args.index_file)
        raise SystemExit(0 if ok else 1)

    if args.search:
        metadata_path = args.index_file.with_name(args.index_file.stem + "_metadata.json")
        smoke_search(args.search, args.index_file, metadata_path)
        raise SystemExit(0)

    if args.rebuild_from_metadata:
        rebuild_from_metadata(args.rebuild_from_metadata, args.index_file)
    else:
        build_vector_index(args.processed_rules_dir, args.index_file)
