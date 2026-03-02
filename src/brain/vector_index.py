"""Build a FAISS vector index from processed FIA markdown rule files."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
from huggingface_hub import InferenceClient

DEFAULT_RULES_DIR = Path("processed_rules")
DEFAULT_INDEX_FILE = Path("src/brain/fia_rules.index")
DEFAULT_PROGRESS_FILE = Path("src/brain/indexing_progress.json")
HF_API_KEY = os.getenv("HF_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
YEAR_PATTERN = re.compile(r"^(19|20)\d{2}$")
IGNORED_CATEGORY_PARTS = {"rules", "processed_rules", "documents", "fia"}

BATCH_SIZE = 50
BATCHES_BEFORE_SAVE = 5
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5


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
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
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
    index_file: str | Path = DEFAULT_INDEX_FILE,
    progress_file: str | Path = DEFAULT_PROGRESS_FILE,
    embedding_model: str = EMBEDDING_MODEL,
) -> Path:
    processed_rules_path = Path(processed_rules_dir).resolve()
    index_path = Path(index_file).resolve()
    progress_path = Path(progress_file).resolve()

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

    progress = {}
    if progress_path.exists():
        with open(progress_path, "r") as f:
            progress = json.load(f)
            completed_chunks = set(progress.get("completed_chunks", []))
        logging.info(
            "Resuming from checkpoint: %d chunks already embedded",
            len(completed_chunks),
        )
    else:
        completed_chunks = set()

    client = InferenceClient(token=HF_API_KEY)
    logging.info(
        "Generating embeddings with model: %s (batch size: %d)",
        embedding_model,
        BATCH_SIZE,
    )

    all_embeddings = []
    batch_count = 0
    successful_batches = 0
    dimension = None

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(texts))))

        remaining_in_batch = [
            idx for idx in batch_indices if idx not in completed_chunks
        ]

        if not remaining_in_batch:
            logging.info(
                "Skipping batch %d-%d (already complete)",
                i + 1,
                min(i + BATCH_SIZE, len(texts)),
            )
            continue

        batch_to_embed = [texts[idx] for idx in remaining_in_batch]

        for attempt in range(MAX_RETRIES + 1):
            try:
                batch_embeddings = []
                for text in batch_to_embed:
                    embedding = client.feature_extraction(text, model=embedding_model)
                    batch_embeddings.append(embedding)
                break
            except Exception as e:
                error_str = str(e)
                if "500" in error_str or "503" in error_str:
                    if attempt < MAX_RETRIES:
                        wait_time = RETRY_DELAY_SECONDS * (2**attempt)
                        logging.warning(
                            "Batch %d-%d hit %s error, retrying in %ds (attempt %d/%d)",
                            i + 1,
                            min(i + BATCH_SIZE, len(texts)),
                            error_str[:50],
                            wait_time,
                            attempt + 1,
                            MAX_RETRIES,
                        )
                        time.sleep(wait_time)
                    else:
                        logging.error("Batch failed after %d retries", MAX_RETRIES)
                        raise
                else:
                    raise

        embedding_array = np.array(batch_embeddings, dtype=np.float32)

        if dimension is None:
            dimension = embedding_array.shape[1]

        for j, idx in enumerate(remaining_in_batch):
            while len(all_embeddings) <= idx:
                all_embeddings.append(None)
            all_embeddings[idx] = embedding_array[j]

        completed_chunks.update(remaining_in_batch)

        progress = {
            "completed_chunks": list(completed_chunks),
            "last_batch_index": i,
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_path, "w") as f:
            json.dump(progress, f)

        batch_count += 1
        successful_batches += 1
        logging.info(
            "Embedded batch %d-%d (%d/%d chunks done)",
            i + 1,
            min(i + BATCH_SIZE, len(texts)),
            len(completed_chunks),
            len(texts),
        )

        if successful_batches > 0 and successful_batches % BATCHES_BEFORE_SAVE == 0:
            valid_embeddings = [e for e in all_embeddings if e is not None]
            if valid_embeddings:
                embeddings_matrix = np.vstack(valid_embeddings)
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings_matrix)
                index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(index, str(index_path))
                logging.info(
                    "Incremental save: FAISS index saved after %d batches",
                    successful_batches,
                )

    valid_embeddings = [e for e in all_embeddings if e is not None]
    embeddings = np.vstack(valid_embeddings)

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with open(str(index_path).replace(".index", "_metadata.json"), "w") as f:
        json.dump(
            {
                "texts": texts,
                "metadatas": metadatas,
            },
            f,
        )

    if progress_path.exists():
        progress_path.unlink()

    manifest = {
        "embedding_model": embedding_model,
        "processed_rules_dir": str(processed_rules_path),
        "index_file": str(index_path),
        "markdown_files": len(markdown_files),
        "chunks_indexed": len(texts),
    }
    manifest_path = index_path.parent / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
        "--index-file",
        type=Path,
        default=DEFAULT_INDEX_FILE,
        help="Path where the FAISS index will be saved.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help="HuggingFace embedding model.",
    )
    return parser


def test_search(
    query: str = "Article 33.4", index_file: str | Path = DEFAULT_INDEX_FILE
):
    index_path = Path(index_file).resolve()
    metadata_path = index_path.parent / "fia_rules_metadata.json"

    if not index_path.exists():
        print(f"Index not found at {index_path}. Run build_vector_index first.")
        return

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r") as f:
        data = json.load(f)
        texts = data["texts"]

    client = InferenceClient(token=HF_API_KEY)
    query_embedding = client.feature_extraction(
        query,
        model=EMBEDDING_MODEL,
    )
    query_vector = np.array([query_embedding], dtype=np.float32)

    k = 1
    distances, indices = index.search(query_vector, k)

    print(f"\n{'=' * 60}")
    print(f"Test Search Query: '{query}'")
    print(f"{'=' * 60}")
    print(f"Top {k} match:")
    print(f"Distance: {distances[0][0]:.4f}")
    print(f"Index: {indices[0][0]}")
    print(f"\nContent:\n{texts[indices[0][0]][:500]}...")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _build_arg_parser().parse_args()
    index_path = build_vector_index(
        processed_rules_dir=args.processed_rules_dir,
        index_file=args.index_file,
        embedding_model=args.embedding_model,
    )
    test_search(index_file=index_path)
