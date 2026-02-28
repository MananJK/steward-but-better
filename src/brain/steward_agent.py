"""RAG steward agent that grounds rulings in FIA articles and telemetry facts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from mistralai import Mistral

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = Path("steward_vector_db")
DEFAULT_MODEL = "mistral-large-latest"
MAX_CONTEXT_CHARS = 1400


def _extract_years(value: Any, years: set[str] | None = None) -> set[str]:
    if years is None:
        years = set()

    if isinstance(value, dict):
        for key, subvalue in value.items():
            key_lower = str(key).lower()
            if key_lower in {"year", "season"}:
                maybe_year = str(subvalue)
                if re.fullmatch(r"(19|20)\d{2}", maybe_year):
                    years.add(maybe_year)
            _extract_years(subvalue, years)
    elif isinstance(value, list):
        for item in value:
            _extract_years(item, years)
    elif isinstance(value, int):
        maybe_year = str(value)
        if re.fullmatch(r"(19|20)\d{2}", maybe_year):
            years.add(maybe_year)
    elif isinstance(value, str):
        for match in re.findall(r"(?:19|20)\d{2}", value):
            years.add(match)

    return years


def _coerce_incident_json(incident_json: str) -> Dict[str, Any]:
    try:
        loaded = json.loads(incident_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Incident JSON is not valid JSON.") from exc

    if not isinstance(loaded, dict):
        raise ValueError("Incident JSON must be a JSON object.")
    return loaded


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                blocks = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                ]
                return "\n".join(blocks).strip()
    if hasattr(response, "choices"):
        choices = getattr(response, "choices", [])
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                blocks = []
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        blocks.append(block["text"])
                    else:
                        text_attr = getattr(block, "text", None)
                        if isinstance(text_attr, str):
                            blocks.append(text_attr)
                return "\n".join(blocks).strip()
    if hasattr(response, "model_dump"):
        return _extract_response_text(response.model_dump())
    return str(response)


def _load_vector_store(index_dir: str | Path) -> FAISS:
    index_path = Path(index_dir).resolve()
    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector index directory not found: {index_path}. "
            "Build it first with vector_index.py."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _retrieve_articles(
    vector_store: FAISS,
    query: str,
    incident_data: Dict[str, Any],
    k: int = 6,
) -> List[Any]:
    retrieved_docs = vector_store.similarity_search(query, k=max(k * 4, 20))
    year_hints = _extract_years(incident_data)

    if not year_hints:
        return retrieved_docs[:k]

    filtered_docs = [
        doc
        for doc in retrieved_docs
        if str(doc.metadata.get("Year", "unknown")) in year_hints
    ]
    if filtered_docs:
        return filtered_docs[:k]

    return retrieved_docs[:k]


def _format_retrieved_context(docs: Sequence[Any]) -> str:
    context_blocks: List[str] = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        snippet = doc.page_content.strip().replace("\r", "")
        snippet = snippet[:MAX_CONTEXT_CHARS]

        block = (
            f"[Article {index}]\n"
            f"Source: {metadata.get('source', 'unknown')}\n"
            f"Year: {metadata.get('Year', 'unknown')}\n"
            f"Document Category: {metadata.get('Document Category', 'Unknown')}\n"
            f"Text:\n{snippet}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def run_steward_agent(
    query: str,
    incident_json: str,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    model: str = DEFAULT_MODEL,
    k: int = 6,
) -> Dict[str, Any]:
    load_dotenv(override=True)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is missing. Add it to your .env file.")

    incident_data = _coerce_incident_json(incident_json)
    vector_store = _load_vector_store(index_dir=index_dir)
    docs = _retrieve_articles(
        vector_store=vector_store,
        query=query,
        incident_data=incident_data,
        k=k,
    )

    if not docs:
        raise ValueError("No relevant FIA articles were retrieved from the index.")

    articles_context = _format_retrieved_context(docs)
    telemetry_facts = json.dumps(incident_data, indent=2, sort_keys=True)

    system_prompt = (
        "You are an FIA Steward Decision Agent.\n"
        "Use only the provided retrieved FIA articles and telemetry facts.\n"
        "Do not use outside knowledge.\n"
        "If evidence is insufficient, say 'Insufficient evidence based on retrieved FIA articles.'\n"
        "Return exactly these sections:\n"
        "1) Verdict\n"
        "2) Applicable Articles (cite by Article number from context labels)\n"
        "3) Fact-to-Rule Reasoning\n"
        "4) Confidence (High/Medium/Low)"
    )
    user_prompt = (
        f"Query:\n{query}\n\n"
        f"Incident JSON:\n{telemetry_facts}\n\n"
        f"Retrieved FIA Articles:\n{articles_context}\n\n"
        "Generate the final ruling."
    )

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    verdict = _extract_response_text(response).strip()
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "year": doc.metadata.get("Year", "unknown"),
            "document_category": doc.metadata.get("Document Category", "Unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
        }
        for doc in docs
    ]

    return {
        "query": query,
        "verdict": verdict,
        "retrieved_articles": sources,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a steward RAG agent against a FAISS index and telemetry JSON."
    )
    parser.add_argument("--query", type=str, required=True, help="Steward query.")
    parser.add_argument(
        "--incident-json",
        type=str,
        required=True,
        help="Incident telemetry payload as a JSON string.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Path to persisted FAISS index.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Mistral chat model.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of retrieved chunks to use.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _build_arg_parser().parse_args()
    result = run_steward_agent(
        query=args.query,
        incident_json=args.incident_json,
        index_dir=args.index_dir,
        model=args.model,
        k=args.k,
    )
    print(json.dumps(result, indent=2))
