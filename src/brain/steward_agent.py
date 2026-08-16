"""Telemetry-aware steward agent that retrieves FIA rules and emits dashboard-ready verdicts.

Design notes:
- Heavy dependencies (faiss, langchain, mistralai) are imported lazily so the
  pure decision logic in this module stays testable without them installed.
- The vector store is loaded once per process and validated against its
  metadata file: the FAISS index must contain exactly one vector per text.
- Confidence is a heuristic evidence-strength score derived from the number of
  independent telemetry signals that fired. It is not a calibrated probability.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / "fia_rules.index"
MISTRAL_MODEL = "mistral-small-latest"

CURRENT_RULES_YEAR = "2025"
# Documents that can govern on-track driving conduct. Technical Regulations
# govern car construction, not driving, so they are excluded from conduct
# incidents (still retrievable when a component failure is flagged).
CONDUCT_CATEGORIES = {"Sporting Regulations", "Driving Standards", "Steward Standards"}
TECHNICAL_CATEGORIES = {"Technical Regulations"}

HIGH_PRIORITY_INCIDENT_TYPES = {
    "high_g_event",
    "collision",
    "contact",
    "off_track",
    "forced_wide",
    "driver_agnostic_incident",
}

SYSTEM_PROMPT = """You are an objective FIA Steward. You are being provided with anonymized telemetry for 'Driver A' and 'Driver B'. Ignore any historical context or driver reputation and judge solely on the 2025 Driving Standards Guidelines."""

_VECTOR_STORE_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# LLM verdict (optional; deterministic path works without it)
# ---------------------------------------------------------------------------

def _get_mistral_client():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    from mistralai import Mistral

    return Mistral(api_key=api_key)


def _generate_llm_verdict(
    incident_data: dict[str, Any],
    features: dict[str, Any],
    retrieved_docs: list,
) -> dict[str, Any] | None:
    """Generate a natural-language verdict with Mistral.

    Driver identity is deliberately withheld from the prompt so the LLM judges
    the telemetry, not the driver.
    """
    try:
        client = _get_mistral_client()
    except Exception as e:
        print(f"[LLM] Mistral client unavailable: {e}")
        return None

    rules_context = (
        "\n\n".join(f"- {str(doc.page_content)[:500]}" for doc in retrieved_docs[:3])
        if retrieved_docs
        else "No specific rules retrieved."
    )

    prompt = f"""You are an FIA Steward analyzing an F1 racing incident.

TELEMETRY DATA (anonymized):
- Driver under review: Driver A (identity withheld)
- Rival car involved: Driver B (identity withheld)
- Speed: {incident_data.get('speed_kph', 0)} km/h
- Lateral G-Force: {incident_data.get('lateral_g', 0)}G
- Sector: {incident_data.get('sector', 'UNKNOWN')}
- Lap: {incident_data.get('lap', 0)}
- Incident Type: {incident_data.get('incident_type', 'unknown_incident')}

DETECTED FEATURES:
- High lateral load: {features.get('high_lateral_load', False)}
- Collision signal: {features.get('collision_signal', False)}
- Off-track signal: {features.get('off_track_signal', False)}

RELEVANT FIA RULES:
{rules_context}

Based on the telemetry and FIA regulations, issue a verdict.

Determine:
1. RULING: PENALTY, INVESTIGATION, or NO_FURTHER_ACTION
2. CITATION: The specific FIA article violated (e.g., "FIA International Sporting Code - Appendix L, Chapter IV, Article 2(d)")
3. SUMMARY: A 1-2 sentence explanation of the decision

Respond in JSON format:
{{
  "ruling": "PENALTY",
  "article_cited": "FIA International Sporting Code - Appendix L, Chapter IV, Article 2(d)",
  "rule_summary": "..."
}}"""

    try:
        chat_response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        response_text = chat_response.choices[0].message.content
        print(f"[LLM] Raw response: {response_text[:200]}...")
        return json.loads(response_text)

    except Exception as e:
        print(f"[LLM] Error generating verdict: {e}")
        return None


# ---------------------------------------------------------------------------
# Incident coercion helpers
# ---------------------------------------------------------------------------

def _coerce_incident_json(incident_json: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(incident_json, dict):
        return incident_json
    try:
        loaded = json.loads(incident_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Incident JSON is not valid JSON.") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Incident JSON must be a JSON object.")
    return loaded


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return None


def _parse_lateral_series(incident_data: dict[str, Any]) -> list[float]:
    telemetry = incident_data.get("telemetry")
    candidates = [
        incident_data.get("lateral_g_series"),
        incident_data.get("lateral_gs"),
        telemetry.get("lateral_g_series") if isinstance(telemetry, dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            parsed = [_safe_float(item) for item in candidate]
            return [item for item in parsed if item is not None]
    return []


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(incident_data: dict[str, Any]) -> dict[str, Any]:
    lateral_g = _safe_float(incident_data.get("lateral_g"))
    braking_force = _safe_float(incident_data.get("braking_force"))
    apex_clearance = _safe_float(
        incident_data.get("apex_clearance") or incident_data.get("apex_gap")
    )
    incident_type = str(incident_data.get("incident_type", "")).lower()
    description = str(
        incident_data.get("incident_description")
        or incident_data.get("incident_snapshot")
        or ""
    ).lower()

    text_blob = " ".join([incident_type, description])

    evasive_braking = _coerce_bool(incident_data.get("evasive_braking"))
    no_evasive_braking = (
        evasive_braking is False
        or (braking_force is not None and braking_force < 0.75)
    )

    lateral_series = _parse_lateral_series(incident_data)
    sudden_lateral_drop = False
    if len(lateral_series) >= 2:
        peak = max(lateral_series)
        latest = lateral_series[-1]
        largest_step = max(
            abs(lateral_series[i] - lateral_series[i - 1])
            for i in range(1, len(lateral_series))
        )
        sudden_lateral_drop = (peak - latest) >= 1.2 or largest_step >= 1.0

    component_failure_flag = _coerce_bool(incident_data.get("component_failure"))
    if component_failure_flag is None:
        component_failure_flag = "component failure" in text_blob

    return {
        "lateral_g": lateral_g,
        "braking_force": braking_force,
        "apex_clearance": apex_clearance,
        "high_lateral_load": lateral_g is not None and lateral_g >= 4.5,
        "hard_braking": braking_force is not None and braking_force >= 0.75,
        "no_evasive_braking": no_evasive_braking,
        "low_clearance": apex_clearance is not None and apex_clearance < 2.0,
        "sudden_lateral_drop": sudden_lateral_drop,
        "collision_signal": any(
            token in text_blob
            for token in ["collision", "contact", "hit", "crash", "impact"]
        )
        or sudden_lateral_drop,
        "off_track_signal": any(
            token in text_blob
            for token in ["off track", "off-track", "leaving the track", "forced wide"]
        ),
        "component_failure_flag": component_failure_flag is True,
        "incident_type": incident_type,
    }


def _is_driving_conduct_incident(features: dict[str, Any]) -> bool:
    if features.get("component_failure_flag", False):
        return False
    return True


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _load_vector_store(index_dir: str | Path):
    """Load (and cache) the FAISS store, refusing mismatched index/metadata pairs."""
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    import faiss

    index_path = Path(index_dir).resolve()
    cache_key = str(index_path)
    if cache_key in _VECTOR_STORE_CACHE:
        return _VECTOR_STORE_CACHE[cache_key]

    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector index path not found: {index_path}. "
            "Build it first with src/brain/vector_index.py."
        )

    metadata_path = (
        index_path.with_name(f"{index_path.stem}_metadata.json")
        if index_path.suffix == ".index"
        else index_path / "fia_rules_metadata.json"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Vector index metadata file not found: {metadata_path}.")

    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    texts = metadata_payload.get("texts", [])
    metadatas = metadata_payload.get("metadatas", [])
    if len(texts) != len(metadatas):
        raise ValueError(
            "Vector metadata mismatch: texts and metadatas lengths do not match."
        )

    raw_index = faiss.read_index(str(index_path))
    if raw_index.ntotal != len(texts):
        raise ValueError(
            f"Index/metadata mismatch: FAISS index has {raw_index.ntotal} vectors "
            f"but metadata lists {len(texts)} texts. Rebuild a consistent set with: "
            f"python src/brain/vector_index.py --rebuild-from-metadata {metadata_path}"
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    documents = {
        str(i): Document(page_content=texts[i], metadata=metadatas[i])
        for i in range(len(texts))
    }
    index_to_docstore_id = {i: str(i) for i in range(len(texts))}

    store = FAISS(
        embedding_function=embeddings,
        index=raw_index,
        docstore=InMemoryDocstore(documents),
        index_to_docstore_id=index_to_docstore_id,
    )
    _VECTOR_STORE_CACHE[cache_key] = store
    return store


def _doc_category(doc: Any) -> str:
    return str((getattr(doc, "metadata", {}) or {}).get("Document Category", ""))


def _doc_year(doc: Any) -> str:
    return str((getattr(doc, "metadata", {}) or {}).get("Year", "unknown"))


def _retrieve_articles(
    vector_store,
    query: str,
    features: dict[str, Any],
    k: int = 6,
) -> list:
    """Retrieve rule chunks using metadata filters instead of string heuristics.

    Physics-flavored queries (speeds, G-loads) semantically match the Technical
    Regulations, so we over-fetch and filter by category: conduct incidents
    must cite driving/sporting documents, never car-construction rules.
    """
    candidates = vector_store.similarity_search(query, k=max(k * 8, 48))
    print(f"[RAG_RETRIEVAL] Query: '{query}' | candidates: {len(candidates)}")

    if _is_driving_conduct_incident(features):
        candidates = [
            doc for doc in candidates
            if _doc_category(doc) not in TECHNICAL_CATEGORIES
        ]

    # Prefer current-season rules; older seasons remain available as precedent.
    current = [d for d in candidates if _doc_year(d) == CURRENT_RULES_YEAR]
    older = [d for d in candidates if _doc_year(d) != CURRENT_RULES_YEAR]
    ordered = current + older

    print(
        f"[RAG_RETRIEVAL] After filters: {len(ordered)} docs "
        f"({len(current)} from {CURRENT_RULES_YEAR})"
    )
    return ordered[:k]


def _derive_citation(doc: Any) -> str:
    metadata = doc.metadata or {}
    article = metadata.get("article")
    if article:
        return str(article)

    content = str(getattr(doc, "page_content", ""))
    match = re.search(r"Article\s*\d+(?:\.\d+)*", content, flags=re.IGNORECASE)
    if match:
        return match.group(0).strip()

    source = str(metadata.get("source", ""))
    filename = source.rsplit("/", 1)[-1].removesuffix(".md") if source else ""
    return filename if filename else "FIA Rule Reference"


def _summarize_rule(doc: Any) -> str:
    text = str(getattr(doc, "page_content", ""))
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # strip md headings
    text = text.replace("\n", " ").strip()
    if not text:
        return "Rule summary unavailable."

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    summary_sentences = []
    char_count = 0
    for sentence in sentences:
        summary_sentences.append(sentence)
        char_count += len(sentence)
        if char_count > 200 or len(summary_sentences) >= 2:
            break

    return ". ".join(summary_sentences) + "."


def _detect_rule_conflict(docs: list) -> tuple[bool, list[str]]:
    article_hits: list[str] = []
    for doc in docs:
        content = str(getattr(doc, "page_content", "")).lower()
        articles = re.findall(r"article\s*\d+(?:\.\d+)*", content, flags=re.IGNORECASE)
        article_hits.extend(
            re.sub(r"\s+", " ", item.strip().title()) for item in articles
        )

    unique_articles = sorted(set(article_hits))
    # Multiple distinct articles from the retrieved context is normal in the
    # regulations; treat it as informational unless it is crowded.
    conflicting = len(unique_articles) >= 4
    return conflicting, unique_articles


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def _decide_verdict(features: dict[str, Any]) -> tuple[str, list[str], float]:
    severity = 0
    reasons: list[str] = []

    if features["collision_signal"]:
        severity += 4
        reasons.append(
            "Telemetry and incident descriptors indicate potential contact/collision."
        )
    if features["off_track_signal"]:
        severity += 4
        reasons.append(
            "Signals suggest a possible leaving-the-track or forcing-wide event."
        )
    if features["lateral_g"] and features["lateral_g"] >= 3.75:
        severity += 5
        reasons.append(
            f"Very high lateral G-force ({features['lateral_g']:.1f}G) indicates "
            "significant cornering conflict or collision impact."
        )
    if features["low_clearance"]:
        severity += 2
        reasons.append("Apex clearance is below one car width threshold (2.0 m).")
    if features["hard_braking"]:
        severity += 1
        reasons.append(
            "Braking force indicates an aggressive braking phase during the incident window."
        )
    if features["high_lateral_load"]:
        severity += 1
        reasons.append("High lateral load supports a high-risk cornering conflict context.")

    if severity >= 4:
        ruling = "PENALTY"
    elif severity >= 2:
        ruling = "INVESTIGATION"
    else:
        ruling = "NO_FURTHER_ACTION"

    # Heuristic evidence-strength score, not a calibrated probability.
    confidence = min(0.95, 0.5 + min(0.45, severity * 0.08))
    return ruling, reasons, confidence


# ---------------------------------------------------------------------------
# Payload building (single source of truth for the dashboard shape)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_dashboard_payload(
    incident_data: dict[str, Any],
    query: str,
    *,
    ruling: str,
    reasons: list[str],
    confidence: float,
    article_cited: str | None,
    rule_summary: str,
    retrieved_docs: list,
    judicial_verdict: str,
) -> dict[str, Any]:
    return {
        "id": incident_data.get("id", "live-incident"),
        "sessionName": incident_data.get("sessionName")
        or incident_data.get("track")
        or "Race Control",
        "track": incident_data.get("track"),
        "timestamp": incident_data.get("timestamp") or _now_iso(),
        "lastUpdated": _now_iso(),
        "driver": incident_data.get("driver", "--"),
        "driver_a": incident_data.get("driver_a"),
        "driver_b": incident_data.get("driver_b"),
        "incident_type": incident_data.get("incident_type", "incident_review"),
        "incident_description": incident_data.get("incident_description")
        or incident_data.get("incident_snapshot")
        or query,
        "speed_kph": incident_data.get("speed_kph"),
        "delta_to_leader": incident_data.get("delta_to_leader")
        or incident_data.get("apex_gap"),
        "track_temp_c": incident_data.get("track_temp_c"),
        "sector": incident_data.get("sector", "N/A"),
        "lap": incident_data.get("lap", 0),
        "article_cited": article_cited,
        "rule_summary": rule_summary,
        "ruling": ruling,
        "verdict": ruling,
        "confidence_score": round(confidence, 2),
        "judicial_verdict": judicial_verdict,
        "retrieved_articles": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "year": doc.metadata.get("Year", "unknown"),
                "document_category": doc.metadata.get("Document Category", "Unknown"),
                "article": doc.metadata.get("article", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            }
            for doc in retrieved_docs
        ],
        "query": query,
        "system_prompt": SYSTEM_PROMPT,
    }


def _no_action_payload(
    incident_data: dict[str, Any], query: str, message: str, confidence: float
) -> dict[str, Any]:
    return _build_dashboard_payload(
        incident_data,
        query,
        ruling="NO_FURTHER_ACTION",
        reasons=[],
        confidence=confidence,
        article_cited=None,
        rule_summary=message,
        retrieved_docs=[],
        judicial_verdict=f"Judicial Verdict: NO_FURTHER_ACTION. {message}",
    )


# ---------------------------------------------------------------------------
# Retrieval query
# ---------------------------------------------------------------------------

def _build_retrieval_query(
    query: str, incident_data: dict[str, Any]
) -> str:
    incident_type = incident_data.get("incident_type", "unknown")
    speed = incident_data.get("speed_kph")
    lateral_g = incident_data.get("lateral_g")

    speed_delta_trigger = incident_data.get("speed_delta_trigger", {})
    proximity_trigger = incident_data.get("proximity_trigger", {})
    speed_reason = (
        speed_delta_trigger.get("reason", "")
        if isinstance(speed_delta_trigger, dict)
        else ""
    )
    proximity_reason = (
        proximity_trigger.get("reason", "")
        if isinstance(proximity_trigger, dict)
        else ""
    )

    base_query = f"Incident: {incident_type}"
    if speed is not None:
        base_query += f" at {speed} km/h"
    if lateral_g is not None:
        base_query += f" with {lateral_g}G lateral force"
    if speed_reason:
        base_query += f" - {speed_reason}"
    elif proximity_reason:
        base_query += f" - {proximity_reason}"
    # Conduct language steers retrieval toward driving/sporting rules: pure
    # telemetry phrasing semantically matches the Technical Regulations.
    base_query += (
        ". Driving conduct review: possible collision, contact, or forcing"
        " another car off track while overtaking, judged under the FIA Sporting"
        " Regulations and Driving Standards Guidelines."
    )
    if query and query != incident_type:
        base_query += f" Context: {query}"

    return base_query


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_steward_agent(
    query: str,
    incident_json: str | dict[str, Any],
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    k: int = 6,
) -> dict[str, Any]:
    incident_data = _coerce_incident_json(incident_json)
    features = _extract_features(incident_data)
    incident_type = str(incident_data.get("incident_type", "")).lower()

    print(f"[STEWARD_AGENT] Processing incident_type: {incident_type}")

    is_high_priority = incident_type in HIGH_PRIORITY_INCIDENT_TYPES or any(
        trigger in incident_type for trigger in HIGH_PRIORITY_INCIDENT_TYPES
    )

    if not is_high_priority:
        message = (
            "Telemetry shows normal racing parameters with no incident indicators."
            if incident_type == "normal_telemetry"
            else "No high-priority incident triggers detected in telemetry."
        )
        return _no_action_payload(incident_data, query, message, 0.95)

    vector_store = _load_vector_store(index_dir=index_dir)
    print(f"Link established with {len(vector_store.docstore._dict)} rule chunks")

    augmented_query = _build_retrieval_query(query, incident_data)
    print(f"[RETRIEVAL_QUERY] '{augmented_query}'")
    docs = _retrieve_articles(
        vector_store=vector_store,
        query=augmented_query,
        features=features,
        k=k,
    )
    print(f"[RETRIEVED] {len(docs)} docs")

    ruling, reasons, confidence = _decide_verdict(features)

    article_cited = "General Driving Standards"
    rule_summary = (
        reasons[0] if reasons else "High G-force event detected."
    )

    if docs:
        article_cited = _derive_citation(docs[0])
        rule_summary = _summarize_rule(docs[0])

    llm_result = _generate_llm_verdict(
        incident_data=incident_data, features=features, retrieved_docs=docs
    )
    if llm_result:
        ruling = llm_result.get("ruling", ruling)
        rule_summary = llm_result.get("rule_summary", rule_summary)
        article_cited = llm_result.get("article_cited", article_cited)
        reasons = [rule_summary] if rule_summary else reasons
        print(f"[LLM] LLM verdict applied: ruling={ruling}, article={article_cited}")

    conflict_detected, conflicting_articles = _detect_rule_conflict(docs)
    if conflict_detected:
        reasons.append(
            "Retrieved rule set spans several articles; verdict downgraded pending "
            f"manual review ({', '.join(conflicting_articles[:4])})."
        )
        if ruling == "PENALTY":
            ruling = "INVESTIGATION"

    if not docs and not llm_result:
        rule_summary = f"Incident flagged. {rule_summary}"

    evidence_lines = [f"Retrieved citation: {article_cited}", f"Rule text: {rule_summary}"]
    evidence_lines.extend(reasons)
    judicial_verdict = (
        f"Judicial Verdict: {ruling}. "
        f"Telemetry-to-rule reasoning: {' '.join(evidence_lines)}"
    )

    payload = _build_dashboard_payload(
        incident_data,
        query,
        ruling=ruling,
        reasons=reasons,
        confidence=confidence,
        article_cited=article_cited,
        rule_summary=rule_summary,
        retrieved_docs=docs,
        judicial_verdict=judicial_verdict,
    )

    print(
        f"[STEWARD_AGENT] FINAL VERDICT: ruling={ruling}, "
        f"article_cited={article_cited}, confidence={confidence:.2f}"
    )
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run telemetry-aware steward reasoning against a FAISS FIA index."
    )
    parser.add_argument("--query", type=str, required=True, help="Steward incident query.")
    parser.add_argument(
        "--incident-json", type=str, required=True,
        help="Incident telemetry payload as a JSON string.",
    )
    parser.add_argument(
        "--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
        help="Path to persisted FAISS index.",
    )
    parser.add_argument("--k", type=int, default=6, help="Number of retrieved chunks.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = run_steward_agent(
        query=args.query,
        incident_json=args.incident_json,
        index_dir=args.index_dir,
        k=args.k,
    )
    print(json.dumps(result, indent=2))
