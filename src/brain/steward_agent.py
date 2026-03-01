"""Telemetry-aware steward agent that retrieves FIA rules and emits dashboard-ready verdicts."""

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

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from mistralai import Mistral

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / "fia_rules.index"
LEGAL_CONTEXT_PRIORITY_TERM = "Appendix L Chapter IV: Code of Driving Conduct"
TRACK_LIMITS_ARTICLE = "Article 33.3"
INCIDENTS_ARTICLE = "Article 54.3"

MISTRAL_MODEL = "mistral-small-latest"

SYSTEM_PROMPT = """You are an objective FIA Steward. You are being provided with anonymized telemetry for 'Driver A' and 'Driver B'. Ignore any historical context or driver reputation and judge solely on the 2025 Driving Standards Guidelines."""


def _get_mistral_client():
    """Get Mistral client from API key."""
    api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    return Mistral(api_key=api_key)


def _generate_llm_verdict(
    incident_data: dict[str, Any],
    features: dict[str, Any],
    retrieved_docs: list[Document],
    ruling: str,
    reasons: list[str],
) -> dict[str, Any]:
    """Generate natural language verdict using Mistral LLM with RAG."""
    try:
        client = _get_mistral_client()
    except Exception as e:
        print(f"[LLM] Could not initialize Mistral client: {e}")
        return None

    driver = incident_data.get("driver", "UNKNOWN")
    speed = incident_data.get("speed_kph", 0)
    lateral_g = incident_data.get("lateral_g", 0)
    sector = incident_data.get("sector", "UNKNOWN")
    lap = incident_data.get("lap", 0)
    incident_type = incident_data.get("incident_type", "unknown_incident")

    rules_context = (
        "\n\n".join(f"- {doc.page_content[:500]}" for doc in retrieved_docs[:3])
        if retrieved_docs
        else "No specific rules retrieved."
    )

    prompt = f"""You are an FIA Steward analyzing an F1 racing incident.

TELEMETRY DATA:
- Driver: {driver}
- Speed: {speed} km/h
- Lateral G-Force: {lateral_g}G
- Sector: {sector}
- Lap: {lap}
- Incident Type: {incident_type}

DETECTED FEATURES:
- High lateral load: {features.get("high_lateral_load", False)}
- Low clearance: {features.get("low_clearance", False)}
- Collision signal: {features.get("collision_signal", False)}
- Off-track signal: {features.get("off_track_signal", False)}

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

        llm_result = json.loads(response_text)

        return {
            "ruling": llm_result.get("ruling", ruling),
            "article_cited": llm_result.get(
                "article_cited", "General Driving Standards"
            ),
            "rule_summary": llm_result.get(
                "rule_summary", reasons[0] if reasons else "Incident analyzed."
            ),
            "judicial_verdict": f"Judicial Verdict: {llm_result.get('ruling', ruling)}. {llm_result.get('rule_summary', '')}",
            "confidence_score": 0.92,
        }

    except Exception as e:
        print(f"[LLM] Error generating verdict: {e}")
        return None


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


def _extract_visual_signals(incident_data: dict[str, Any]) -> dict[str, Any]:
    visual_blob = (
        incident_data.get("visual_evidence")
        or incident_data.get("vision_evidence")
        or incident_data.get("detector_output")
        or incident_data.get("visual_analysis")
        or incident_data.get("vision")
    )

    text_parts = [str(visual_blob)] if visual_blob is not None else []
    if incident_data.get("incident_description"):
        text_parts.append(str(incident_data.get("incident_description")))
    if incident_data.get("incident_snapshot"):
        text_parts.append(str(incident_data.get("incident_snapshot")))
    text_blob = " ".join(text_parts).lower()

    visual_clearance = None
    tires_over_line = None
    if isinstance(visual_blob, dict):
        visual_clearance = _safe_float(
            visual_blob.get("apex_clearance_meters")
            or visual_blob.get("apex_clearance")
            or visual_blob.get("apex_gap")
        )
        tires_over_line = _coerce_bool(visual_blob.get("tires_over_line"))

    if visual_clearance is None:
        match = re.search(r"apex clearance:\s*([0-9]+(?:\.[0-9]+)?)m", text_blob)
        if match:
            visual_clearance = _safe_float(match.group(1))

    if tires_over_line is None:
        if re.search(r"tires over line:\s*\[(yes|true|1)\]", text_blob):
            tires_over_line = True
        elif re.search(r"tires over line:\s*\[(no|false|0)\]", text_blob):
            tires_over_line = False

    if tires_over_line is None:
        if "over the line" in text_blob or "over line" in text_blob:
            tires_over_line = True
        elif "within track limits" in text_blob:
            tires_over_line = False

    return {
        "visual_apex_clearance": visual_clearance,
        "visual_over_line": tires_over_line is True,
        "visual_signal_present": visual_blob is not None
        or "tires over line" in text_blob,
    }


def _parse_lateral_series(incident_data: dict[str, Any]) -> list[float]:
    candidates = [
        incident_data.get("lateral_g_series"),
        incident_data.get("lateral_gs"),
        incident_data.get("telemetry", {}).get("lateral_g_series")
        if isinstance(incident_data.get("telemetry"), dict)
        else None,
    ]

    for candidate in candidates:
        if isinstance(candidate, list):
            parsed = [_safe_float(item) for item in candidate]
            return [item for item in parsed if item is not None]
    return []


def _extract_features(incident_data: dict[str, Any], query: str) -> dict[str, Any]:
    lateral_g = _safe_float(incident_data.get("lateral_g"))
    braking_force = _safe_float(incident_data.get("braking_force"))
    apex_clearance = _safe_float(
        incident_data.get("apex_clearance") or incident_data.get("apex_gap")
    )
    speed_kph = _safe_float(incident_data.get("speed_kph"))
    incident_type = str(incident_data.get("incident_type", "")).lower()
    description = str(
        incident_data.get("incident_description")
        or incident_data.get("incident_snapshot")
        or ""
    ).lower()
    query_lower = query.lower()
    visual_signals = _extract_visual_signals(incident_data)

    evasive_braking = _coerce_bool(incident_data.get("evasive_braking"))
    no_evasive_braking_flag = _coerce_bool(incident_data.get("no_evasive_braking"))
    no_evasive_braking = (
        no_evasive_braking_flag is True
        or evasive_braking is False
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

    text_blob = " ".join([incident_type, description, query_lower])
    component_failure_flag = _coerce_bool(incident_data.get("component_failure"))
    if component_failure_flag is None:
        component_failure_flag = _coerce_bool(
            incident_data.get("component_failure_flag")
        )
    if component_failure_flag is None:
        flags = incident_data.get("flags")
        if isinstance(flags, list):
            component_failure_flag = any(
                "component failure" in str(item).lower() for item in flags
            )
        elif isinstance(flags, dict):
            component_failure_flag = _coerce_bool(flags.get("component_failure"))
    if component_failure_flag is None:
        component_failure_flag = "component failure" in text_blob

    return {
        "lateral_g": lateral_g,
        "braking_force": braking_force,
        "apex_clearance": apex_clearance,
        "speed_kph": speed_kph,
        "sudden_lateral_drop": sudden_lateral_drop,
        "high_lateral_load": lateral_g is not None and lateral_g >= 4.5,
        "hard_braking": braking_force is not None and braking_force >= 0.75,
        "no_evasive_braking": no_evasive_braking,
        "low_clearance": apex_clearance is not None and apex_clearance < 2.0,
        "high_clearance": apex_clearance is not None and apex_clearance > 2.0,
        "visual_apex_clearance": visual_signals["visual_apex_clearance"],
        "visual_over_line": visual_signals["visual_over_line"],
        "visual_signal_present": visual_signals["visual_signal_present"],
        "visual_low_clearance": visual_signals["visual_apex_clearance"] is not None
        and visual_signals["visual_apex_clearance"] < 2.0,
        "visual_high_clearance": visual_signals["visual_apex_clearance"] is not None
        and visual_signals["visual_apex_clearance"] > 2.0,
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


def _extract_telemetry_year(incident_data: dict[str, Any]) -> str | None:
    telemetry = incident_data.get("telemetry")
    candidates: list[Any] = []
    if isinstance(telemetry, dict):
        candidates.extend(
            [
                telemetry.get("year"),
                telemetry.get("season"),
                telemetry.get("telemetry_year"),
            ]
        )
    candidates.extend(
        [
            incident_data.get("year"),
            incident_data.get("season"),
            incident_data.get("telemetry_year"),
        ]
    )

    for candidate in candidates:
        if candidate is None:
            continue
        maybe_year = str(candidate).strip()
        if re.fullmatch(r"(19|20)\d{2}", maybe_year):
            return maybe_year

    extracted = sorted(_extract_years(incident_data))
    return extracted[0] if extracted else None


def _is_driving_conduct_incident(features: dict[str, Any]) -> bool:
    if features.get("component_failure_flag", False):
        return False
    return True


def _build_retrieval_query(
    query: str, incident_data: dict[str, Any], features: dict[str, Any]
) -> str:
    incident_type = incident_data.get("incident_type", "unknown")

    speed = incident_data.get("speed_kph")
    lateral_g = incident_data.get("lateral_g")
    driver = incident_data.get("driver", "unknown")

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
    if driver and driver != "--":
        base_query += f" involving driver {driver}"
    if speed is not None:
        base_query += f" at {speed} km/h"
    if lateral_g is not None:
        base_query += f" with {lateral_g}G lateral force"
    if speed_reason:
        base_query += f" - {speed_reason}"
    elif proximity_reason:
        base_query += f" - {proximity_reason}"

    return base_query


def _load_vector_store(index_dir: str | Path) -> FAISS:
    index_path = Path(index_dir).resolve()
    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector index path not found: {index_path}. Build it first with vector_index.py."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if index_path.is_file() and index_path.suffix == ".index":
        metadata_path = index_path.with_name(f"{index_path.stem}_metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector index metadata file not found: {metadata_path}."
            )

        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        texts = metadata_payload.get("texts", [])
        metadatas = metadata_payload.get("metadatas", [])

        if len(texts) != len(metadatas):
            raise ValueError(
                "Vector metadata mismatch: texts and metadatas lengths do not match."
            )

        index = faiss.read_index(str(index_path))
        documents = {
            str(i): Document(page_content=texts[i], metadata=metadatas[i])
            for i in range(len(texts))
        }
        index_to_docstore_id = {i: str(i) for i in range(len(texts))}

        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(documents),
            index_to_docstore_id=index_to_docstore_id,
        )

    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _extract_years(value: Any, years: set[str] | None = None) -> set[str]:
    if years is None:
        years = set()

    if isinstance(value, dict):
        for key, subvalue in value.items():
            if str(key).lower() in {"year", "season"}:
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


def _is_technical_only_doc(doc: Any) -> bool:
    haystack = " ".join(
        [
            str(getattr(doc, "page_content", "")),
            str((getattr(doc, "metadata", {}) or {}).get("source", "")),
            str((getattr(doc, "metadata", {}) or {}).get("Document Category", "")),
        ]
    ).lower()
    blocked_terms = ["secondary roll structure", "technical regulations"]
    return any(term in haystack for term in blocked_terms)


def _is_art_33_4_doc(doc: Any) -> bool:
    haystack = " ".join(
        [
            str(getattr(doc, "page_content", "")),
            str((getattr(doc, "metadata", {}) or {}).get("source", "")),
        ]
    ).lower()
    return bool(re.search(r"(art(?:icle)?\.?\s*33\.4)", haystack, flags=re.IGNORECASE))


def _retrieve_articles(
    vector_store: FAISS,
    query: str,
    incident_data: dict[str, Any],
    features: dict[str, Any],
    k: int = 6,
) -> list[Any]:
    incident_type = str(incident_data.get("incident_type", "")).lower()
    is_conduct_incident = (
        incident_type == "driver_agnostic_incident"
        or _is_driving_conduct_incident(features)
    )

    try:
        docs_with_scores = vector_store.similarity_search_with_score(
            query, k=max(k * 4, 20)
        )
        docs = [doc for doc, _score in docs_with_scores]
    except Exception:
        docs = vector_store.similarity_search(query, k=max(k * 4, 20))

    print(f"[RAG_RETRIEVAL] Query: '{query}' | Initial docs retrieved: {len(docs)}")

    if is_conduct_incident:
        docs = [
            doc
            for doc in docs
            if "technical" not in str(doc.metadata.get("source", "")).lower()
            and "technical"
            not in str(doc.metadata.get("Document Category", "")).lower()
            and "fuel" not in str(doc.metadata.get("source", "")).lower()
            and "chassis" not in str(doc.metadata.get("source", "")).lower()
        ]

    sporting_or_driving_docs = [
        doc
        for doc in docs
        if "sporting" in str(doc.metadata.get("Document Category", "")).lower()
        or "sporting" in str(doc.metadata.get("source", "")).lower()
        or "driving" in str(doc.metadata.get("source", "")).lower()
        or "appendix" in str(doc.metadata.get("source", "")).lower()
    ]
    if sporting_or_driving_docs:
        docs = sporting_or_driving_docs

    sporting_docs = [
        doc
        for doc in docs
        if "sporting" in str(doc.metadata.get("Document Category", "")).lower()
        or "sporting" in str(doc.metadata.get("source", "")).lower()
    ]
    technical_docs = [
        doc
        for doc in docs
        if "technical" in str(doc.metadata.get("Document Category", "")).lower()
        or "technical" in str(doc.metadata.get("source", "")).lower()
    ]
    if sporting_docs and technical_docs:
        docs = sporting_docs + technical_docs
        print(
            f"[RAG_RETRIEVAL] Prioritized Sporting Regulations over Technical Regulations: {len(sporting_docs)} sporting, {len(technical_docs)} technical"
        )

    print(f"[RAG_RETRIEVAL] After filtering: {len(docs)} docs")

    if not features.get("component_failure_flag", False):
        docs = [doc for doc in docs if not _is_technical_only_doc(doc)]
    if _is_driving_conduct_incident(features):
        non_art_33_4_docs = [doc for doc in docs if not _is_art_33_4_doc(doc)]
        if non_art_33_4_docs:
            docs = non_art_33_4_docs

    return docs[:k]


def _derive_citation(doc: Any) -> str:
    metadata = doc.metadata or {}
    source = str(metadata.get("source", ""))
    content = str(getattr(doc, "page_content", ""))

    patterns = [
        r"Art(?:icle)?\.?\s*\d+(?:\.\d+)*",
        r"Appendix\s+[A-Z]",
        r"Chapter\s+[IVXLC]+",
    ]

    for pattern in patterns:
        source_match = re.search(pattern, source, flags=re.IGNORECASE)
        if source_match:
            return source_match.group(0).strip()

    article_match = re.search(r"Article\s*\d+(?:\.\d+)*", content, flags=re.IGNORECASE)
    if article_match:
        return article_match.group(0).strip()

    return source if source else "FIA Rule Reference"


def _summarize_rule(doc: Any) -> str:
    text = str(getattr(doc, "page_content", "")).replace("\n", " ").strip()
    if not text:
        return "Rule summary unavailable."

    sentences = re.split(r"[.!?]+", text)
    summary_sentences = []
    char_count = 0
    max_chars = 200

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        summary_sentences.append(sentence)
        char_count += len(sentence)
        if char_count > max_chars or len(summary_sentences) >= 2:
            break

    if not summary_sentences:
        return "Rule summary unavailable."

    return ". ".join(summary_sentences) + "."


def _detect_rule_conflict(docs: list[Any]) -> tuple[bool, list[str]]:
    article_hits: list[str] = []
    topic_tags: set[str] = set()

    for doc in docs:
        content = str(getattr(doc, "page_content", "")).lower()
        source = str((getattr(doc, "metadata", {}) or {}).get("source", "")).lower()
        haystack = f"{content} {source}"
        articles = re.findall(r"article\s*\d+(?:\.\d+)*", haystack, flags=re.IGNORECASE)
        for item in articles:
            normalized = re.sub(r"\s+", " ", item.strip().title())
            article_hits.append(normalized)
        if "track limits" in haystack or "leave the track" in haystack:
            topic_tags.add("track_limits")
        if (
            "car width" in haystack
            or "space at apex" in haystack
            or "alongside" in haystack
        ):
            topic_tags.add("leaving_space")

    unique_articles = sorted(set(article_hits))
    conflicting = len(unique_articles) >= 2 and len(topic_tags) >= 2
    return conflicting, unique_articles


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
            f"Very high lateral G-force ({features['lateral_g']:.1f}G) indicates significant cornering conflict or collision impact."
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
        reasons.append(
            "High lateral load supports a high-risk cornering conflict context."
        )
    if features["visual_low_clearance"]:
        severity += 2
        reasons.append(
            "Visual evidence indicates insufficient apex clearance relative to safe passing distance."
        )
    if features["visual_over_line"]:
        severity += 2
        reasons.append("Visual evidence shows the car over the line at the apex.")

    # Deterministic synergy bump: visual breach + no evasive braking escalates penalty severity.
    if features["visual_over_line"] and features["no_evasive_braking"]:
        severity += 2
        reasons.append(
            "Combined visual and telemetry evidence: over-the-line apex with no evasive braking."
        )

    if severity >= 4:
        ruling = "PENALTY"
    elif severity >= 2:
        ruling = "INVESTIGATION"
    else:
        ruling = "NO_FURTHER_ACTION"

    base_confidence = 0.84
    confidence = min(0.99, base_confidence + min(0.12, severity * 0.03))

    return ruling, reasons, confidence


def run_steward_agent(
    query: str,
    incident_json: str | dict[str, Any],
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    k: int = 6,
) -> dict[str, Any]:
    incident_data = _coerce_incident_json(incident_json)
    features = _extract_features(incident_data, query)
    incident_type = str(incident_data.get("incident_type", "")).lower()

    print(f"[STEWARD_AGENT] Processing incident_type: {incident_type}")

    HIGH_PRIORITY_INCIDENT_TYPES = {
        "high_g_event",
        "collision",
        "contact",
        "off_track",
        "forced_wide",
        "driver_agnostic_incident",
    }

    if incident_type == "normal_telemetry":
        ruling = "No Action Required"
        confidence = 0.98
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "id": incident_data.get("id", "live-incident"),
            "sessionName": incident_data.get("sessionName")
            or incident_data.get("track")
            or "Race Control",
            "track": incident_data.get("track"),
            "timestamp": incident_data.get("timestamp") or now_iso,
            "lastUpdated": now_iso,
            "driver": incident_data.get("driver", "--"),
            "incident_type": incident_type,
            "incident_description": incident_data.get("incident_description")
            or incident_data.get("incident_snapshot")
            or query,
            "speed_kph": incident_data.get("speed_kph"),
            "delta_to_leader": incident_data.get("delta_to_leader")
            or incident_data.get("apex_gap"),
            "track_temp_c": incident_data.get("track_temp_c"),
            "sector": incident_data.get("sector", "N/A"),
            "lap": incident_data.get("lap", 0),
            "article_cited": None,
            "rule_summary": "No rule violation detected in standard telemetry data.",
            "ruling": ruling,
            "verdict": ruling,
            "confidence_score": confidence,
            "judicial_verdict": f"Judicial Verdict: {ruling}. Telemetry shows normal racing parameters with no incident indicators.",
            "retrieved_articles": [],
            "query": query,
            "system_prompt": SYSTEM_PROMPT,
        }

    is_high_priority = incident_type in HIGH_PRIORITY_INCIDENT_TYPES or any(
        trigger in incident_type for trigger in HIGH_PRIORITY_INCIDENT_TYPES
    )

    if not is_high_priority:
        ruling = "No Action Required"
        confidence = 0.95
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "id": incident_data.get("id", "live-incident"),
            "sessionName": incident_data.get("sessionName")
            or incident_data.get("track")
            or "Race Control",
            "track": incident_data.get("track"),
            "timestamp": incident_data.get("timestamp") or now_iso,
            "lastUpdated": now_iso,
            "driver": incident_data.get("driver", "--"),
            "incident_type": incident_type,
            "incident_description": incident_data.get("incident_description")
            or incident_data.get("incident_snapshot")
            or query,
            "speed_kph": incident_data.get("speed_kph"),
            "delta_to_leader": incident_data.get("delta_to_leader")
            or incident_data.get("apex_gap"),
            "track_temp_c": incident_data.get("track_temp_c"),
            "sector": incident_data.get("sector", "N/A"),
            "lap": incident_data.get("lap", 0),
            "article_cited": None,
            "rule_summary": "No significant incident detected in telemetry data.",
            "ruling": ruling,
            "verdict": ruling,
            "confidence_score": confidence,
            "judicial_verdict": f"Judicial Verdict: {ruling}. No high-priority incident triggers detected in telemetry.",
            "retrieved_articles": [],
            "query": query,
            "system_prompt": SYSTEM_PROMPT,
        }

    vector_store = _load_vector_store(index_dir=index_dir)
    print(f"Link established with {len(vector_store.docstore._dict)} rule chunks")
    augmented_query = _build_retrieval_query(query, incident_data, features)
    print(f"[RETRIEVAL_QUERY] '{augmented_query}'")
    docs = _retrieve_articles(
        vector_store=vector_store,
        query=augmented_query,
        incident_data=incident_data,
        features=features,
        k=k,
    )

    if docs:
        print(f"[RETRIEVED] {len(docs)} docs")
        for i, doc in enumerate(docs[:3]):
            src = doc.metadata.get("source", "unknown")
            print(f"  Doc {i + 1}: {src[:60]}...")
    else:
        print("[RETRIEVED] No docs returned from RAG")

    if not docs:
        ruling, reasons, confidence = _decide_verdict(features)

        lateral_g = features.get("lateral_g", 0)
        is_collision = lateral_g and lateral_g >= 3.75

        llm_result = _generate_llm_verdict(
            incident_data=incident_data,
            features=features,
            retrieved_docs=[],
            ruling=ruling,
            reasons=reasons,
        )

        if llm_result:
            ruling = llm_result.get("ruling", ruling)
            reasons = [
                llm_result.get(
                    "rule_summary", reasons[0] if reasons else "Incident analyzed."
                )
            ]
            confidence = llm_result.get("confidence_score", confidence)
            article_cited = llm_result.get("article_cited", "General Driving Standards")
            rule_summary = llm_result.get("rule_summary", "Incident analyzed.")
            print(f"[LLM] Fallback path - LLM verdict: ruling={ruling}")
        else:
            article_cited = "General Driving Standards"
            rule_summary_text = (
                reasons[0] if reasons else "High G-force event detected."
            )
            rule_summary = rule_summary_text

        fallback_payload = {
            "id": incident_data.get("id", "live-incident"),
            "sessionName": incident_data.get("sessionName")
            or incident_data.get("track")
            or "Race Control",
            "track": incident_data.get("track"),
            "timestamp": incident_data.get("timestamp")
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "lastUpdated": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "driver": incident_data.get("driver", "--"),
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
            "rule_summary": f"Incident flagged. {rule_summary}",
            "ruling": ruling,
            "verdict": ruling,
            "confidence_score": confidence,
            "judicial_verdict": f"Judicial Verdict: {ruling}. Telemetry analysis: {'; '.join(reasons) if reasons else 'No specific violations detected.'}",
            "retrieved_articles": [],
            "query": query,
            "system_prompt": SYSTEM_PROMPT,
        }
        print(
            f"[STEWARD_AGENT] No articles retrieved, using fallback verdict: {fallback_payload['ruling']}"
        )
        return fallback_payload

    top_doc = docs[0]
    citation = _derive_citation(top_doc)
    rule_summary = _summarize_rule(top_doc)

    ruling, reasons, confidence = _decide_verdict(features)

    llm_result = _generate_llm_verdict(
        incident_data=incident_data,
        features=features,
        retrieved_docs=docs,
        ruling=ruling,
        reasons=reasons,
    )

    if llm_result:
        ruling = llm_result.get("ruling", ruling)
        reasons = [
            llm_result.get(
                "rule_summary", reasons[0] if reasons else "Incident analyzed."
            )
        ]
        confidence = llm_result.get("confidence_score", confidence)
        article_cited = llm_result.get(
            "article_cited", citation if docs else "General Driving Standards"
        )
        rule_summary = llm_result.get("rule_summary", rule_summary)
        print(f"[LLM] LLM verdict applied: ruling={ruling}, article={article_cited}")
    else:
        article_cited = citation if docs else "General Driving Standards"

    if (
        article_cited
        and article_cited != "FIA Rule Reference"
        and features.get("high_lateral_load", False)
    ):
        ruling = "PENALTY"
        reasons.append(
            f"Citation found ({article_cited}) with high lateral G-force - automatic penalty threshold."
        )
        confidence = max(confidence, 0.85)

    conflict_detected, conflicting_articles = _detect_rule_conflict(docs)

    if conflict_detected:
        reasons.append(
            f"Retrieved rule set includes potentially conflicting guidance ({', '.join(conflicting_articles[:4])})."
        )
        confidence = max(0.0, confidence - 0.18)
        if ruling == "PENALTY":
            ruling = "PENDING FURTHER DATA"
            reasons.append(
                "Penalty decision deferred pending additional telemetry/video alignment due to rule conflict."
            )

    doc_years = [doc.metadata.get("Year") for doc in docs if doc.metadata.get("Year")]
    years_found = sorted(set(str(y) for y in doc_years if y))
    if "2025" in years_found:
        older_years = [y for y in years_found if y != "2025"]
        if older_years:
            reasons.append(
                f"Selected 2025 regulations over {', '.join(older_years)} as the primary legal basis due to current season applicability and updated driving conduct standards in Appendix L Chapter IV."
            )
            confidence = min(0.99, confidence + 0.06)
        else:
            reasons.append(
                "Applied 2025 FIA Sporting Regulations as the primary legal basis for this incident."
            )
            confidence = min(0.99, confidence + 0.03)
    elif years_found:
        reasons.append(
            f"Applied {max(years_found)} regulations retrieved from the index for this incident analysis."
        )

    if (
        not conflict_detected
        and article_cited != "FIA Rule Reference"
        and rule_summary != "Rule summary unavailable."
    ):
        confidence = max(confidence, 0.91)

    evidence_lines = [
        f"Retrieved citation: {article_cited}",
        f"Rule text: {rule_summary}",
    ]
    evidence_lines.extend(reasons)

    judicial_verdict = (
        f"Judicial Verdict: {ruling}. "
        f"Telemetry-to-rule reasoning: {' '.join(evidence_lines)} "
        f"Contextual legal focus considered between {TRACK_LIMITS_ARTICLE} and driving conduct regulations."
    )

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    ruling_reason = reasons[0] if reasons else "Telemetry analysis complete"

    driver_a = incident_data.get("driver", "UNKNOWN")
    speed = incident_data.get("speed_kph", 0)
    lateral_g = features.get("lateral_g", 0)
    incident_detail = (
        f"Driver: {driver_a} | Speed: {speed} km/h | Lateral G: {lateral_g:.2f}G"
        if driver_a != "--"
        else f"Speed: {speed} km/h | Lateral G: {lateral_g:.2f}G"
    )

    formatted_rule_summary = (
        f"[{article_cited}] {rule_summary} | {incident_detail} | {ruling_reason}"
    )

    dashboard_payload = {
        "id": incident_data.get("id", "live-incident"),
        "sessionName": incident_data.get("sessionName")
        or incident_data.get("track")
        or "Race Control",
        "track": incident_data.get("track"),
        "timestamp": incident_data.get("timestamp") or now_iso,
        "lastUpdated": now_iso,
        "driver": incident_data.get("driver", "--"),
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
        "rule_summary": formatted_rule_summary,
        "ruling": ruling,
        "verdict": ruling,
        "confidence_score": round(confidence, 2),
        "judicial_verdict": judicial_verdict,
        "retrieved_articles": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "year": doc.metadata.get("Year", "unknown"),
                "document_category": doc.metadata.get("Document Category", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            }
            for doc in docs
        ],
        "query": query,
        "system_prompt": SYSTEM_PROMPT,
    }

    print(
        f"[STEWARD_AGENT] FINAL VERDICT: ruling={ruling}, article_cited={citation}, confidence={confidence:.2f}"
    )
    print(f"[STEWARD_AGENT] verdict_payload: {json.dumps(dashboard_payload, indent=2)}")

    return dashboard_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run telemetry-aware steward reasoning against a FAISS FIA index."
    )
    parser.add_argument(
        "--query", type=str, required=True, help="Steward incident query."
    )
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
