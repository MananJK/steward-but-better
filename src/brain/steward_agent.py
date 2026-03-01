"""Telemetry-aware steward agent that retrieves FIA rules and emits dashboard-ready verdicts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / "fia_rules.index"
LEGAL_CONTEXT_PRIORITY_TERM = "Appendix L Chapter IV: Code of Driving Conduct"
TRACK_LIMITS_ARTICLE = "Article 33.3"
LEAVING_SPACE_ARTICLE = "Article 33.4"
INCIDENTS_ARTICLE = "Article 54.3"
YEAR_2025_QUERY_PREFIX = (
    "Using the 2025 FIA Sporting Regulations, identify the specific article for driving conduct."
)
DRIVING_CONDUCT_NEGATIVE_CONSTRAINT = (
    "Do not cite technical survival cell rules (Art 33.4) for driving conduct incidents; "
    "prioritize Appendix L, Chapter IV and Article 54.3."
)


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
        "visual_signal_present": visual_blob is not None or "tires over line" in text_blob,
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
        component_failure_flag = _coerce_bool(incident_data.get("component_failure_flag"))
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


def _build_retrieval_query(query: str, incident_data: dict[str, Any], features: dict[str, Any]) -> str:
    keywords: list[str] = []
    reasoned_focus_terms: list[str] = []
    query_prefix = ""
    telemetry_year = _extract_telemetry_year(incident_data)
    driving_conduct_incident = _is_driving_conduct_incident(features)

    if telemetry_year == "2025":
        query_prefix = f"{YEAR_2025_QUERY_PREFIX}\n"
    if driving_conduct_incident and telemetry_year == "2025":
        reasoned_focus_terms.append(DRIVING_CONDUCT_NEGATIVE_CONSTRAINT)
        keywords.extend([LEGAL_CONTEXT_PRIORITY_TERM, INCIDENTS_ARTICLE])
    elif driving_conduct_incident:
        keywords.extend([LEGAL_CONTEXT_PRIORITY_TERM, INCIDENTS_ARTICLE])

    prefers_track_limits = bool(
        features["high_lateral_load"]
        and (features["high_clearance"] or features["visual_high_clearance"])
    )

    if features["collision_signal"]:
        keywords.extend(["collision", "causing a collision", "avoidable contact"])
    if features["off_track_signal"]:
        keywords.extend(["leaving the track", "forcing a car off track"])
    if features["low_clearance"] and not prefers_track_limits:
        keywords.extend(["overtaking", "car width", "space at apex"])
    if features["hard_braking"]:
        keywords.extend(["unsafe maneuver", "dangerous driving", "late braking"])
    if features["visual_over_line"] or prefers_track_limits:
        keywords.extend(["track limits", TRACK_LIMITS_ARTICLE, "over the line at apex"])
        reasoned_focus_terms.extend(
            [
                "contextual focus: high lateral_g with apex clearance >2.0m",
                f"prefer {TRACK_LIMITS_ARTICLE} track limits over {LEAVING_SPACE_ARTICLE} leaving space",
            ]
        )
    elif features["low_clearance"] or features["visual_low_clearance"]:
        keywords.extend(["leaving space", LEAVING_SPACE_ARTICLE, "space at apex"])

    lateral_g = features.get("lateral_g")
    if lateral_g is not None:
        keywords.append(f"lateral {lateral_g:.2f}G")

    incident_type = incident_data.get("incident_type")
    if incident_type:
        keywords.append(str(incident_type))
    if features.get("incident_type") == "high_g_event":
        keywords.append(LEGAL_CONTEXT_PRIORITY_TERM)
    keywords.extend(reasoned_focus_terms)

    suffix = " | ".join(dict.fromkeys(keywords))
    return (
        f"{query_prefix}{query}\n"
        f"Telemetry context: {json.dumps(incident_data, sort_keys=True)}\n"
        f"Priority terms: {suffix}"
    ).strip()


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


def _is_survival_cell_rule_doc(doc: Any) -> bool:
    haystack = " ".join(
        [
            str(getattr(doc, "page_content", "")),
            str((getattr(doc, "metadata", {}) or {}).get("source", "")),
            str((getattr(doc, "metadata", {}) or {}).get("Document Category", "")),
        ]
    ).lower()
    return "survival cell" in haystack and (
        "technical regulations" in haystack or "article 12" in haystack or "article 13" in haystack
    )


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
    try:
        docs_with_scores = vector_store.similarity_search_with_score(query, k=max(k * 4, 20))
        docs = [doc for doc, _score in docs_with_scores]
    except Exception:
        docs = vector_store.similarity_search(query, k=max(k * 4, 20))

    telemetry_year = _extract_telemetry_year(incident_data)
    if telemetry_year:
        telemetry_year_filtered = [
            doc for doc in docs if str(doc.metadata.get("Year", "unknown")) == telemetry_year
        ]
        if telemetry_year_filtered:
            docs = telemetry_year_filtered

    year_hints = _extract_years(incident_data)
    if year_hints:
        year_filtered = [
            doc for doc in docs if str(doc.metadata.get("Year", "unknown")) in year_hints
        ]
        if year_filtered:
            docs = year_filtered

    if not features.get("component_failure_flag", False):
        docs = [doc for doc in docs if not _is_technical_only_doc(doc)]
    if _is_driving_conduct_incident(features):
        docs = [doc for doc in docs if not _is_survival_cell_rule_doc(doc)]
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

    sentence_end = re.search(r"[.!?]", text)
    if sentence_end and sentence_end.start() > 80:
        summary = text[: sentence_end.start() + 1]
    else:
        summary = text[:320]

    return summary.strip()


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
        if "car width" in haystack or "space at apex" in haystack or "alongside" in haystack:
            topic_tags.add("leaving_space")

    unique_articles = sorted(set(article_hits))
    conflicting = len(unique_articles) >= 2 and len(topic_tags) >= 2
    return conflicting, unique_articles


def _decide_verdict(features: dict[str, Any]) -> tuple[str, list[str], float]:
    severity = 0
    reasons: list[str] = []

    if features["collision_signal"]:
        severity += 2
        reasons.append("Telemetry and incident descriptors indicate potential contact/collision.")
    if features["off_track_signal"]:
        severity += 2
        reasons.append("Signals suggest a possible leaving-the-track or forcing-wide event.")
    if features["low_clearance"]:
        severity += 2
        reasons.append("Apex clearance is below one car width threshold (2.0 m).")
    if features["hard_braking"]:
        severity += 1
        reasons.append("Braking force indicates an aggressive braking phase during the incident window.")
    if features["high_lateral_load"]:
        severity += 1
        reasons.append("High lateral load supports a high-risk cornering conflict context.")
    if features["visual_low_clearance"]:
        severity += 2
        reasons.append("Visual evidence indicates insufficient apex clearance relative to Article 33.4.")
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

    vector_store = _load_vector_store(index_dir=index_dir)
    print("Link established with 3,725 rule chunks")
    augmented_query = _build_retrieval_query(query, incident_data, features)
    docs = _retrieve_articles(
        vector_store=vector_store,
        query=augmented_query,
        incident_data=incident_data,
        features=features,
        k=k,
    )

    if not docs:
        raise ValueError("No relevant FIA articles were retrieved from the index.")

    top_doc = docs[0]
    citation = _derive_citation(top_doc)
    rule_summary = _summarize_rule(top_doc)

    ruling, reasons, confidence = _decide_verdict(features)
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

    if (
        not conflict_detected
        and citation != "FIA Rule Reference"
        and rule_summary != "Rule summary unavailable."
    ):
        confidence = max(confidence, 0.91)

    evidence_lines = [
        f"Retrieved citation: {citation}",
        f"Rule text: {rule_summary}",
    ]
    evidence_lines.extend(reasons)

    judicial_verdict = (
        f"Judicial Verdict: {ruling}. "
        f"Telemetry-to-rule reasoning: {' '.join(evidence_lines)} "
        f"Contextual legal focus considered between {TRACK_LIMITS_ARTICLE} and {LEAVING_SPACE_ARTICLE}."
    )

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    dashboard_payload = {
        "id": incident_data.get("id", "live-incident"),
        "sessionName": incident_data.get("sessionName") or incident_data.get("track") or "Race Control",
        "track": incident_data.get("track"),
        "timestamp": incident_data.get("timestamp") or now_iso,
        "lastUpdated": now_iso,
        "driver": incident_data.get("driver", "--"),
        "incident_type": incident_data.get("incident_type", "incident_review"),
        "incident_description": incident_data.get("incident_description") or incident_data.get("incident_snapshot") or query,
        "speed_kph": incident_data.get("speed_kph"),
        "delta_to_leader": incident_data.get("delta_to_leader") or incident_data.get("apex_gap"),
        "track_temp_c": incident_data.get("track_temp_c"),
        "sector": incident_data.get("sector", "N/A"),
        "lap": incident_data.get("lap", 0),
        "article_cited": citation,
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
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            }
            for doc in docs
        ],
        "query": query,
    }

    return dashboard_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run telemetry-aware steward reasoning against a FAISS FIA index."
    )
    parser.add_argument("--query", type=str, required=True, help="Steward incident query.")
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
