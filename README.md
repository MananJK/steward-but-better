# STEWARD BUT BETTER

An AI race-steward prototype: it replays real F1 telemetry, detects potential
incidents from physics, retrieves the applicable FIA rules with RAG, and shows
verdicts with citations on a live dashboard.

> **History:** this is the post-hackathon rebuild. The original hackathon
> submission is preserved verbatim on the `OG/hackathon-v1` branch. The rebuild
> replaced fabricated data paths (synthetic G-forces, a truncated vector index,
> hard-coded confidence scores) with real, validated ones — see
> *What is real* below.

---

## How it works

```
FastF1 (real historical telemetry, cached)
        │
        ▼
live_simulator.py            replays a lap at 1 packet/s, POSTs to the dashboard
  ├─ telemetry_utils.py        real G-force physics (v·yaw-rate from X/Y track positions)
  └─ driver_agnostic_detector  proximity / anomaly triggers, anonymized driver labels
        │
        ▼
Next.js API  /api/telemetry   normalizes packets, owns dashboard state
  ├─ brain service (FastAPI)  long-lived; loads the FAISS index once
  │    └─ steward_agent.py    deterministic verdict + optional Mistral LLM wording
  └─ data/*.json              locked, atomic state (NOT in public/)
        │
        ▼
Dashboard                    driver grid, verdict cards with citations, inquiry log
```

### The brain (RAG over the FIA rulebook)

- 3,725 chunks covering the 2021–2025 Sporting Regulations, 2025 Technical
  Regulations, and the 2025 Driving Standards Guidelines (OCR'd from the FIA
  PDFs via `src/ingestion/ocr_processor.py`).
- Embedded locally with `sentence-transformers/all-MiniLM-L6-v2`; index,
  metadata, and manifest are written as one **validated, atomic set**
  (`vector count == chunk count`, checked on every load).
- Retrieval filters by metadata (document category, year — current season
  preferred) instead of string heuristics, and citations come from
  article/section numbers extracted into chunk metadata.

### The telemetry physics

- Lateral G is computed from the car's actual trajectory: speed magnitude from
  the Speed channel, yaw rate from smoothed X/Y positions
  (`a_lat = v · ω`). Verified: ~5.20G measured vs 5.24G theoretical on a
  synthetic 200 km/h, 60 m-radius circle; realistic 0.9–4.6G range on real
  Yas Marina laps.
- FastF1 position data samples at ~4 Hz, so G values are smoothed estimates,
  not 250 Hz car telemetry.
- Crash-class triggers require sustained ≥5G **plus** a ≥50 km/h speed drop
  between packets — normal cornering (4–6G peaks) does not flag an incident.
- When position channels are missing, lateral G is honestly reported as
  unavailable rather than invented.

## What is real vs. heuristic

| Piece | Status |
|---|---|
| Telemetry source | Real (FastF1, official live-timing data, cached locally) |
| G-forces | Real, computed from track positions (4 Hz, smoothed) |
| Rule retrieval | Real FAISS RAG over the actual FIA rulebooks |
| Verdict ruling | Deterministic evidence-scoring rules (see `_decide_verdict`) |
| Verdict wording | Optional Mistral LLM (`MISTRAL_API_KEY`); system degrades gracefully without it |
| Confidence score | Heuristic evidence-strength score — **not** a calibrated probability |
| Incident detection thresholds | Hand-tuned (5G, 50 km/h drop, 30% proximity change) |
| Simulator cadence | 1 packet/second replay of a historical lap |

## Getting started

```bash
python -m pip install -r requirements.txt

# 1. Brain service (loads the FAISS index once; port 8000)
cd src/brain && python -m uvicorn server:app --port 8000

# 2. Dashboard (port 3000)
cd src/ui && npm install && npm run dev

# 3. Telemetry replay (Abu Dhabi 2021, lap 58)
cd src/telemetry && python live_simulator.py --year 2021 --gp "Abu Dhabi Grand Prix" --start-lap 58
```

Optional: set `MISTRAL_API_KEY` in `.env` for LLM-worded verdicts. The
deterministic path needs no API keys.

Useful brain utilities:

```bash
python src/brain/vector_index.py --validate          # check index/metadata consistency
python src/brain/vector_index.py --search "leaving the track and gaining a lasting advantage"
python src/brain/vector_index.py --rebuild-from-metadata src/brain/fia_rules_metadata.json
```

## Tests

```bash
python -m pytest tests/ -q
```

Covers the G-force physics (synthetic trajectories), article-aware chunking,
metadata enrichment, verdict thresholds and payload shape, index-mismatch
rejection, proximity-trigger semantics, and the evaluator's geometry and
dive-bomb direction.

## Known limitations

- One lap at a time: the simulator replays a single lap per driver; gaps and
  positions across laps are approximated with a fixed lap length.
- Detection thresholds are hand-tuned against the 2021 Abu Dhabi sample, not
  validated across races.
- The steward agent's severity scoring is transparent but simplistic; a
  production system would calibrate it against historical stewarding decisions.
- The dashboard state store is a locked JSON file set — fine for a single
  demo instance, not for multiple dashboard processes.

## License

MIT.
