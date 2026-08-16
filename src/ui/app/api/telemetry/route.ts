import { NextResponse } from "next/server";

import {
  STATE_FILES,
  isFileLockError,
  mutateJson,
  readJson,
  writeJson,
} from "@/app/lib/state-store";
import type { ActiveInvestigation } from "@/app/types/incident";

export const runtime = "nodejs";

const BRAIN_SERVICE_URL =
  process.env.BRAIN_SERVICE_URL ?? "http://127.0.0.1:8000";
const BRAIN_TIMEOUT_MS = Number(process.env.BRAIN_TIMEOUT_MS ?? 25_000);
const MAX_INVESTIGATIONS = 5;

type DriverTelemetry = {
  driver_code?: string;
  driver_number?: string;
  position_rank?: number;
  lap_number?: number;
  current_speed?: number;
  distance_offset?: number;
  lateral_g?: number;
  sector?: string;
  delta_to_leader?: number | null;
  incident_detected?: boolean;
  status?: string;
};

type SimulatorPacket = {
  driver?: string;
  driver_a?: string;
  driver_b?: string;
  speed?: number;
  lateral_g?: number;
  longitudinal_g?: number;
  session_status?: string;
  distance_to_apex?: number | null;
  trigger_steward?: boolean;
  lap?: number;
  sector?: string;
  delta_to_leader?: number;
  all_drivers?: DriverTelemetry[];
  track?: string;
  timestamp?: string | number;
  agnostic_incident?: Record<string, unknown>;
};

type IncidentPayload = Record<string, unknown> & {
  id: string;
  verdict?: string;
  ruling?: string;
  confidence_score?: number | null;
  lateral_g?: number;
  article_cited?: string | null;
  rule_summary?: string;
};

function toIsoTimestamp(value: unknown): string {
  if (typeof value === "string") {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString();
    }
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    // Numeric values are session-relative seconds; keep them relative to the
    // session wall clock is not knowable here, so record ingest time instead
    // of fabricating an offset-based timestamp.
    return new Date().toISOString();
  }
  return new Date().toISOString();
}

function buildBaseIncident(packet: SimulatorPacket): IncidentPayload {
  const speedKph = Number(packet.speed ?? 0);
  const apexGap = packet.distance_to_apex ?? null;
  const lateralG = Number(packet.lateral_g ?? 0);
  const brakingForce = Number(packet.longitudinal_g ?? 0);

  const triggerSteward = packet.trigger_steward === true;
  const isSessionFinished =
    String(packet.session_status ?? "").toUpperCase() === "FINISHED";
  const sessionName = isSessionFinished
    ? "POST-RACE SCRUTINEERING"
    : packet.track ?? "Live Simulation";
  const agnosticIncident = packet.agnostic_incident ?? null;

  return {
    id: `incident-${Date.now()}`,
    sessionName,
    track: packet.track ?? "Live Simulation",
    session_status: String(packet.session_status ?? "").toUpperCase() || "UNKNOWN",
    driver: packet.driver ?? "--",
    driver_a: packet.driver_a ?? null,
    driver_b: packet.driver_b ?? null,
    lap: Number(packet.lap ?? 0),
    sector: typeof packet.sector === "string" ? packet.sector.toUpperCase() : undefined,
    timestamp: toIsoTimestamp(packet.timestamp),
    lastUpdated: new Date().toISOString(),
    speed_kph: speedKph,
    apex_gap: apexGap,
    apex_clearance: apexGap,
    lateral_g: lateralG,
    braking_force: brakingForce,
    delta_to_leader:
      typeof packet.delta_to_leader === "number" &&
      Number.isFinite(packet.delta_to_leader)
        ? packet.delta_to_leader
        : undefined,
    all_drivers: Array.isArray(packet.all_drivers) ? packet.all_drivers : [],
    agnostic_incident: agnosticIncident,
    incident_type: triggerSteward
      ? agnosticIncident
        ? "driver_agnostic_incident"
        : "high_g_event"
      : "normal_telemetry",
    incident_snapshot: agnosticIncident
      ? `Driver-agnostic incident: ${JSON.stringify(agnosticIncident)}`
      : `Car ${packet.driver ?? "--"} telemetry; speed ${speedKph.toFixed(1)} km/h; ` +
        `lateral load ${lateralG.toFixed(2)}G; distance to apex ${apexGap ?? "N/A"}m.`,
    article_cited: null,
    rule_summary: triggerSteward
      ? "Steward review triggered by telemetry threshold."
      : "No rule violation detected in standard telemetry data.",
    verdict: triggerSteward ? "INVESTIGATION" : "NO_FURTHER_ACTION",
    ruling: triggerSteward ? "INVESTIGATION" : "NO_FURTHER_ACTION",
    confidence_score: null,
    trigger_steward: triggerSteward,
  };
}

/**
 * Ask the long-lived Python brain service for a verdict. The service loads
 * the FAISS index once; this is a plain HTTP call with a real timeout, so a
 * hung brain can never leak processes the way the old subprocess spawn did.
 */
async function requestVerdict(
  incident: IncidentPayload
): Promise<IncidentPayload | null> {
  const query =
    typeof incident.incident_snapshot === "string" &&
    incident.incident_snapshot.trim().length > 0
      ? incident.incident_snapshot
      : "Review this telemetry incident for FIA compliance.";

  try {
    const response = await fetch(`${BRAIN_SERVICE_URL}/verdict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, incident, k: 6 }),
      signal: AbortSignal.timeout(BRAIN_TIMEOUT_MS),
    });

    if (!response.ok) {
      console.error(
        `[TELEMETRY_ROUTE] Brain service returned ${response.status}: ${await response.text()}`
      );
      return null;
    }
    return (await response.json()) as IncidentPayload;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`[TELEMETRY_ROUTE] Brain service unreachable: ${message}`);
    return null;
  }
}

function brainUnavailableFallback(incident: IncidentPayload): IncidentPayload {
  return {
    ...incident,
    ruling: "INVESTIGATION",
    verdict: "INVESTIGATION",
    confidence_score: null,
    article_cited: null,
    rule_summary:
      "Steward brain service unavailable — incident queued for manual review.",
  };
}

function cleanRuleSummary(text: unknown): string {
  if (!text || typeof text !== "string") return "No summary available";
  const cleaned = text.replace(/\[rules\/[^\]]+\]/g, "").replace(/\*\*/g, "").trim();
  return cleaned || "No summary available";
}

function cleanArticleCited(text: unknown): string {
  if (!text || typeof text !== "string") return "No article cited";
  const match = text.match(/rules\/[^/]+\/(.+?)\.md/);
  if (match) return match[1].replace(/_/g, " ");
  return text.replace(/^rules\//, "").replace(/_/g, " ") || "No article cited";
}

function toGLoad(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.abs(parsed) : 0;
}

function toInvestigationEntry(payload: IncidentPayload): ActiveInvestigation {
  const confidence = Number(payload.confidence_score ?? 0);
  const confidenceScore = confidence <= 1 ? Math.round(confidence * 100) : Math.round(confidence);

  return {
    id: String(payload.id ?? `incident-${Date.now()}`),
    timestamp: String(payload.timestamp ?? new Date().toISOString()),
    driver: String(payload.driver ?? "--"),
    lap: Number(payload.lap ?? 0),
    incident_type: String(payload.incident_type ?? "unknown_incident"),
    incident_description: String(
      payload.incident_description ?? payload.incident_snapshot ?? "No details."
    ),
    speed_kph: Number(payload.speed_kph ?? 0),
    lateral_g: Number(payload.lateral_g ?? 0),
    rule_summary: cleanRuleSummary(payload.rule_summary),
    ruling: String(payload.ruling ?? payload.verdict ?? "INVESTIGATION"),
    confidence_score: Math.max(0, Math.min(100, confidenceScore)),
    article_cited: cleanArticleCited(payload.article_cited),
    driver_a: payload.driver_a ? String(payload.driver_a) : undefined,
    driver_b: payload.driver_b ? String(payload.driver_b) : undefined,
  };
}

/** Keep the newest investigations; when full, drop the lowest-severity (G) one. */
function insertInvestigation(
  investigations: ActiveInvestigation[],
  entry: ActiveInvestigation
): ActiveInvestigation[] {
  if (investigations.length < MAX_INVESTIGATIONS) {
    return [entry, ...investigations];
  }

  let minLoadIndex = 0;
  let minLoad = toGLoad(investigations[0]?.lateral_g);
  for (let index = 1; index < investigations.length; index += 1) {
    const candidateLoad = toGLoad(investigations[index]?.lateral_g);
    if (candidateLoad <= minLoad) {
      minLoad = candidateLoad;
      minLoadIndex = index;
    }
  }

  if (toGLoad(entry.lateral_g) > minLoad) {
    const next = [...investigations];
    next.splice(minLoadIndex, 1);
    return [entry, ...next];
  }
  return investigations;
}

export async function GET() {
  const [investigations, live] = await Promise.all([
    readJson<ActiveInvestigation[]>(STATE_FILES.investigations, []),
    readJson<IncidentPayload | null>(STATE_FILES.live, null),
  ]);
  return NextResponse.json(
    { investigations: Array.isArray(investigations) ? investigations : [], live },
    { status: 200 }
  );
}

export async function POST(request: Request) {
  let packet: SimulatorPacket;
  try {
    packet = (await request.json()) as SimulatorPacket;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid JSON body";
    return NextResponse.json({ ok: false, error: message }, { status: 400 });
  }

  try {
    const baseIncident = buildBaseIncident(packet);
    const isSessionFinished =
      String(packet.session_status ?? "").toUpperCase() === "FINISHED";

    if (isSessionFinished) {
      await writeJson(STATE_FILES.inquiry, {
        manual_clear_required: false,
        dismissed: true,
        updated_at: new Date().toISOString(),
        inquiry: null,
      });
      await writeJson(STATE_FILES.investigations, []);
      await writeJson(STATE_FILES.live, { ...baseIncident, trigger_steward: false });

      return NextResponse.json(
        { ok: true, trigger_steward: false, session_status: "FINISHED" },
        { status: 200 }
      );
    }

    if (packet.trigger_steward === true) {
      const verdict = await requestVerdict(baseIncident);
      const resolved = verdict ?? brainUnavailableFallback(baseIncident);

      const merged: IncidentPayload = {
        ...baseIncident,
        ...resolved,
        rule_summary: cleanRuleSummary(resolved.rule_summary),
        article_cited: resolved.article_cited
          ? cleanArticleCited(resolved.article_cited)
          : null,
        all_drivers: baseIncident.all_drivers,
      };

      await mutateJson(STATE_FILES.inquiry, null, () => ({
        manual_clear_required: true,
        dismissed: false,
        updated_at: new Date().toISOString(),
        inquiry: merged,
      }));
      await mutateJson<ActiveInvestigation[]>(STATE_FILES.investigations, [], (list) =>
        insertInvestigation(Array.isArray(list) ? list : [], toInvestigationEntry(merged))
      );
      await writeJson(STATE_FILES.live, { ...merged, trigger_steward: false });

      return NextResponse.json(
        {
          ok: true,
          trigger_steward: true,
          verdict: merged.ruling ?? merged.verdict ?? null,
          confidence_score: merged.confidence_score ?? null,
        },
        { status: 200 }
      );
    }

    await writeJson(STATE_FILES.live, baseIncident);
    return NextResponse.json(
      {
        ok: true,
        trigger_steward: false,
        verdict: baseIncident.ruling ?? baseIncident.verdict ?? null,
      },
      { status: 200 }
    );
  } catch (error) {
    if (isFileLockError(error)) {
      return NextResponse.json(
        {
          ok: true,
          accepted: true,
          message: "Telemetry accepted but deferred because a state file is temporarily locked.",
        },
        { status: 202 }
      );
    }

    const message = error instanceof Error ? error.message : "Unknown telemetry route error";
    console.error("[TELEMETRY_ROUTE] Error:", message);
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
