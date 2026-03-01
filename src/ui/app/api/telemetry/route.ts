import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

import { NextResponse } from "next/server";

export const runtime = "nodejs";

const execFileAsync = promisify(execFile);

type SimulatorPacket = {
  driver?: string;
  speed?: number;
  lateral_g?: number;
  longitudinal_g?: number;
  session_status?: string;
  distance_to_apex?: number | null;
  trigger_steward?: boolean;
  lap?: number;
  sector?: string;
  delta_to_leader?: number;
  all_drivers?: Array<{
    driver_code: string;
    driver_number: string;
    position_rank: number;
    lap_number: number;
    current_speed: number;
    distance_offset: number;
    lateral_g: number;
    sector: string;
    delta_to_leader?: number | null;
    incident_detected?: boolean;
  }>;
  track?: string;
  timestamp?: string | number;
  agnostic_incident?: Record<string, unknown>;
};

type IncidentPayload = Record<string, unknown> & {
  verdict?: string;
  ruling?: string;
  confidence_score?: number;
};

type CurrentInquiryRecord = {
  manual_clear_required: boolean;
  dismissed: boolean;
  updated_at: string;
  inquiry: IncidentPayload | null;
};

type ActiveInvestigation = {
  id: string;
  timestamp: string;
  driver: string;
  lap: number;
  incident_type: string;
  incident_description: string;
  speed_kph: number;
  lateral_g: number;
  rule_summary: string;
  ruling: string;
  confidence_score: number;
  article_cited: string;
  driver_a?: string;
  driver_b?: string;
};

function toIsoTimestamp(value: unknown): string {
  if (typeof value === "string") {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString();
    }
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    return new Date(Date.now() + value * 1000).toISOString();
  }

  return new Date().toISOString();
}

function buildBaseIncident(packet: SimulatorPacket): IncidentPayload {
  const now = new Date().toISOString();
  const id = `incident-${Date.now()}`;
  const speedKph = Number(packet.speed ?? 0);
  const apexGap = packet.distance_to_apex ?? null;
  const lateralG = Number(packet.lateral_g ?? 0);
  const brakingForce = Number(packet.longitudinal_g ?? 0);
  const normalizedSector =
    typeof packet.sector === "string" && packet.sector.trim().length > 0
      ? packet.sector.trim().toUpperCase()
      : undefined;
  const normalizedDeltaToLeader =
    typeof packet.delta_to_leader === "number" && Number.isFinite(packet.delta_to_leader)
      ? packet.delta_to_leader
      : undefined;

  let allDrivers: Record<string, Record<string, unknown>> = {};
  
  if (Array.isArray(packet.all_drivers)) {
    for (const driver of packet.all_drivers) {
      const driverCode = String(driver.driver_code ?? "").trim().toUpperCase();
      if (!driverCode) {
        continue;
      }
      allDrivers[driverCode] = {
        driver_number: driver.driver_number,
        position_rank: Number.isFinite(driver.position_rank) ? driver.position_rank : undefined,
        lap: Number.isFinite(driver.lap_number) ? driver.lap_number : undefined,
        speed: Number.isFinite(driver.current_speed) ? driver.current_speed : undefined,
        distance_offset: Number.isFinite(driver.distance_offset) ? driver.distance_offset : undefined,
        lateral_g: Number.isFinite(driver.lateral_g) ? driver.lateral_g : undefined,
        sector: typeof driver.sector === "string" ? driver.sector.trim().toUpperCase() : undefined,
        delta_to_leader: Number.isFinite(driver.delta_to_leader) ? driver.delta_to_leader : undefined,
        incident_detected: driver.incident_detected === true,
      };
    }
  } else if (packet.all_drivers && typeof packet.all_drivers === "object") {
    allDrivers = Object.entries(packet.all_drivers as Record<string, unknown>).reduce<
      Record<string, Record<string, unknown>>
    >((acc, [driverCode, driverTelemetry]) => {
      const dt = driverTelemetry as Record<string, unknown>;
      const speed = typeof dt.speed === "number" && Number.isFinite(dt.speed as number) ? dt.speed : undefined;
      const latG = typeof dt.lateral_g === "number" && Number.isFinite(dt.lateral_g as number) ? dt.lateral_g : undefined;
      const distOffset = typeof dt.distance_offset === "number" && Number.isFinite(dt.distance_offset as number) ? dt.distance_offset : undefined;
      const lap = typeof dt.lap === "number" && Number.isFinite(dt.lap as number) ? dt.lap : undefined;
      const delta = typeof dt.delta_to_leader === "number" && Number.isFinite(dt.delta_to_leader as number) ? dt.delta_to_leader : undefined;
      const positionRank = typeof dt.position_rank === "number" && Number.isFinite(dt.position_rank as number) ? dt.position_rank : undefined;
      const sec = typeof dt.sector === "string" && dt.sector.trim().length > 0 ? dt.sector.trim().toUpperCase() : undefined;

      acc[driverCode] = {
        speed,
        lateral_g: latG,
        distance_offset: distOffset,
        sector: sec,
        lap,
        delta_to_leader: delta,
        position_rank: positionRank,
        incident_detected: dt.incident_detected === true,
      };
      return acc;
    }, {});
  }

  const triggerSteward = packet.trigger_steward === true;
  const normalizedSessionStatus = String(packet.session_status ?? "").toUpperCase();
  const isSessionFinished = normalizedSessionStatus === "FINISHED";
  const sessionName = isSessionFinished ? "POST-RACE SCRUTINEERING" : packet.track ?? "Live Simulation";
  const isNormalTelemetry = !triggerSteward;

  const agnosticIncident = packet.agnostic_incident;

  return {
    id,
    sessionName,
    track: packet.track ?? "Live Simulation",
    session_status: normalizedSessionStatus || "UNKNOWN",
    driver: packet.driver ?? "--",
    lap: Number(packet.lap ?? 0),
    sector: normalizedSector,
    timestamp: toIsoTimestamp(packet.timestamp),
    lastUpdated: now,
    speed_kph: speedKph,
    apex_gap: apexGap,
    apex_clearance: apexGap,
    lateral_g: lateralG,
    braking_force: brakingForce,
    delta_to_leader: normalizedDeltaToLeader,
    all_drivers: allDrivers,
    agnostic_incident: agnosticIncident ?? null,
    incident_type: triggerSteward
      ? (agnosticIncident ? "driver_agnostic_incident" : "high_g_event")
      : "normal_telemetry",
    incident_snapshot:
      agnosticIncident
        ? `Driver-agnostic incident: ${JSON.stringify(agnosticIncident)}`
        : `Car ${packet.driver ?? "--"} telemetry; speed ${speedKph.toFixed(1)} km/h; ` +
          `lateral load ${lateralG.toFixed(2)}G; distance to apex ${apexGap ?? "N/A"}m.`,
    article_cited: isNormalTelemetry ? null : "Awaiting Article",
    rule_summary: triggerSteward
      ? (agnosticIncident
          ? "Driver-agnostic incident detected - analyzing proximity and telemetry anomalies."
          : "Steward review triggered by telemetry threshold.")
      : "No rule violation detected in standard telemetry data.",
    verdict: triggerSteward ? "INVESTIGATION" : "NO_FURTHER_ACTION",
    ruling: triggerSteward ? "INVESTIGATION" : "NO_FURTHER_ACTION",
    confidence_score: triggerSteward ? 0.5 : 0.98,
    trigger_steward: triggerSteward,
  };

  console.log("[ROUTE] delta_to_leader:", normalizedDeltaToLeader);
}

async function runStewardAgent(incident: IncidentPayload): Promise<IncidentPayload> {
  const projectRoot = process.cwd();
  const sourceRoot = path.resolve(projectRoot, "..");
  const stewardAgentPath = path.join(sourceRoot, "brain", "steward_agent.py");

  const query =
    typeof incident.incident_snapshot === "string" && incident.incident_snapshot.trim().length > 0
      ? incident.incident_snapshot
      : "Review this telemetry incident for FIA compliance.";

  const pythonSnippet = [
    "import importlib.util, json, sys",
    "module_path = sys.argv[1]",
    "query = sys.argv[2]",
    "incident_json = sys.argv[3]",
    "spec = importlib.util.spec_from_file_location('steward_agent', module_path)",
    "module = importlib.util.module_from_spec(spec)",
    "spec.loader.exec_module(module)",
    "result = module.run_steward_agent(query=query, incident_json=incident_json)",
    "print(json.dumps(result))",
  ].join("; ");

  const runAgent = execFileAsync(
    "python",
    ["-c", pythonSnippet, stewardAgentPath, query, JSON.stringify(incident)],
    { cwd: sourceRoot, maxBuffer: 10 * 1024 * 1024 }
  );
  const timeoutMs = 30_000;
  const { stdout } = await Promise.race([
    runAgent,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`Steward agent timed out after ${timeoutMs}ms`)), timeoutMs)
    ),
  ]);

  const trimmed = stdout.trim();
  const lastLine = trimmed.split(/\r?\n/).pop() ?? "{}";
  return JSON.parse(lastLine) as IncidentPayload;
}

async function writeJsonAtomic(filePath: string, payload: unknown): Promise<void> {
  const jsonContent = `${JSON.stringify(payload, null, 2)}\n`;
  const tempPath = `${filePath}.tmp`;
  const maxRetries = 3;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      await fs.writeFile(tempPath, jsonContent, "utf-8");
      const handle = await fs.open(tempPath, "r+");
      await handle.sync();
      await handle.close();
      await fs.rename(tempPath, filePath);
      return;
    } catch (error) {
      const err = error as NodeJS.ErrnoException;
      if (err.code === "EPERM" && attempt < maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        continue;
      }
      throw error;
    }
  }
}

async function writeLiveIncident(payload: IncidentPayload): Promise<void> {
  const publicDir = path.join(process.cwd(), "public");
  const liveIncidentPath = path.join(publicDir, "live_incident.json");
  const liveIncidentTempPath = path.join(publicDir, "live_incident.tmp");
  const jsonContent = `${JSON.stringify(payload, null, 2)}\n`;
  const maxRetries = 3;

  await fs.mkdir(publicDir, { recursive: true });

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      await fs.writeFile(liveIncidentTempPath, jsonContent, "utf-8");
      const handle = await fs.open(liveIncidentTempPath, "r+");
      await handle.sync();
      await handle.close();
      await fs.rename(liveIncidentTempPath, liveIncidentPath);
      return;
    } catch (error) {
      const err = error as NodeJS.ErrnoException;
      if ((err.code === "EPERM" || err.code === "EBUSY") && attempt < maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        continue;
      }
      throw error;
    }
  }
}

async function writeCurrentInquiry(payload: IncidentPayload): Promise<void> {
  const publicDir = path.join(process.cwd(), "public");
  const currentInquiryPath = path.join(publicDir, "current_inquiry.json");
  const inquiryRecord: CurrentInquiryRecord = {
    manual_clear_required: true,
    dismissed: false,
    updated_at: new Date().toISOString(),
    inquiry: payload,
  };

  await fs.mkdir(publicDir, { recursive: true });
  await writeJsonAtomic(currentInquiryPath, inquiryRecord);
}

async function clearCurrentInquiry(): Promise<void> {
  const publicDir = path.join(process.cwd(), "public");
  const currentInquiryPath = path.join(publicDir, "current_inquiry.json");
  const inquiryRecord: CurrentInquiryRecord = {
    manual_clear_required: false,
    dismissed: true,
    updated_at: new Date().toISOString(),
    inquiry: null,
  };

  await fs.mkdir(publicDir, { recursive: true });
  await writeJsonAtomic(currentInquiryPath, inquiryRecord);
}

function toGLoad(value: unknown): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return 0;
  }

  return Math.abs(parsed);
}

function isFileLockError(error: unknown): boolean {
  const err = error as NodeJS.ErrnoException | undefined;
  const code = err?.code ?? "";
  return code === "EPERM" || code === "EACCES" || code === "EBUSY";
}

async function clearActiveInvestigations(): Promise<void> {
  const publicDir = path.join(process.cwd(), "public");
  const investigationsPath = path.join(publicDir, "active_investigations.json");

  await fs.mkdir(publicDir, { recursive: true });
  await writeJsonAtomic(investigationsPath, []);
}

async function appendActiveInvestigation(payload: IncidentPayload): Promise<void> {
  const publicDir = path.join(process.cwd(), "public");
  const investigationsPath = path.join(publicDir, "active_investigations.json");
  const maxQueueSize = 5;
  let investigations: ActiveInvestigation[] = [];

  try {
    const raw = await fs.readFile(investigationsPath, "utf-8");
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) {
      investigations = parsed as ActiveInvestigation[];
    }
  } catch {
    investigations = [];
  }

  const normalizedConfidence = Number(payload.confidence_score ?? 0);
  const confidenceScore =
    normalizedConfidence <= 1
      ? Math.round(normalizedConfidence * 100)
      : Math.round(normalizedConfidence);

  const entry: ActiveInvestigation = {
    id: String(payload.id ?? `incident-${Date.now()}`),
    timestamp: String(payload.timestamp ?? new Date().toISOString()),
    driver: String(payload.driver ?? "--"),
    lap: Number(payload.lap ?? 0),
    incident_type: String(payload.incident_type ?? "unknown_incident"),
    incident_description: String(payload.incident_description ?? payload.incident_snapshot ?? "No details."),
    speed_kph: Number(payload.speed_kph ?? 0),
    lateral_g: Number(payload.lateral_g ?? 0),
    rule_summary: String(payload.rule_summary ?? ""),
    ruling: String(payload.ruling ?? payload.verdict ?? "INVESTIGATION"),
    confidence_score: Math.max(0, Math.min(100, confidenceScore)),
    article_cited: String(payload.article_cited ?? "Awaiting Article"),
    driver_a: String(payload.driver_a ?? ""),
    driver_b: String(payload.driver_b ?? ""),
  };

  console.log("[ROUTE] appendActiveInvestigation: Writing", entry.id, "to active_investigations.json");
  
  if (investigations.length < maxQueueSize) {
    investigations.unshift(entry);
  } else {
    const incomingLoad = toGLoad(entry.lateral_g);
    let minLoadIndex = 0;
    let minLoad = toGLoad(investigations[0]?.lateral_g);

    for (let index = 1; index < investigations.length; index += 1) {
      const candidateLoad = toGLoad(investigations[index]?.lateral_g);
      if (candidateLoad <= minLoad) {
        minLoad = candidateLoad;
        minLoadIndex = index;
      }
    }

    if (incomingLoad > minLoad) {
      investigations.splice(minLoadIndex, 1);
      investigations.unshift(entry);
    }
  }

  await writeJsonAtomic(investigationsPath, investigations);
  console.log("[ROUTE] active_investigations.json updated. Total entries:", investigations.length);
}

export async function GET() {
  try {
    const investigationsPath = path.join(process.cwd(), "public", "active_investigations.json");
    const raw = await fs.readFile(investigationsPath, "utf-8");
    const parsed = JSON.parse(raw) as unknown;
    const investigations = Array.isArray(parsed) ? parsed : [];
    return NextResponse.json({ investigations }, { status: 200 });
  } catch {
    return NextResponse.json({ investigations: [] }, { status: 200 });
  }
}

export async function POST(request: Request) {
  try {
    const packet = (await request.json()) as SimulatorPacket;
    const baseIncident = buildBaseIncident(packet);
    const isSessionFinished = String(packet.session_status ?? "").toUpperCase() === "FINISHED";

    if (isSessionFinished) {
      await clearCurrentInquiry();
      await clearActiveInvestigations();
      await writeLiveIncident({
        ...baseIncident,
        trigger_steward: false,
      });

      return NextResponse.json(
        {
          ok: true,
          trigger_steward: false,
          session_status: "FINISHED",
          sessionName: "POST-RACE SCRUTINEERING",
          verdict: baseIncident.verdict ?? baseIncident.ruling ?? null,
          confidence_score: baseIncident.confidence_score ?? null,
        },
        { status: 200 }
      );
    }

    if (packet.trigger_steward === true) {
      console.log("[TELEMETRY_ROUTE] Running steward_agent for incident_type:", baseIncident.incident_type);
      console.log("[TELEMETRY_ROUTE] baseIncident keys:", Object.keys(baseIncident));
      
      const resolvedPayload = await runStewardAgent(baseIncident);
      console.log("[TELEMETRY_ROUTE] steward_agent returned:", JSON.stringify(resolvedPayload, null, 2));
      
      function cleanRuleSummary(text: unknown): string {
        if (!text || typeof text !== "string") return "No summary available";
        let cleaned = text.replace(/\[rules\/[^\]]+\]/g, "").trim();
        cleaned = cleaned.replace(/\*\*/g, "").replace(/\|/g, "•");
        return cleaned || "No summary available";
      }

      function cleanArticleCited(text: unknown): string {
        if (!text || typeof text !== "string") return "No article cited";
        const match = text.match(/rules\/driving_standards\/(.+?)\.md/);
        if (match) return `Driving Standards - ${match[1].replace(/_/g, " ")}`;
        return text.replace(/^rules\//, "").replace(/_/g, " ") || "No article cited";
      }

      const cleanedPayload = {
        ...resolvedPayload,
        rule_summary: cleanRuleSummary(resolvedPayload.rule_summary),
        article_cited: cleanArticleCited(resolvedPayload.article_cited),
      };

      const mergedPayload = {
        ...baseIncident,
        ...cleanedPayload,
      };

      await writeCurrentInquiry(mergedPayload);
      await appendActiveInvestigation(mergedPayload);
      await writeLiveIncident({
        ...mergedPayload,
        trigger_steward: false,
      });

      return NextResponse.json(
        {
          ok: true,
          trigger_steward: true,
          verdict: mergedPayload.verdict ?? mergedPayload.ruling ?? null,
          confidence_score: mergedPayload.confidence_score ?? null,
        },
        { status: 200 }
      );
    }

    await writeLiveIncident(baseIncident);
    return NextResponse.json(
      {
        ok: true,
        trigger_steward: false,
        verdict: baseIncident.verdict ?? baseIncident.ruling ?? null,
        confidence_score: baseIncident.confidence_score ?? null,
      },
      { status: 200 }
    );
  } catch (error) {
    if (isFileLockError(error)) {
      return NextResponse.json(
        {
          ok: true,
          accepted: true,
          message: "Telemetry accepted but deferred because incident file is temporarily locked.",
        },
        { status: 202 }
      );
    }

    const message = error instanceof Error ? error.message : "Unknown telemetry route error";
    console.error("[TELEMETRY_ROUTE] Error:", message);
    return NextResponse.json(
      { 
        ok: true, 
        trigger_steward: false, 
        verdict: "INVESTIGATION_CONTINUING", 
        error: message 
      }, 
      { status: 200 }
    );
  }
}
