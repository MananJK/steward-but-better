import { execFile } from "node:child_process";
import { promises as fs, renameSync, writeFileSync } from "node:fs";
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
  track?: string;
  timestamp?: string | number;
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
  const triggerSteward = packet.trigger_steward === true;
  const normalizedSessionStatus = String(packet.session_status ?? "").toUpperCase();
  const isSessionFinished = normalizedSessionStatus === "FINISHED";
  const sessionName = isSessionFinished ? "POST-RACE SCRUTINEERING" : packet.track ?? "Live Simulation";

  return {
    id,
    sessionName,
    track: packet.track ?? "Live Simulation",
    session_status: normalizedSessionStatus || "UNKNOWN",
    driver: packet.driver ?? "--",
    lap: Number(packet.lap ?? 0),
    timestamp: toIsoTimestamp(packet.timestamp),
    lastUpdated: now,
    speed_kph: speedKph,
    apex_gap: apexGap,
    apex_clearance: apexGap,
    lateral_g: lateralG,
    braking_force: brakingForce,
    incident_type: triggerSteward ? "high_g_event" : "normal_telemetry",
    incident_snapshot:
      `Car ${packet.driver ?? "--"} telemetry; speed ${speedKph.toFixed(1)} km/h; ` +
      `lateral load ${lateralG.toFixed(2)}G; distance to apex ${apexGap ?? "N/A"}m.`,
    article_cited: "Awaiting Article",
    rule_summary: triggerSteward
      ? "Steward review triggered by telemetry threshold."
      : "No steward trigger in current packet.",
    verdict: triggerSteward ? "INVESTIGATION" : "NO_FURTHER_ACTION",
    ruling: triggerSteward ? "INVESTIGATION" : "NO_FURTHER_ACTION",
    confidence_score: triggerSteward ? 0.5 : 0.9,
    trigger_steward: triggerSteward,
  };
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

  await fs.mkdir(publicDir, { recursive: true });
  writeFileSync(liveIncidentTempPath, jsonContent, "utf-8");
  renameSync(liveIncidentTempPath, liveIncidentPath);
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
  };

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

    await writeLiveIncident(baseIncident);

    if (packet.trigger_steward !== true) {
      return NextResponse.json(
        {
          ok: true,
          trigger_steward: false,
          verdict: baseIncident.verdict ?? baseIncident.ruling ?? null,
          confidence_score: baseIncident.confidence_score ?? null,
        },
        { status: 200 }
      );
    }

    const resolvedPayload = await runStewardAgent(baseIncident);
    const mergedPayload = {
      ...baseIncident,
      ...resolvedPayload,
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
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
