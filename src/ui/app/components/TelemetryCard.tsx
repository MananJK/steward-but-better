"use client";

import { useEffect, useMemo, useState } from "react";
import type { IncidentFact, LiveIncidentPayload } from "../types/incident";

type TelemetryUpdate = {
  currentFact: IncidentFact | null;
  recentJudgements: string[];
  sessionName: string;
  lastUpdated: string;
};

type TelemetryCardProps = {
  onTelemetryUpdate?: (update: TelemetryUpdate) => void;
};

const EMPTY_UPDATE: TelemetryUpdate = {
  currentFact: null,
  recentJudgements: [],
  sessionName: "No Session Loaded",
  lastUpdated: "",
};

export default function TelemetryCard({ onTelemetryUpdate }: TelemetryCardProps) {
  const [payload, setPayload] = useState<LiveIncidentPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hasMounted, setHasMounted] = useState(false);

  useEffect(() => {
    setHasMounted(true);
  }, []);

  useEffect(() => {
    let isMounted = true;

    const fetchTelemetry = async () => {
      try {
        const response = await fetch("/live_incident.json", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Telemetry fetch failed (${response.status})`);
        }

        const data = (await response.json()) as LiveIncidentPayload;
        if (isMounted) {
          setPayload(data);
          setError(null);
        }
      } catch (fetchError) {
        if (isMounted) {
          setError(fetchError instanceof Error ? fetchError.message : "Unknown telemetry error");
        }
      }
    };

    fetchTelemetry();
    const refreshInterval = window.setInterval(fetchTelemetry, 2000);

    return () => {
      isMounted = false;
      window.clearInterval(refreshInterval);
    };
  }, []);

  const currentFact = useMemo(() => {
    if (!payload) {
      return null;
    }

    const normalizedConfidence = payload.confidence_score ?? 0;
    const confidence =
      normalizedConfidence <= 1 ? Math.round(normalizedConfidence * 100) : Math.round(normalizedConfidence);

    return {
      id: payload.id ?? "live-incident",
      timestamp: payload.timestamp ?? "",
      lap: payload.lap ?? 0,
      driver: payload.driver ?? "--",
      incidentType: payload.incident_type ?? "--",
      speedKph: payload.speed_kph ?? 0,
      deltaToLeader: payload.delta_to_leader ?? payload.apex_gap ?? 0,
      trackTempC: payload.track_temp_c ?? 0,
      sector: payload.sector ?? "N/A",
      incident: payload.incident_description ?? payload.incident_type ?? "No incident details.",
      confidence: Math.max(0, Math.min(100, confidence)),
      fiaArticle: payload.article_cited ?? "Awaiting Article",
      ruleSummary: payload.rule_summary ?? "",
      ruling: payload.ruling ?? payload.verdict ?? "Pending",
    } satisfies IncidentFact;
  }, [payload]);

  const sessionName = payload?.sessionName ?? payload?.track ?? EMPTY_UPDATE.sessionName;
  const recentJudgements = useMemo(() => payload?.recentJudgements ?? [], [payload?.recentJudgements]);
  const lastUpdated = payload?.lastUpdated ?? payload?.timestamp ?? EMPTY_UPDATE.lastUpdated;

  useEffect(() => {
    onTelemetryUpdate?.({
      currentFact,
      recentJudgements,
      sessionName,
      lastUpdated,
    });
  }, [currentFact, lastUpdated, onTelemetryUpdate, recentJudgements, sessionName]);

  const confidenceWidth = `${Math.max(0, Math.min(100, currentFact?.confidence ?? 0))}%`;
  const confidence = currentFact?.confidence ?? 0;
  const confidenceBarColorClass =
    confidence < 50 ? "bg-[#FF1801]" : confidence <= 80 ? "bg-[#FF8C00]" : "bg-[#16A34A]";
  const isSectorPending = !currentFact?.sector || currentFact.sector === "N/A";
  const confidenceStyle = hasMounted
    ? { width: confidenceWidth }
    : { width: "0%" };

  return (
    <section className="race-panel h-full rounded-2xl border border-white/10 p-5 shadow-2xl shadow-black/40">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-zinc-400">Telemetry Feed</p>
          <h2 className="mt-2 text-xl font-semibold text-zinc-100">{sessionName || "Awaiting Feed"}</h2>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-emerald-300">
          <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />
          Live
        </div>
      </div>

      {error ? <p className="mt-4 text-sm text-red-300">{error}</p> : null}

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <Metric label="Driver" value={currentFact?.driver ?? "--"} />
        <Metric label="Incident Type" value={currentFact?.incidentType ?? "--"} />
        <Metric label="Lap" value={currentFact ? `Lap ${currentFact.lap}` : "--"} />
        <Metric
          label="Speed"
          value={currentFact ? `${currentFact.speedKph.toFixed(1)} km/h` : "--"}
          accent
        />
        <Metric
          label="Delta"
          value={currentFact ? `${currentFact.deltaToLeader.toFixed(3)}s` : "--"}
        />
        <Metric
          label="Sector"
          value={isSectorPending ? "Calculating..." : currentFact?.sector ?? "--"}
          pulse={isSectorPending}
        />
      </div>

      <div className="mt-5">
        <p className="text-xs uppercase tracking-[0.22em] text-zinc-500">Incident Snapshot</p>
        <p className="mt-2 text-sm leading-relaxed text-zinc-300">
          {currentFact?.incident ?? "Waiting for live incident data feed."}
        </p>
      </div>

      <div className="mt-5">
        <div className="flex items-center justify-between text-xs uppercase tracking-[0.22em] text-zinc-500">
          <span>Brain Confidence</span>
          <span className="text-zinc-300">{currentFact?.confidence ?? 0}%</span>
        </div>
        <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
          <div
            suppressHydrationWarning
            className={`h-full rounded-full transition-[width,background-color] duration-500 ${confidenceBarColorClass}`}
            style={confidenceStyle}
          />
        </div>
      </div>

      <p className="mt-5 text-xs text-zinc-500">
        Last Sync: {lastUpdated || "Pending"}
      </p>
    </section>
  );
}

type MetricProps = {
  label: string;
  value: string;
  accent?: boolean;
  pulse?: boolean;
};

function Metric({ label, value, accent, pulse }: MetricProps) {
  return (
    <div className="rounded-xl border border-white/5 bg-black/30 p-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">{label}</p>
      <p className={`mt-1 text-sm font-semibold ${pulse ? "animate-pulse text-zinc-300" : accent ? "text-[#FF1801]" : "text-zinc-100"}`}>{value}</p>
    </div>
  );
}
