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
  const [livePayload, setLivePayload] = useState<LiveIncidentPayload | null>(null);
  const [inquiryPayload, setInquiryPayload] = useState<LiveIncidentPayload | null>(null);
  const [dismissInFlight, setDismissInFlight] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMounted, setHasMounted] = useState(false);

  useEffect(() => {
    setHasMounted(true);
  }, []);

  useEffect(() => {
    let isMounted = true;

    const fetchTelemetry = async () => {
      try {
        const liveResponse = await fetch("/live_incident.json", { cache: "no-store" });
        if (!liveResponse.ok) {
          throw new Error(`Telemetry fetch failed (${liveResponse.status})`);
        }

        const liveData = (await liveResponse.json()) as LiveIncidentPayload;
        if (isMounted) {
          setLivePayload(liveData);
        }
      } catch (liveError) {
        if (isMounted) {
          setError(liveError instanceof Error ? liveError.message : "Unknown telemetry error");
        }
      }

      try {
        const inquiryResponse = await fetch("/api/current-inquiry", { cache: "no-store" });
        if (!inquiryResponse.ok) {
          throw new Error(`Current inquiry fetch failed (${inquiryResponse.status})`);
        }

        const inquiryData = (await inquiryResponse.json()) as {
          manual_clear_required?: boolean;
          inquiry?: LiveIncidentPayload | null;
        };

        if (isMounted) {
          const pinned =
            inquiryData.manual_clear_required === true && inquiryData.inquiry ? inquiryData.inquiry : null;
          setInquiryPayload(pinned);
          setError(null);
        }
      } catch (inquiryError) {
        if (isMounted) {
          setInquiryPayload(null);
          setError(inquiryError instanceof Error ? inquiryError.message : "Unknown inquiry error");
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

  const displayedPayload = inquiryPayload ?? livePayload;
  const liveSector = livePayload?.sector;

  const currentFact = useMemo(() => {
    if (!displayedPayload) {
      return null;
    }

    const normalizedConfidence = displayedPayload.confidence_score ?? 0;
    const confidence =
      normalizedConfidence <= 1 ? Math.round(normalizedConfidence * 100) : Math.round(normalizedConfidence);

    return {
      id: displayedPayload.id ?? "live-incident",
      timestamp: displayedPayload.timestamp ?? "",
      lap: displayedPayload.lap ?? 0,
      driver: displayedPayload.driver ?? "--",
      incidentType: displayedPayload.incident_type ?? "--",
      lateralG: displayedPayload.lateral_g ?? 0,
      speedKph: displayedPayload.speed_kph ?? 0,
      deltaToLeader: displayedPayload.delta_to_leader ?? displayedPayload.apex_gap ?? 0,
      trackTempC: displayedPayload.track_temp_c ?? 0,
      sector: liveSector ?? displayedPayload.sector ?? "N/A",
      incident: displayedPayload.incident_description ?? displayedPayload.incident_type ?? "No incident details.",
      confidence: Math.max(0, Math.min(100, confidence)),
      fiaArticle: displayedPayload.article_cited ?? "Awaiting Article",
      ruleSummary: displayedPayload.rule_summary ?? "",
      ruling: displayedPayload.ruling ?? displayedPayload.verdict ?? "Pending",
      triggerSteward: displayedPayload.trigger_steward === true,
    } satisfies IncidentFact;
  }, [displayedPayload, liveSector]);

  const sessionName = displayedPayload?.sessionName ?? displayedPayload?.track ?? EMPTY_UPDATE.sessionName;
  const recentJudgements = useMemo(
    () => displayedPayload?.recentJudgements ?? [],
    [displayedPayload?.recentJudgements]
  );
  const lastUpdated = displayedPayload?.lastUpdated ?? displayedPayload?.timestamp ?? EMPTY_UPDATE.lastUpdated;

  const handleDismissInquiry = async () => {
    try {
      setDismissInFlight(true);
      const response = await fetch("/api/current-inquiry", { method: "DELETE" });
      if (!response.ok) {
        throw new Error(`Dismiss failed (${response.status})`);
      }
      setInquiryPayload(null);
      setError(null);
    } catch (dismissError) {
      setError(dismissError instanceof Error ? dismissError.message : "Unable to dismiss current inquiry");
    } finally {
      setDismissInFlight(false);
    }
  };

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

      {inquiryPayload ? (
        <div className="mt-4 flex items-center justify-between gap-3 rounded-xl border border-[#FF1801]/35 bg-[#FF1801]/10 p-3">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[#FF1801]">
            Manual Clear Required
          </p>
          <button
            type="button"
            onClick={handleDismissInquiry}
            disabled={dismissInFlight}
            className="rounded-md border border-[#FF1801]/45 bg-[#FF1801]/12 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em] text-[#FF1801] hover:bg-[#FF1801]/20 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {dismissInFlight ? "Dismissing..." : "DISMISS"}
          </button>
        </div>
      ) : null}

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <Metric label="Driver" value={currentFact?.driver ?? "--"} />
        <Metric
          label="Incident Type"
          value={(currentFact?.incidentType ?? "--").replace(/_/g, ' ')}
          valueClassName="capitalize"
        />
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
  valueClassName?: string;
};

function Metric({ label, value, accent, pulse, valueClassName }: MetricProps) {
  return (
    <div className="rounded-xl border border-white/5 bg-black/30 p-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">{label}</p>
      <p className={`mt-1 text-sm font-semibold ${pulse ? "animate-pulse text-zinc-300" : accent ? "text-[#FF1801]" : "text-zinc-100"} ${valueClassName ?? ""}`}>{value}</p>
    </div>
  );
}
