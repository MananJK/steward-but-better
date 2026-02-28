"use client";

import { useEffect, useMemo, useState } from "react";
import type { IncidentFact, IncidentFactsPayload } from "../types/incident";

type TelemetryUpdate = {
  currentFact: IncidentFact | null;
  facts: IncidentFact[];
  verdicts: string[];
  session: string;
  lastUpdated: string;
};

type TelemetryCardProps = {
  onTelemetryUpdate?: (update: TelemetryUpdate) => void;
};

const EMPTY_UPDATE: TelemetryUpdate = {
  currentFact: null,
  facts: [],
  verdicts: [],
  session: "No Session Loaded",
  lastUpdated: "",
};

export default function TelemetryCard({ onTelemetryUpdate }: TelemetryCardProps) {
  const [payload, setPayload] = useState<IncidentFactsPayload | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const fetchTelemetry = async () => {
      try {
        const response = await fetch("/incident_facts.json", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Telemetry fetch failed (${response.status})`);
        }

        const data = (await response.json()) as IncidentFactsPayload;
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
    const refreshInterval = window.setInterval(fetchTelemetry, 20000);

    return () => {
      isMounted = false;
      window.clearInterval(refreshInterval);
    };
  }, []);

  useEffect(() => {
    if (!payload?.facts?.length) {
      setActiveIndex(0);
      return;
    }

    const cycleInterval = window.setInterval(() => {
      setActiveIndex((previous) => (previous + 1) % payload.facts.length);
    }, 2600);

    return () => window.clearInterval(cycleInterval);
  }, [payload?.facts]);

  const currentFact = useMemo(
    () => payload?.facts?.[activeIndex] ?? null,
    [payload?.facts, activeIndex],
  );

  useEffect(() => {
    onTelemetryUpdate?.({
      currentFact,
      facts: payload?.facts ?? [],
      verdicts: payload?.aiVerdicts ?? [],
      session: payload?.session ?? EMPTY_UPDATE.session,
      lastUpdated: payload?.lastUpdated ?? EMPTY_UPDATE.lastUpdated,
    });
  }, [currentFact, onTelemetryUpdate, payload]);

  const confidenceWidth = `${Math.max(0, Math.min(100, currentFact?.confidence ?? 0))}%`;

  return (
    <section className="race-panel h-full rounded-2xl border border-white/10 p-5 shadow-2xl shadow-black/40">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-zinc-400">Telemetry Feed</p>
          <h2 className="mt-2 text-xl font-semibold text-zinc-100">{payload?.session ?? "Awaiting Feed"}</h2>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-emerald-300">
          <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />
          Live
        </div>
      </div>

      {error ? <p className="mt-4 text-sm text-red-300">{error}</p> : null}

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <Metric label="Driver" value={currentFact?.driver ?? "--"} />
        <Metric label="Sector" value={currentFact?.sector ?? "--"} />
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
          label="Track Temp"
          value={currentFact ? `${currentFact.trackTempC.toFixed(1)} C` : "--"}
        />
      </div>

      <div className="mt-5">
        <p className="text-xs uppercase tracking-[0.22em] text-zinc-500">Incident Snapshot</p>
        <p className="mt-2 text-sm leading-relaxed text-zinc-300">
          {currentFact?.incident ?? "Waiting for incident facts from Kilo-code data feed."}
        </p>
      </div>

      <div className="mt-5">
        <div className="flex items-center justify-between text-xs uppercase tracking-[0.22em] text-zinc-500">
          <span>Brain Confidence</span>
          <span className="text-zinc-300">{currentFact?.confidence ?? 0}%</span>
        </div>
        <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
          <div className="h-full rounded-full bg-[#FF1801] transition-[width] duration-500" style={{ width: confidenceWidth }} />
        </div>
      </div>

      <p className="mt-5 text-xs text-zinc-500">
        Last Sync: {payload?.lastUpdated ?? "Pending"}
      </p>
    </section>
  );
}

type MetricProps = {
  label: string;
  value: string;
  accent?: boolean;
};

function Metric({ label, value, accent }: MetricProps) {
  return (
    <div className="rounded-xl border border-white/5 bg-black/30 p-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">{label}</p>
      <p className={`mt-1 text-sm font-semibold ${accent ? "text-[#FF1801]" : "text-zinc-100"}`}>{value}</p>
    </div>
  );
}
