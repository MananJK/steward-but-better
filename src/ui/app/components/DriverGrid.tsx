"use client";

import { useEffect, useMemo, useState, memo } from "react";

import type { ActiveInvestigation, LiveIncidentPayload } from "../types/incident";

type DriverSnapshot = {
  driverCode: string;
  speed: number | null;
  positionRank: number | null;
  lap: number | null;
  sector: string | null;
  deltaToLeader: number | null;
  incidentDetected: boolean;
};

const GRID_CELL_COUNT = 24;

function parseDriverEntry(entry: Record<string, unknown>): DriverSnapshot | null {
  const driverCode = String(entry.driver_code ?? entry.driver ?? "").trim().toUpperCase();
  if (!driverCode) {
    return null;
  }

  const speed = typeof entry.current_speed === "number"
    ? entry.current_speed
    : typeof entry.speed === "number"
      ? entry.speed
      : typeof entry.speed_kph === "number"
        ? entry.speed_kph
        : null;
  const positionRank = typeof entry.position_rank === "number" ? entry.position_rank : null;
  const lap = typeof entry.lap_number === "number"
    ? entry.lap_number
    : typeof entry.lap === "number"
      ? entry.lap
      : null;
  const sector = typeof entry.sector === "string" ? entry.sector : null;
  const deltaToLeader = typeof entry.delta_to_leader === "number" ? entry.delta_to_leader : null;

  return {
    driverCode,
    speed,
    positionRank,
    lap,
    sector,
    deltaToLeader,
    incidentDetected: entry.incident_detected === true,
  } satisfies DriverSnapshot;
}

function normalizeEntries(payload: unknown): DriverSnapshot[] {
  if (!payload) {
    return [];
  }

  if (Array.isArray(payload)) {
    return payload
      .map((entry) => parseDriverEntry(entry as Record<string, unknown>))
      .filter((entry): entry is DriverSnapshot => entry !== null);
  }

  const payloadObj = payload as Record<string, unknown>;
  const allDrivers = payloadObj.all_drivers;

  if (!allDrivers) {
    return [];
  }

  if (Array.isArray(allDrivers)) {
    return allDrivers
      .map((entry) => parseDriverEntry(entry as Record<string, unknown>))
      .filter((entry): entry is DriverSnapshot => entry !== null);
  }

  return Object.entries(allDrivers as Record<string, unknown>).map(([driverCode, entry]) => {
    const entryWithDriver = { ...entry as Record<string, unknown>, driver_code: driverCode };
    const parsed = parseDriverEntry(entryWithDriver);
    if (parsed) {
      return { ...parsed, driverCode: driverCode.toUpperCase() };
    }
    return null;
  }).filter((entry): entry is DriverSnapshot => entry !== null);
}

const DriverGrid = memo(function DriverGrid() {
  const [payload, setPayload] = useState<LiveIncidentPayload | null>(null);
  const [investigations, setInvestigations] = useState<ActiveInvestigation[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lastValidDrivers, setLastValidDrivers] = useState<DriverSnapshot[]>([]);

  useEffect(() => {
    let isMounted = true;

    const fetchLiveAndInvestigations = async () => {
      try {
        const [liveResponse, telemetryResponse] = await Promise.all([
          fetch("/live_incident.json", { cache: "no-store" }),
          fetch("/api/telemetry", { cache: "no-store" }),
        ]);

        let nextPayload: LiveIncidentPayload | null = null;
        if (liveResponse.ok) {
          nextPayload = await liveResponse.json() as LiveIncidentPayload;
        }

        let nextInvestigations: ActiveInvestigation[] = [];
        if (telemetryResponse.ok) {
          const telemetryData = await telemetryResponse.json() as { investigations: ActiveInvestigation[] };
          nextInvestigations = Array.isArray(telemetryData.investigations) ? telemetryData.investigations : [];
        }

        if (isMounted) {
          setPayload(nextPayload);
          setInvestigations(nextInvestigations);
          setError(null);
        }
      } catch (fetchError) {
        if (isMounted) {
          setError(fetchError instanceof Error ? fetchError.message : "Unable to load dashboard feeds");
        }
      }
    };

    fetchLiveAndInvestigations();
    const interval = window.setInterval(fetchLiveAndInvestigations, 2000);

    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const entries = normalizeEntries(payload);
    if (entries.length > 0) {
      setLastValidDrivers(entries);
    }
  }, [payload]);

  const driverSlots = useMemo(() => {
    const entries = normalizeEntries(payload);
    const displayEntries = entries.length > 0 ? entries : lastValidDrivers;
    
    return Array.from({ length: GRID_CELL_COUNT }, (_, index) => ({
      grid: index + 1,
      telemetry: displayEntries[index] ?? null,
      isStale: entries.length === 0 && lastValidDrivers.length > 0,
    }));
  }, [payload, lastValidDrivers]);

  const investigatedDrivers = useMemo(() => {
    const set = new Set<string>();
    investigations.forEach((entry) => {
      if (entry.driver) {
        set.add(entry.driver.toUpperCase());
      }
    });
    return set;
  }, [investigations]);

  return (
    <section className="race-panel rounded-2xl border border-white/10 p-4 shadow-2xl shadow-black/40">
      <div className="mb-4 flex items-end justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-zinc-400">Driver Matrix</p>
          <h2 className="mt-1 text-xl font-semibold text-zinc-100">Live 4x6 Race Grid</h2>
        </div>
        <p className="text-xs text-zinc-400">Session: {payload?.sessionName ?? payload?.track ?? "Live Feed"}</p>
      </div>

      {error ? <p className="mb-3 text-sm text-red-300">{error}</p> : null}

      <div className="grid grid-cols-6 gap-3">
        {driverSlots.map((slot) => {
          const telemetry = slot.telemetry;
          const isStale = slot.isStale;
          
          if (!telemetry) {
            return (
              <article
                key={slot.grid}
                className="min-h-[170px] rounded-xl border border-white/10 bg-black/20 p-4"
              >
                <p className="text-[10px] uppercase tracking-[0.2em] text-zinc-500">Slot {slot.grid}</p>
                <p className="mt-6 text-center text-sm font-semibold uppercase tracking-[0.16em] text-zinc-500">
                  NO_DRIVER
                </p>
              </article>
            );
          }

          const isActiveInvestigation = investigatedDrivers.has(telemetry.driverCode);
          const isIncident = telemetry.incidentDetected;
          const speedText = telemetry.speed !== null ? `${telemetry.speed.toFixed(1)} km/h` : "--";
          const rankText = telemetry.positionRank !== null && telemetry.positionRank > 0 ? `P${telemetry.positionRank}` : "--";
          const lapText = telemetry.lap !== null ? `Lap ${telemetry.lap}` : "--";
          const sectorText = telemetry.sector ?? "--";
          const deltaText = telemetry.deltaToLeader !== null ? `${telemetry.deltaToLeader.toFixed(3)}s` : "--";

          return (
            <article
              key={slot.grid}
              className={`min-h-[170px] rounded-xl border bg-black/30 p-4 transition-colors ${
                isActiveInvestigation
                  ? "border-red-500 shadow-[0_0_22px_rgba(239,68,68,0.55)]"
                  : isIncident
                    ? "border-red-400/70 shadow-[0_0_14px_rgba(248,113,113,0.4)]"
                    : isStale
                      ? "border-amber-500/30"
                      : "border-white/10"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2">
                  <p className="text-[10px] uppercase tracking-[0.2em] text-zinc-500">Slot {slot.grid}</p>
                  {isStale && (
                    <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-wider text-amber-400">
                      STALE
                    </span>
                  )}
                </div>
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#FF1801]">{rankText}</p>
              </div>
              <p className={`mt-1 text-lg font-semibold ${isStale ? "text-amber-400" : "text-zinc-100"}`}>{telemetry.driverCode}</p>

              <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2">
                <Metric label="Speed" value={speedText} />
                <Metric label="Lap" value={lapText} />
                <Metric label="Sector" value={sectorText} />
                <Metric label="Delta" value={deltaText} />
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
});

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">{label}</p>
      <p className="text-sm font-semibold text-zinc-200">{value}</p>
    </div>
  );
}

export default DriverGrid;
