"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { ActiveInvestigation } from "../types/incident";

type LocalInvestigation = ActiveInvestigation & {
  isExiting?: boolean;
};

const DISMISS_ANIMATION_MS = 2000;

export default function IncidentLog() {
  const [investigations, setInvestigations] = useState<LocalInvestigation[]>([]);
  const [closedIds, setClosedIds] = useState<Record<string, boolean>>({});
  const [dismissInFlightIds, setDismissInFlightIds] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const dismissTimeoutsRef = useRef<Record<string, number>>({});

  useEffect(() => {
    let isMounted = true;
    const dismissTimeouts = dismissTimeoutsRef.current;

    const fetchInvestigations = async () => {
      try {
        const response = await fetch("/active_investigations.json", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Investigation fetch failed (${response.status})`);
        }

        const data = (await response.json()) as ActiveInvestigation[];
        if (isMounted) {
          setInvestigations((previous) => {
            const exitingIds = new Set(previous.filter((entry) => entry.isExiting).map((entry) => entry.id));
            const nextEntries = Array.isArray(data) ? data : [];
            return nextEntries.map((entry) =>
              exitingIds.has(entry.id) ? { ...entry, isExiting: true } : entry
            );
          });
          setError(null);
        }
      } catch (fetchError) {
        if (isMounted) {
          setError(fetchError instanceof Error ? fetchError.message : "Unknown investigation error");
        }
      }
    };

    fetchInvestigations();
    const interval = window.setInterval(fetchInvestigations, 2000);

    return () => {
      isMounted = false;
      window.clearInterval(interval);
      Object.values(dismissTimeouts).forEach((timeoutId) => window.clearTimeout(timeoutId));
    };
  }, []);

  const handleClose = (id: string) => {
    setClosedIds((previous) => ({ ...previous, [id]: true }));
  };

  const handleDismissInvestigation = (id: string) => {
    if (dismissInFlightIds[id]) {
      return;
    }

    setDismissInFlightIds((previous) => ({ ...previous, [id]: true }));
    setInvestigations((previous) =>
      previous.map((entry) => (entry.id === id ? { ...entry, isExiting: true } : entry))
    );

    const timeoutId = window.setTimeout(async () => {
      try {
        const response = await fetch(`/api/investigations/${encodeURIComponent(id)}`, {
          method: "DELETE",
        });
        if (!response.ok) {
          throw new Error(`Dismiss failed (${response.status})`);
        }

        setInvestigations((previous) => previous.filter((entry) => entry.id !== id));
        setError(null);
      } catch (dismissError) {
        setInvestigations((previous) =>
          previous.map((entry) => (entry.id === id ? { ...entry, isExiting: false } : entry))
        );
        setError(dismissError instanceof Error ? dismissError.message : "Unable to dismiss investigation");
      } finally {
        setDismissInFlightIds((previous) => {
          const next = { ...previous };
          delete next[id];
          return next;
        });
        delete dismissTimeoutsRef.current[id];
      }
    }, DISMISS_ANIMATION_MS);

    dismissTimeoutsRef.current[id] = timeoutId;
  };

  const totalOpen = useMemo(
    () => investigations.filter((entry) => !closedIds[entry.id] && !entry.isExiting).length,
    [closedIds, investigations]
  );

  return (
    <aside className="race-panel h-full rounded-2xl border border-white/10 p-5 shadow-2xl shadow-black/40">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-zinc-400">Incident History</p>
          <h2 className="mt-2 text-xl font-semibold text-zinc-100">Active Investigations</h2>
        </div>
        <div className="rounded-full border border-[#FF1801]/40 bg-[#FF1801]/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-[#FF1801]">
          {totalOpen} Open
        </div>
      </div>

      {error ? <p className="mt-4 text-sm text-red-300">{error}</p> : null}

      <div className="mt-5 max-h-[62vh] space-y-4 overflow-y-auto pr-1">
        {investigations.length === 0 ? (
          <div className="rounded-xl border border-white/10 bg-black/30 p-4 text-sm text-zinc-400">
            No investigations in the log.
          </div>
        ) : (
          investigations.map((entry) => {
            const isClosed = closedIds[entry.id] === true;
            const isDismissing = dismissInFlightIds[entry.id] === true;
            return (
              <article
                key={entry.id}
                className={`rounded-xl border border-white/10 bg-black/30 p-4 transition-all duration-[2000ms] ease-in-out ${
                  entry.isExiting ? "translate-x-12 opacity-0 pointer-events-none" : "translate-x-0 opacity-100"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">
                      {entry.timestamp}
                    </p>
                    <h3 className="mt-1 text-base font-semibold text-zinc-100">
                      {entry.driver} | Lap {entry.lap}
                    </h3>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleClose(entry.id)}
                    disabled={isClosed}
                    className={`rounded-md border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] ${
                      isClosed
                        ? "cursor-default border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                        : "border-zinc-500/50 bg-zinc-800/60 text-zinc-200 hover:border-zinc-300"
                    }`}
                  >
                    {isClosed ? "CLOSED" : "CLOSE"}
                  </button>
                </div>

                <p className="mt-3 text-sm leading-relaxed text-zinc-300">{entry.incident_description}</p>

                <div className="mt-3 grid gap-2 text-xs text-zinc-400 sm:grid-cols-2">
                  <p>Type: {entry.incident_type}</p>
                  <p>Speed: {entry.speed_kph.toFixed(1)} km/h</p>
                  <p>Lateral G: {entry.lateral_g.toFixed(2)}G</p>
                  <p>Confidence: {entry.confidence_score}%</p>
                </div>

                <p className="mt-3 text-xs text-zinc-500">Article: {entry.article_cited}</p>
                <p className="mt-1 text-xs text-zinc-500">Ruling: {entry.ruling}</p>

                <button
                  type="button"
                  onClick={() => handleDismissInvestigation(entry.id)}
                  disabled={isDismissing}
                  className="mt-4 rounded-md border border-[#FF1801]/45 bg-[#FF1801]/12 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em] text-[#FF1801] hover:bg-[#FF1801]/20 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isDismissing ? "Dismissing..." : "Dismiss Investigation"}
                </button>
              </article>
            );
          })
        )}
      </div>
    </aside>
  );
}
