"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { ActiveInvestigation } from "../types/incident";

const DISMISS_ANIMATION_MS = 2000;

export default function IncidentLog() {
  const [investigations, setInvestigations] = useState<ActiveInvestigation[]>([]);
  const [closedIds, setClosedIds] = useState<Record<string, boolean>>({});
  const [dismissInFlightIds, setDismissInFlightIds] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const dismissTimeoutsRef = useRef<Record<string, number>>({});
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);

  const mutate = async (key: string) => {
    if (key !== "/api/telemetry") {
      return;
    }

    const response = await fetch("/api/telemetry", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Investigation refresh failed (${response.status})`);
    }

    const json = (await response.json()) as { investigations: ActiveInvestigation[] };
    const data = json.investigations ?? [];
    setInvestigations(Array.isArray(data) ? data : []);
  };

  useEffect(() => {
    let isMounted = true;
    const dismissTimeouts = dismissTimeoutsRef.current;

    const fetchInvestigations = async () => {
      try {
        const response = await fetch("/api/telemetry", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Investigation fetch failed (${response.status})`);
        }

        const json = (await response.json()) as { investigations: ActiveInvestigation[] };
        const data = json.investigations ?? [];
        if (isMounted) {
          setInvestigations(Array.isArray(data) ? data : []);
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

    const timeoutId = window.setTimeout(async () => {
        try {
          const response = await fetch(`/api/investigations/${encodeURIComponent(id)}`, {
            method: "DELETE",
          });
          if (!response.ok) {
            throw new Error(`Dismiss failed (${response.status})`);
          }

          await mutate("/api/investigations");
          setInvestigations((previous) => previous.filter((entry) => entry.id !== id));
          setClosedIds((previous) => {
            if (!(id in previous)) {
              return previous;
            }
            const next = { ...previous };
            delete next[id];
            return next;
          });
          setError(null);
        } catch (dismissError) {
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
    () => investigations.filter((entry) => !closedIds[entry.id]).length,
    [closedIds, investigations]
  );
  const orderedInvestigations = useMemo(() => [...investigations].reverse(), [investigations]);

  useEffect(() => {
    if (!scrollContainerRef.current) {
      return;
    }

    scrollContainerRef.current.scrollTo({
      top: scrollContainerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [orderedInvestigations]);

  return (
    <aside className="race-panel h-full rounded-2xl border border-white/10 p-5 shadow-2xl shadow-black/40">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-zinc-400">Incident Log</p>
          <h2 className="mt-2 text-xl font-semibold text-zinc-100">Active Investigations</h2>
        </div>
        <div className="rounded-full border border-[#FF1801]/40 bg-[#FF1801]/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-[#FF1801]">
          {totalOpen} Open
        </div>
      </div>

      {error ? <p className="mt-4 text-sm text-red-300">{error}</p> : null}

      <div ref={scrollContainerRef} className="mt-5 max-h-[82vh] space-y-4 overflow-y-auto pr-1">
        {orderedInvestigations.length === 0 ? (
          <div className="rounded-xl border border-white/10 bg-black/30 p-4 text-sm text-zinc-400">
            No investigations in the log.
          </div>
        ) : (
          orderedInvestigations.map((entry) => {
            const isClosed = closedIds[entry.id] === true;
            const isDismissing = dismissInFlightIds[entry.id] === true;
            const ruling = entry.ruling;
            const article_cited = entry.article_cited;
            const rule_summary = entry.rule_summary;
            const hasValidRuling = ruling && ruling.trim().length > 0;
            const verdictText = hasValidRuling ? ruling : "Awaiting Verdict...";
            const articleText = article_cited && article_cited.trim().length > 0 && !article_cited.includes("Awaiting") ? article_cited : "Awaiting Verdict...";
            const summaryText = rule_summary && rule_summary.trim().length > 0 && !rule_summary.includes("Awaiting") ? rule_summary : "Awaiting Verdict...";
            return (
              <article
                key={entry.id}
                className={`rounded-xl border bg-black/30 p-4 transition-all duration-300 ease-in-out ${
                  isDismissing
                    ? "border-emerald-400/70 opacity-50"
                    : "border-white/10 opacity-100"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">
                      {entry.timestamp}
                    </p>
                    <h3 className="mt-1 text-base font-semibold text-zinc-100">
                      Verdict Card | Lap {entry.lap}
                    </h3>
                    {(entry.driver_a || entry.driver_b) && (
                      <p className="mt-1 text-xs font-medium text-amber-400">
                        {entry.driver_a}
                        {entry.driver_b && ` vs ${entry.driver_b}`}
                      </p>
                    )}
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

                <div className="mt-5 space-y-4 text-sm text-zinc-200">
                  <div>
                    <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500">Verdict</p>
                    <p className="text-lg font-semibold uppercase text-zinc-100">{verdictText}</p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500">
                      Article Cited <span className="text-[10px] font-normal text-zinc-500">(legal basis)</span>
                    </p>
                    <p className="text-lg font-bold text-[#FF1801]">{articleText}</p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500">Summary</p>
                    <p className="max-h-[150px] overflow-y-auto text-sm leading-relaxed text-zinc-300">{summaryText}</p>
                  </div>
                </div>

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
