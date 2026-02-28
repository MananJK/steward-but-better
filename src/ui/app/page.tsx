"use client";

import { useCallback, useState } from "react";
import RulesReference from "./components/RulesReference";
import TelemetryCard from "./components/TelemetryCard";
import VerdictTicker from "./components/VerdictTicker";
import type { IncidentFact } from "./types/incident";

type DashboardState = {
  currentFact: IncidentFact | null;
  recentJudgements: string[];
  sessionName: string;
};

export default function Home() {
  const [dashboard, setDashboard] = useState<DashboardState>({
    currentFact: null,
    recentJudgements: [],
    sessionName: "Race Control",
  });
  const handleTelemetryUpdate = useCallback((update: {
    currentFact: IncidentFact | null;
    recentJudgements: string[];
    sessionName: string;
    lastUpdated: string;
  }) => {
    setDashboard((previous) => {
      const sameFact = previous.currentFact?.id === update.currentFact?.id
        && previous.currentFact?.timestamp === update.currentFact?.timestamp
        && previous.currentFact?.ruling === update.currentFact?.ruling;
      const sameSession = previous.sessionName === update.sessionName;
      const sameJudgements = previous.recentJudgements.join("|") === update.recentJudgements.join("|");

      if (sameFact && sameSession && sameJudgements) {
        return previous;
      }

      return {
        currentFact: update.currentFact,
        recentJudgements: update.recentJudgements,
        sessionName: update.sessionName,
      };
    });
  }, []);

  return (
    <div className="relative min-h-screen pb-20">
      <main className="mx-auto grid w-full max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[1.7fr_1fr] lg:px-10">
        <section className="space-y-6">
          <header className="race-panel rounded-2xl border border-white/10 px-5 py-4 shadow-2xl shadow-black/40">
            <p className="text-xs uppercase tracking-[0.24em] text-zinc-400">StewardButBetter</p>
            <h1 className="mt-2 text-3xl font-semibold text-zinc-100">Race Control Interface</h1>
            <p className="mt-2 text-sm text-zinc-300">
              Session: <span className="font-semibold text-[#FF1801]">{dashboard.sessionName}</span>
            </p>
          </header>

          <TelemetryCard onTelemetryUpdate={handleTelemetryUpdate} />
        </section>

        <RulesReference fact={dashboard.currentFact} />
      </main>

      <VerdictTicker
        recentJudgements={dashboard.recentJudgements}
        activeRuling={dashboard.currentFact?.ruling}
      />
    </div>
  );
}
