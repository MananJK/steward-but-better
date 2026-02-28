"use client";

import { useState } from "react";
import RulesReference from "./components/RulesReference";
import TelemetryCard from "./components/TelemetryCard";
import VerdictTicker from "./components/VerdictTicker";
import type { IncidentFact } from "./types/incident";

type DashboardState = {
  currentFact: IncidentFact | null;
  verdicts: string[];
  session: string;
};

export default function Home() {
  const [dashboard, setDashboard] = useState<DashboardState>({
    currentFact: null,
    verdicts: [],
    session: "Race Control",
  });

  return (
    <div className="relative min-h-screen pb-20">
      <main className="mx-auto grid w-full max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[1.7fr_1fr] lg:px-10">
        <section className="space-y-6">
          <header className="race-panel rounded-2xl border border-white/10 px-5 py-4 shadow-2xl shadow-black/40">
            <p className="text-xs uppercase tracking-[0.24em] text-zinc-400">StewardButSmarter</p>
            <h1 className="mt-2 text-3xl font-semibold text-zinc-100">Race Control Interface</h1>
            <p className="mt-2 text-sm text-zinc-300">
              Session: <span className="font-semibold text-[#FF1801]">{dashboard.session}</span>
            </p>
          </header>

          <TelemetryCard
            onTelemetryUpdate={(update) =>
              setDashboard({
                currentFact: update.currentFact,
                verdicts: update.verdicts,
                session: update.session,
              })
            }
          />
        </section>

        <RulesReference fact={dashboard.currentFact} />
      </main>

      <VerdictTicker verdicts={dashboard.verdicts} activeRuling={dashboard.currentFact?.ruling} />
    </div>
  );
}
