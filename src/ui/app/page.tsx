"use client";

import DriverGrid from "./components/DriverGrid";
import IncidentLog from "./components/IncidentLog";

export default function Home() {
  return (
    <div className="relative min-h-screen">
      <aside className="fixed right-0 top-0 h-screen w-[30%] min-w-[360px] p-4">
        <IncidentLog />
      </aside>

      <main className="w-[70%] px-4 py-6 sm:px-6 lg:px-10">
        <header className="race-panel mb-6 rounded-2xl border border-white/10 px-5 py-4 shadow-2xl shadow-black/40">
          <p className="text-xs uppercase tracking-[0.24em] text-zinc-400">StewardButBetter</p>
          <h1 className="mt-2 text-3xl font-semibold text-zinc-100">Race Control Grid</h1>
          <p className="mt-2 text-sm text-zinc-300">
            Fixed 24-cell driver matrix with live speed, rank, and incident highlighting.
          </p>
        </header>

        <DriverGrid />
      </main>
    </div>
  );
}
