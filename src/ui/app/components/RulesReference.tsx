import type { IncidentFact } from "../types/incident";

type RulesReferenceProps = {
  fact: IncidentFact | null;
};

export default function RulesReference({ fact }: RulesReferenceProps) {
  const hasRuleSummary = Boolean(fact?.ruleSummary?.trim());
  const isPenalty = fact?.ruling?.trim().toUpperCase() === "PENALTY";

  return (
    <aside className="race-panel h-full rounded-2xl border border-white/10 p-5 shadow-2xl shadow-black/40">
      <p className="text-xs uppercase tracking-[0.22em] text-zinc-400">Rules Reference</p>
      <h2 className="mt-2 text-xl font-semibold text-zinc-100">FIA Article</h2>

      <div className="mt-5 rounded-xl border border-[#FF1801]/40 bg-[#FF1801]/10 p-4">
        <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-400">Retrieved Citation</p>
        <p className="mt-2 text-2xl font-semibold text-[#FF1801]">{fact?.fiaArticle ?? "Awaiting Article"}</p>
      </div>

      <div className="mt-5 space-y-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">Applicable Rule</p>
          <p className="mt-2 text-sm leading-relaxed text-zinc-300">
            {fact
              ? hasRuleSummary
                ? fact.ruleSummary
                : "Retrieving from FIA Database..."
              : "No rule context available yet. Telemetry analysis is still in progress."}
          </p>
        </div>
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">AI Ruling Draft</p>
          <p className={`mt-2 text-sm leading-relaxed text-zinc-200 ${isPenalty ? "penalty-glow" : ""}`}>
            {fact?.ruling ?? "Pending"}
          </p>
        </div>
      </div>
    </aside>
  );
}
