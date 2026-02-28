import type { IncidentFact } from "../types/incident";

type RulesReferenceProps = {
  fact: IncidentFact | null;
};

export default function RulesReference({ fact }: RulesReferenceProps) {
  const hasRuleSummary = Boolean(fact?.ruleSummary?.trim());
  const highGThresholdHit = (fact?.lateralG ?? 0) >= 3.5;
  const isPenalty = fact?.ruling?.trim().toUpperCase() === "PENALTY";
  const isReviewing = fact?.triggerSteward === true && highGThresholdHit;
  const isIdle = !fact || fact.triggerSteward === false || !highGThresholdHit;

  return (
    <aside className="race-panel h-full rounded-2xl border border-white/10 p-5 shadow-2xl shadow-black/40">
      <p className="text-xs uppercase tracking-[0.22em] text-zinc-400">Rules Reference</p>
      <h2 className="mt-2 text-xl font-semibold text-zinc-100">FIA Article</h2>

      <div
        className={`mt-5 rounded-xl border p-4 ${
          highGThresholdHit
            ? "border-[#FF1801]/40 bg-[#FF1801]/10"
            : "border-white/10 bg-zinc-900/50"
        }`}
      >
        <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-400">
          {highGThresholdHit ? "Retrieved Citation" : "Monitoring Status"}
        </p>
        <p className={`mt-2 text-2xl font-semibold ${highGThresholdHit ? "text-[#FF1801]" : "text-zinc-200"}`}>
          {highGThresholdHit ? "Article 33.4" : "Standby"}
        </p>
      </div>

      <div className="mt-5 space-y-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">Applicable Rule</p>
          <p className="mt-2 text-sm leading-relaxed text-zinc-300">
            {highGThresholdHit && fact
              ? hasRuleSummary
                ? fact.ruleSummary
                : "Retrieving from FIA Database..."
              : "SCANNING FIA ARTICLES..."}
          </p>
        </div>
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">AI Ruling Draft</p>
          {isIdle ? (
            <div className="mt-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.12em] text-emerald-300">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              SCANNING FIA ARTICLES...
            </div>
          ) : null}
          {isReviewing ? (
            <p className="mt-2 animate-pulse text-sm font-semibold uppercase tracking-[0.12em] text-yellow-300">
              STEWARD REVIEW IN PROGRESS
            </p>
          ) : (
            <p className={`mt-2 text-sm leading-relaxed text-zinc-200 ${isPenalty ? "penalty-glow" : ""}`}>
              {highGThresholdHit ? fact?.ruling ?? "Pending" : "Awaiting threshold trigger"}
            </p>
          )}
        </div>
      </div>
    </aside>
  );
}
