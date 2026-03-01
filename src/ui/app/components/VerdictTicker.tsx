import { memo, useMemo } from "react";

type VerdictTickerProps = {
  recentJudgements: string[];
  activeRuling?: string;
};

const DEFAULT_VERDICTS = [
  "Awaiting Steward AI rulings from incident feed.",
  "Telemetry pipeline online.",
];

function VerdictTickerBase({ recentJudgements, activeRuling }: VerdictTickerProps) {
  const tickerContent = useMemo(() => {
    const stream = [...(recentJudgements.length ? recentJudgements : DEFAULT_VERDICTS)];
    if (activeRuling) {
      stream.unshift(`Live: ${activeRuling}`);
    }
    return stream.join("  |  ");
  }, [activeRuling, recentJudgements]);

  const lane = (
    <div className="verdict-ticker__lane whitespace-nowrap text-sm font-semibold uppercase tracking-[0.16em] text-zinc-200">
      <span className="mr-10 text-[#FF1801]">Verdict Wire</span>
      <span>{tickerContent}</span>
    </div>
  );

  return (
    <div className="fixed inset-x-0 bottom-0 z-30 border-t border-[#FF1801]/40 bg-black/90 backdrop-blur">
      <div className="relative overflow-hidden py-3">
        <div className="verdict-ticker__track">
          {lane}
          <div className="verdict-ticker__lane whitespace-nowrap text-sm font-semibold uppercase tracking-[0.16em] text-zinc-200" aria-hidden="true">
            <span className="mr-10 text-[#FF1801]">Verdict Wire</span>
            <span>{tickerContent}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

const VerdictTicker = memo(
  VerdictTickerBase,
  (previous, next) =>
    previous.activeRuling === next.activeRuling
    && previous.recentJudgements.join("|") === next.recentJudgements.join("|")
);

export default VerdictTicker;
