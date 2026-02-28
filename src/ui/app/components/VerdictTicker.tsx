type VerdictTickerProps = {
  verdicts: string[];
  activeRuling?: string;
};

const DEFAULT_VERDICTS = [
  "Awaiting Steward AI rulings from incident feed.",
  "Telemetry pipeline online.",
];

export default function VerdictTicker({ verdicts, activeRuling }: VerdictTickerProps) {
  const stream = [...(verdicts.length ? verdicts : DEFAULT_VERDICTS)];
  if (activeRuling) {
    stream.unshift(`Live: ${activeRuling}`);
  }

  const tickerContent = stream.join("  |  ");

  return (
    <div className="fixed inset-x-0 bottom-0 z-30 border-t border-[#FF1801]/40 bg-black/90 backdrop-blur">
      <div className="relative overflow-hidden py-3">
        <div className="ticker-track whitespace-nowrap text-sm font-semibold uppercase tracking-[0.16em] text-zinc-200">
          <span className="mr-10 text-[#FF1801]">Verdict Wire</span>
          <span>{tickerContent}</span>
          <span className="ml-10 text-[#FF1801]">Verdict Wire</span>
          <span className="ml-10">{tickerContent}</span>
        </div>
      </div>
    </div>
  );
}
