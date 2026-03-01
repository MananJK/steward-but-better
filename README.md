# STEWARD BUT BETTER
## *Because "We'll Look Into It" Isn't a Strategy*

---

## The Problem with Race Control

Real stewards have milliseconds to decide. They get replays. Coffee breaks. Committee debates. Meanwhile, drivers are doing 320 km/h into Copse wondering if that move was worth a 5-second penalty.

**We fixed the latency.**

---

## Physics Doesn't Lie (Unlike Your Favorite Driver)

Every incident starts with the same thing: metal, rubber, and *a lot* of force. We measure the G-loads that make your breakfast reconsider its life choices.

- **250Hz telemetry ingestion** - Because 60 FPS is for sim racers
- **Sub-80ms latency** - Faster than a Red Bull pit stop
- **5-second de-duplication** - Separating genuine incidents from "eh, that's just racing"

The car can't gaslight you. The data is the data.

---

## 3,725 Articles of Why You're Guilty

Behind every decision is a neural engine with a photographic memory for the rulebook.

**3,725 indexed chunks** of the 2025 FIA Sporting and Technical Regulations, vectorized in FAISS and queried via Mistral 7B. This isn't a language model winging it—it's a legal database with an opinion.

The RAG engine doesn't *interpret*. It **cites**.

> *"Driver exceeded track limits at Turn 4, gaining lasting advantage per Article 33.3. 5-second penalty. No appeal. Next caller."*

---

## The Stack: Engineered for Chaos

### Next.js 14 Dashboard
Atomic state persistence. File-locking that survives red flags. Sub-second telemetry bursts handled without corrupting your logs.

When the Safety Car comes out and incidents pile up like cars at the start of Spa, **the system keeps running.**

### Tested. Stress-Tested. Basically Free.

**5,159 inference requests** across endurance simulations. Cost? Pocket change.

This is what happens when you optimize for lap time, not cloud bills.

---

---

## Getting Started

```bash
# Fire up the telemetry engine
cd src/telemetry && python f1_monitor.py --simulate abu_dhabi_2021_lap58

# Launch the control center
cd src/ui && npm run dev
```

That's it. A championship-caliber steward on your local machine. No DRS required.

---

## What's Next on the Grid

- **Multi-Agent Steward Panels**: Technical, Sporting, Safety—debating incidents in real-time (less arguing than the actual stewards, we promise)
- **Precedent Engine**: 10 years of case law for consistent rulings
- **Broadcast Integration**: Live probability heatmaps so viewers can argue with *data*

We're not just building a tool. We're making sure the sport gets decisions as precise as the engineering.

---

## License

MIT — Open source for anyone who believes racing deserves better than "we're checking."

Commercial licensing available for FIA-accredited partners ready to admit they need the upgrade.

---

*Built by an engineer who knows that at 370 km/h, there's no time for "we'll get back to you."*
