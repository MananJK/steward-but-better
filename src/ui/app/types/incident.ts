export type IncidentFact = {
  id: string;
  timestamp: string;
  lap: number;
  driver: string;
  speedKph: number;
  deltaToLeader: number;
  trackTempC: number;
  sector: string;
  incident: string;
  confidence: number;
  fiaArticle: string;
  ruleSummary: string;
  ruling: string;
};

export type IncidentFactsPayload = {
  session: string;
  lastUpdated: string;
  aiVerdicts: string[];
  facts: IncidentFact[];
};
