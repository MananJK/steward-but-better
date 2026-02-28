export type IncidentFact = {
  id: string;
  timestamp: string;
  lap: number;
  driver: string;
  incidentType: string;
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

export type LiveIncidentPayload = {
  id?: string;
  sessionName?: string;
  track?: string;
  timestamp?: string;
  lastUpdated?: string;
  driver?: string;
  incident_type?: string;
  incident_description?: string;
  speed_kph?: number;
  apex_gap?: number;
  delta_to_leader?: number;
  track_temp_c?: number;
  sector?: string;
  lap?: number;
  article_cited?: string;
  confidence_score?: number;
  rule_summary?: string;
  ruling?: string;
  verdict?: string;
  recentJudgements?: string[];
};
