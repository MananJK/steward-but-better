export type IncidentFact = {
  id: string;
  timestamp: string;
  lap: number;
  driver: string;
  incidentType: string;
  lateralG: number;
  speedKph: number;
  deltaToLeader: number;
  trackTempC: number;
  sector: string;
  incident: string;
  confidence: number;
  fiaArticle: string;
  ruleSummary: string;
  ruling: string;
  triggerSteward: boolean;
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
  lateral_g?: number;
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
  trigger_steward?: boolean;
  recentJudgements?: string[];
};

export type ActiveInvestigation = {
  id: string;
  timestamp: string;
  driver: string;
  lap: number;
  incident_type: string;
  incident_description: string;
  speed_kph: number;
  lateral_g: number;
  rule_summary: string;
  ruling: string;
  confidence_score: number;
  article_cited: string;
};
