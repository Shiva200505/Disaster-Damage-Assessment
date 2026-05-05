export interface InferenceRequest {
  disaster_name: string;
  map_threshold: number;
}

export interface DamageStats {
  total_buildings: number;
  no_damage: number;
  minor_damage: number;
  major_damage: number;
  destroyed: number;
}

export interface InferenceStatus {
  task_id: string;
  status: "PENDING" | "PROGRESS" | "SUCCESS" | "FAILURE";
  step?: string;
  progress_pct?: number;
  created_at: string;
}

export interface InferenceResult {
  task_id: string;
  status: string;
  map_url: string;
  geojson_url: string;
  stats: DamageStats;
  processing_time_seconds: number;
  error?: string;
}
