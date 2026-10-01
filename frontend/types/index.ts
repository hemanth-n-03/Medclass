export interface SimilarQuestion {
  question: string;
  similarity_score: number;
  answer_snippet?: string;
  source?: string;
  source_organization?: string;
  focus_area?: string;
}

export interface EmergencyInfo {
  is_emergency: boolean;
  safety_level?: string;
  warning?: string | null;
  matched_indicators: string[];
}

export interface ExplanationInfo {
  explanation_text: string;
  salient_tokens: string[];
  methodology: string;
}

export interface RoutingInfo {
  recommended_specialist: string;
  clinical_department: string;
  suggested_action: string;
}

export interface SecondaryCandidate {
  category?: string;
  intent?: string;
  confidence: number;
  reason?: string;
}

export interface ClinicalSignal {
  entity: string;
  domain: string;
  signal: string;
}

export interface QueryCorrection {
  original: string;
  replacement: string;
  type: string;
}

export interface ClassifyResponse {
  question: string;
  normalized_question?: string;
  corrections?: QueryCorrection[];
  is_medical: boolean;
  domain_confidence: number;
  message?: string | null;

  // Primary Category & Rationale
  category?: string;
  category_confidence: number;
  primary_category?: string;
  primary_category_confidence?: number;
  primary_category_reason?: string | null;
  secondary_categories: SecondaryCandidate[];

  // Intent
  intent?: string;
  intent_confidence: number;
  secondary_intents: SecondaryCandidate[];

  // Confidence & Uncertainty
  overall_confidence: number;
  uncertain: boolean;
  uncertainty_reason?: string | null;
  clarification_needed?: string | null;

  // Multi-symptom & Entity Recognition
  is_multi_symptom: boolean;
  detected_entities: string[];
  clinical_signals: ClinicalSignal[];

  // Safety, Routing & Guidance
  safety_level?: string;
  suggested_next_steps: string[];
  explanation?: ExplanationInfo | null;
  similar_questions: SimilarQuestion[];
  routing?: RoutingInfo | null;
  emergency?: EmergencyInfo | null;
  safety_disclaimer: string;

  // Localization
  timestamp_ist?: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  model_loaded: boolean;
  device: string;
  supported_categories: number;
  supported_intents: number;
}

export interface AnalyticsSummary {
  total_queries: number;
  medical_queries: number;
  non_medical_queries: number;
  uncertain_queries: number;
  emergency_queries: number;
  avg_confidence: number;
  category_distribution: Record<string, number>;
  intent_distribution: Record<string, number>;
  recent_activity: Array<{
    timestamp: string;
    timestamp_ist?: string;
    question_preview: string;
    is_medical: boolean;
    category?: string;
    primary_category?: string;
    is_multi_symptom?: boolean;
    intent?: string;
    confidence: number;
    uncertain: boolean;
    emergency: boolean;
    safety_level?: string;
  }>;
}

export interface ModelInfo {
  model_name: string;
  backbone: string;
  categories: string[];
  intents: string[];
  benchmarks: {
    summary: string;
    models: Record<string, any>;
    chart_data: Array<{
      task: string;
      Baseline_SVM: number;
      BiLSTM: number;
      Transformer: number;
    }>;
  };
  dataset_summary: {
    source: string;
    total_records: number;
    train_samples: number;
    val_samples: number;
    test_samples: number;
    license: string;
  };
}
