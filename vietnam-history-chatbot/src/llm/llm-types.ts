export type LLMBackendMode = 'none' | 'local' | 'cloud';

export type LLMAnswerMode =
  | 'source_grounded_answer'
  | 'safe_clarification'
  | 'safe_out_of_scope'
  | 'safe_insufficient_data'
  | 'safe_refusal';

export interface RetrievedContextItem {
  doc_id: string;
  source_id: string;
  title: string;
  snippet: string;
  period?: string | null;
  confidence?: 'high' | 'medium' | 'low' | string;
}

export interface CitationPayloadItem {
  marker: string;
  source_id: string;
  title: string;
  supports: string;
}

export interface FinalAnswerInputPackage {
  user_query: string;
  rewritten_query?: string | null;
  intent: string;
  answerability: string;
  safe_mode: string;
  answer_plan?: Record<string, unknown>;
  focus_goal?: string;
  memory_context?: {
    active_topic?: string | null;
    active_entities?: string[];
    active_period?: string | null;
    resolved_references?: string[];
  };
  retrieved_context: RetrievedContextItem[];
  citation_payload: CitationPayloadItem[];
  response_template?: string;
  safety_instructions?: string[];
  forbidden_behaviors?: string[];
}

export interface LLMAnswerOutput {
  answer_text: string;
  citations_used: string[];
  answer_mode: LLMAnswerMode;
  unsupported_claims_self_check: string[];
  confidence: 'high' | 'medium' | 'low';
  needs_more_context: boolean;
  raw_response?: unknown;
}

export interface LLMHealthCheckResult {
  checked: boolean;
  base_url: string;
  is_localhost: boolean;
  server_reachable: boolean;
  model_detected: boolean;
  model_name: string;
  local_test_generation_run: boolean;
  local_test_generation_passed: boolean;
  error?: string;
  notes: string[];
}

export interface LLMProvider {
  name: string;
  mode: LLMBackendMode;
  generate(input: FinalAnswerInputPackage): Promise<LLMAnswerOutput>;
  healthCheck?(): Promise<LLMHealthCheckResult>;
}
