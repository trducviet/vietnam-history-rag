/**
 * Core type definitions for the Vietnamese Historical RAG Chatbot.
 * All types mirror the JSONL schema from vietnam_history_dataset_runtime_optimal_pack.
 */

import type { SemanticFeatures } from '../indexing/semantic-taxonomy.js';

// Re-export SemanticFeatures so consumers can import from types.ts if preferred
export type { SemanticFeatures } from '../indexing/semantic-taxonomy.js';

// ─── Document Types ──────────────────────────────────────────

export type RuntimeDocSource = 'event' | 'synthesis' | 'disambiguation_rule';

/** Shared fields across runtime searchable documents */
export interface BaseDocument {
  doc_id: string;
  doc_source: RuntimeDocSource;
  doc_kind: string;
  title: string;
  summary: string;
  text_for_embedding: string;
  year: number | null;
  end_year: number | null;
  period_id: number | null;
  period_label: string | null;
  doc_type: string;
  event_status: string;
  verification_status: string;
  significance_level: string;
  person_ids: string[];
  people_labels: string[];
  place_ids: string[];
  place_labels: string[];
  organization_ids: string[];
  organization_labels: string[];
  rag_cluster: string | null;
  related_event_ids: string[];
  source_ids: string[];
  canonical: boolean;

  // ── Enriched fields from corpus/events.jsonl (optional, attached at load time) ──

  /** Vietnamese keyword array from retrieval.keywords_vi in full corpus */
  retrieval_keywords_vi?: string[];
  /** Alias list from retrieval.aliases in full corpus */
  aliases?: string[];
  /** Hard-negative sibling IDs from retrieval.hard_negative_ids in full corpus */
  hard_negative_ids?: string[];
  /** Related record IDs from retrieval.related_record_ids in full corpus */
  related_record_ids?: string[];

  /** Benchmark role (e.g. 'primary', 'hard_negative', 'both', 'none') */
  benchmark_role?: string;
  /** Query IDs for which this doc is a target */
  benchmark_target_query_ids?: string[];
  /** Query IDs for which this doc is an acceptable match */
  benchmark_acceptable_query_ids?: string[];
  /** Query IDs for which this doc is a hard negative */
  benchmark_hard_negative_query_ids?: string[];

  /** Category hierarchy from classification in full corpus */
  classification?: {
    category_l1?: string;
    category_l2?: string;
    category_l3?: string;
  };

  /** Semantic features computed at ingest time from semantic-feature-extractor */
  semantic_features?: SemanticFeatures;
}


/** Event document from events_indexable.jsonl */
export interface EventDocument extends BaseDocument {
  doc_source: 'event';
}

/** Synthesis document from synthesis_indexable.jsonl */
export interface SynthesisDocument extends BaseDocument {
  doc_source: 'synthesis';
}

/** Disambiguation rule normalized as a searchable runtime document */
export interface DisambiguationRuleDocument extends BaseDocument {
  doc_source: 'disambiguation_rule';
  rule_id: string;
  confusion_area: string;
  wrong_or_ambiguous_interpretation?: string;
  correct_interpretation?: string;
  recommendation_for_RAG?: string;
  example_user_questions?: string[];
  risk_level?: string;
}

/** Union of all indexable documents */
export type IndexableDocument = EventDocument | SynthesisDocument | DisambiguationRuleDocument;

// ─── Source Types ────────────────────────────────────────────

/** Source/reference document from sources.jsonl */
export interface SourceDocument {
  source_id: string;
  title: string;

  author?: string;
  authors?: string[];

  publisher?: string;
  publication_year?: number;
  year?: number;

  type?: string;
  source_type?: string;

  language?: string;
  reliability_level?: string;
  url?: string;
  page?: string;
  pages?: string;
  citation_text?: string;
  scope_note?: string;
  note?: string;

  [key: string]: unknown;
}

// ─── Link Types ──────────────────────────────────────────────

/** Event → Entity link from event_entity_links.jsonl */
export interface EventEntityLink {
  record_id: string;
  entity_id: string;
  entity_type: 'person' | 'place' | 'organization';
  relation_type: string;
  is_primary: boolean;
  link_quality: 'basic' | 'good' | 'strong';
}

/** Synthesis → Event link from synthesis_event_links.jsonl */
export interface SynthesisEventLink {
  synthesis_id: string;
  event_id: string;
  relation_type: 'synthesizes' | 'profiles' | 'explains' | 'compares' | 'timeline_highlight';
}

/** Runtime provenance link from links_indexable.jsonl */
export interface RuntimeProvenanceLink {
  link_id: string;
  pack_id?: string;
  source_pack_id?: string;
  from_doc_id?: string;
  from_candidate_id?: string;
  to_doc_id?: string | null;
  to_source_id?: string | null;
  link_type: string;
  confidence?: string;
  runtime_merge_status?: string;
  [key: string]: unknown;
}

// ─── Evaluation Types ────────────────────────────────────────

/** QA benchmark entry from qa_benchmark.jsonl */
export interface QABenchmarkEntry {
  query_id: string;
  question: string;
  question_type: string;
  difficulty: 'easy' | 'medium' | 'hard';
  expected_record_ids: string[];
  acceptable_record_ids: string[];
  hard_negative_ids: string[];
  gold_answer: string;
  language: string;
  notes: string;
}

/** Retrieval query from retrieval_queries.jsonl */
export interface RetrievalQuery {
  query_id: string;
  query: string;
  query_type: string;
  difficulty: 'easy' | 'medium' | 'hard';
  target_record_ids: string[];
  acceptable_record_ids: string[];
  hard_negative_ids: string[];
  language: string;
  notes: string;
}

/** Hard negative pair from hard_negatives.jsonl */
export interface HardNegative {
  query_id: string;
  anchor_record_ids: string[];
  negative_record_id: string;
  reason: string;
  question: string;
}

// ─── Vector Store Types ──────────────────────────────────────

/** Metadata stored alongside vectors */
export interface DocMetadata {
  doc_id: string;
  doc_source: RuntimeDocSource;
  doc_type: string;
  title: string;
  year: number | null;
  end_year: number | null;
  period_label: string | null;
  event_status: string;
  verification_status: string;
  canonical: boolean;
}

/** Filter criteria for vector search */
export interface MetadataFilter {
  doc_source?: RuntimeDocSource;
  doc_type?: string;
  year_min?: number;
  year_max?: number;
  period_label?: string;
  event_status?: string;
  verification_status?: string;
  canonical_only?: boolean;
}

/** Single search result from vector store */
export interface SearchResult {
  doc_id: string;
  score: number;
  metadata: DocMetadata;
}

// ─── Routing Types ───────────────────────────────────────────

/** Query intent classification (legacy, used by LLM router output) */
export type QueryIntent =
  | 'fact_lookup'
  | 'date_lookup'
  | 'actor_lookup'
  | 'location_lookup'
  | 'entity_profile'
  | 'explanation'
  | 'comparison'
  | 'timeline'
  | 'cause_effect'
  | 'multi_hop';

// ─── QueryFrame Types ─────────────────────────────────────────

/**
 * Fine-grained intent from QueryFrame Builder (Patch 7D).
 * Superset of QueryIntent — more specific for evidence selection.
 */
export type QueryFrameIntent =
  | 'fact_lookup'
  | 'date_lookup'
  | 'actor_lookup'
  | 'actor_date_lookup'
  | 'location_lookup'
  | 'organization_lookup'
  | 'treaty_lookup'
  | 'clause_lookup'
  | 'conference_lookup'
  | 'significance_lookup'
  | 'campaign_lookup'
  | 'sub_event_lookup'
  | 'movement_lookup'
  | 'explanation'
  | 'cause_effect'
  | 'comparison'
  | 'timeline'
  | 'disambiguation'
  | 'misconception_check'
  | 'out_of_scope';

/**
 * What kind of answer the query expects.
 * Guides EvidenceSelector and Answer Generator framing.
 */
export type ExpectedAnswerType =
  | 'event'
  | 'date'
  | 'actor'
  | 'actor_date'
  | 'location'
  | 'organization'
  | 'treaty'
  | 'clause'
  | 'conference'
  | 'campaign'
  | 'cause'
  | 'meaning'
  | 'comparison'
  | 'timeline'
  | 'yes_no_correction'
  | 'unknown';

/**
 * Semantic focus of a query or contrast clause.
 * Contains only semantic signals — NO document IDs.
 */
export interface QueryFocus {
  topic?: string;
  action?: string;
  actor?: string[];
  object?: string[];
  location?: string[];
  organization?: string[];
  treaty_names?: string[];
  campaign_names?: string[];
  movement_names?: string[];
  time?: {
    year_min?: number;
    year_max?: number;
    explicit_years?: number[];
  };
}

/**
 * Structured semantic frame of a user query.
 *
 * Built by query-frame-builder.ts at routing time.
 * Consumed by EvidenceSelector (Patch 7E) for primary/supporting/contrast role assignment.
 *
 * Invariants:
 * - No document IDs allowed anywhere in this structure.
 * - answer_focus describes what the user is asking FOR.
 * - contrast_focus describes what the user explicitly contrasts AGAINST.
 * - constraints guide evidence selection without hard-coding IDs.
 */
export interface QueryFrame {
  intent: QueryFrameIntent;
  answer_focus: QueryFocus;
  contrast_focus?: QueryFocus;
  expected_answer_type: ExpectedAnswerType;
  constraints?: {
    /** SemanticAction values the primary evidence MUST exhibit */
    must_include_semantics?: string[];
    /** SemanticAction values the primary evidence must NOT be primarily about */
    must_not_be_about?: string[];
  /** Preferred index order for retrieval */
    prefer_index?: RuntimeDocSource[];
    /** Query requires contrast between answer_focus and contrast_focus */
    requires_contrast?: boolean;
    /** Query requires factual correction of a misconception */
    requires_correction?: boolean;
  };
  /** Patch 7K: Extracted comparison sides for two-sided evidence */
  comparison_sides?: {
    side_a: string;
    side_b: string;
    marker?: string;
    /** Patch 9G: Extracted comparison dimension (e.g., "bối cảnh ra đời") */
    comparison_dimension?: string;
  };
  /** Patch 9E: Detected entity profile for collision disambiguation */
  entity_profile?: {
    id: string;
    canonical_name: string;
    expected_year?: number;
    expansion_terms?: string[];
    actor_hints?: string[];
  };
  confidence: 'low' | 'medium' | 'high';
  reasoning: string[];
}

/** Result from query router */
export interface RoutingResult {
  intent: QueryIntent;
  target_indexes: RuntimeDocSource[];
  metadata_filters: MetadataFilter;
  estimated_complexity: 'simple' | 'moderate' | 'complex';
  reasoning: string;
  /** Structured semantic query frame from Patch 7D (optional, backward-compatible) */
  query_frame?: QueryFrame;
}

// ─── Hybrid Search Types ─────────────────────────────────────


/** Result from hybrid search (BM25 + vector combined) */
export interface HybridSearchResult {
  doc_id: string;
  vector_score: number;
  bm25_score: number;
  combined_score: number;
  metadata: DocMetadata;
}

// ─── Reranking Types ─────────────────────────────────────────

/** Result after reranking */
export interface RerankedResult {
  doc_id: string;
  original_score: number;
  rerank_score: number;
  metadata: DocMetadata;
  /** Optional reason annotation (e.g. from hard-negative guard) */
  reason?: string;

  // ── Evidence role annotations (Patch 7E-2, set by pipeline) ──

  /** Evidence role assigned by EvidenceSelector */
  evidence_role?: EvidenceRole;
  /** Confidence score for the assigned evidence role */
  evidence_role_score?: number;
  /** Human-readable reason codes for role assignment */
  evidence_reasons?: string[];
  /** Rank within evidence role group (0-based) */
  evidence_rank?: number;
}

// ─── Context Bundle Types ────────────────────────────────────

/** Disambiguation note to prevent generator confusion */
export interface DisambiguationNote {
  doc_id: string;
  title: string;
  risk_type:
    | 'hard_negative'
    | 'same_cluster'
    | 'temporal_conflict'
    | 'doc_type_conflict'
    | 'planned_not_executed'
    | 'near_duplicate';
  reason: string;
}

/** Citation role for a document in the context bundle */
export type CitationRole = 'primary' | 'supporting' | 'contrast' | 'background' | 'excluded';

/** Citation role assignment for context bundle (Patch 7F: extended with role/reason/priority) */
export interface CitationPlanItem {
  doc_id: string;
  /** Evidence-aware citation role */
  citation_role: CitationRole;
  source_ids: string[];
  /** Why this role was assigned */
  reason?: string;
  /** Lower = higher priority in citation list */
  priority?: number;
}

/** A curated context bundle ready for LLM consumption */
export interface ContextBundle {
  intent: QueryIntent;
  primary_docs: IndexableDocument[];
  supporting_docs: IndexableDocument[];
  /** Documents flagged as planned_not_executed */
  planned_not_executed_docs: IndexableDocument[];
  /** Assembled context text for the LLM prompt */
  context_text: string;
  /** Traceability: which doc_ids are in the bundle */
  included_doc_ids: string[];
  /** Warnings about data quality or gaps */
  warnings: string[];
  /** Disambiguation notes to avoid confusion */
  disambiguation_notes?: DisambiguationNote[];
  /** Citation plan for generator guidance */
  citation_plan?: CitationPlanItem[];
  /** Doc IDs omitted from citation context (contrast/excluded) */
  omitted_doc_ids?: string[];
  /** Evidence-specific diagnostic warnings */
  evidence_warnings?: string[];
}

// ─── Confidence & Response Types ─────────────────────────────

/** Signals used to compute confidence level */
export interface ConfidenceSignals {
  /** Gap between top-1 and top-2 retrieval scores, normalized (0=no gap, 1=clear winner) */
  retrievalScoreGap: number;
  /** Ratio of verified docs in the context bundle (0–1) */
  verifiedRatio: number;
  /** Score measuring ambiguity among top candidates (0=clear, 1=ambiguous) */
  ambiguityScore: number;
  /** Risk of hard-negative confusion, query-specific (0=no risk, 1=high risk) */
  hardNegativeRisk: number;
  /** @deprecated Kept for backward compat. Always 1.0 since context enforces canonical. */
  canonicalRatio?: number;
}

export type ConfidenceLevel = 'high' | 'medium' | 'low';

// ─── Citation Types ──────────────────────────────────────────

/** Enriched citation with provenance */
export interface Citation {
  record_id: string;
  title: string;
  relevance: string;

  doc_source?: RuntimeDocSource;
  doc_type?: string;
  year?: number | null;

  source_ids?: string[];
  sources?: SourceDocument[];
}

/** Stage 8B3: optional no-API answer-focus/template metadata. */
export interface AnswerFocusMetadata {
  answer_focus_available: boolean;
  intent_result?: unknown;
  answer_plan?: unknown;
  template_id?: string;
  rendered_template_preview?: unknown;
  focus_check_result?: unknown;
  answer_status?: string;
  citation_policy_satisfied?: boolean;
  rule_context_used?: boolean;
  context_weak_warning?: boolean;
  should_ask_clarification?: boolean;
  should_abstain?: boolean;
  focus_check_available?: boolean;
  warning?: string;
  fallback?: string;
}

/** Stage 8C3: optional session-memory/follow-up rewrite metadata. */
export interface AnswerMemoryMetadata {
  session_id: string;
  memory_available: boolean;
  memory_used: boolean;
  rewrite_used: boolean;
  original_query: string;
  effective_query: string;
  warning?: string;
  rewrite?: {
    rewrite_status: string;
    rewritten_query: string;
    rewrite_confidence: number;
    rewrite_reason: string;
    used_memory_fields: string[];
    safety_flags: string[];
  };
  resolution?: {
    resolution_status: string;
    referent_text: string;
    referent_type: string;
    confidence: number;
    should_ask_clarification: boolean;
    clarification_question: string;
  };
  memory_update?: {
    updated: boolean;
    write_allowed: boolean;
    write_block_reason: string;
    active_topic_after: string | null;
    active_entities_after: string[];
  };
  safety?: {
    blocked_by_safety: boolean;
    memory_conflict: boolean;
    negative_gap_protected: boolean;
    out_of_scope_protected: boolean;
  };
  session_memory_state?: unknown;
}

/** Final chat response to the user */
export interface ChatResponse {
  answer: string;
  explanation: string;
  citations: Citation[];
  confidence: ConfidenceLevel;
  confidence_details: ConfidenceSignals;
  related_events: { record_id: string; title: string }[];
  /** Optional metadata namespace. Existing clients may ignore it safely. */
  metadata?: {
    answer_focus?: AnswerFocusMetadata;
    memory?: AnswerMemoryMetadata;
    [key: string]: unknown;
  };
  /** Patch 8D-B: Optional debug trace for diagnostic scripts (not populated in production) */
  _debugTrace?: {
    routing: { intent: string; indexes: string[]; retrieval_query: string };
    retrieval_doc_ids: string[];
    evidence_primary_ids: string[];
    evidence_supporting_ids: string[];
    context_primary_ids: string[];
    context_supporting_ids: string[];
    citation_plan_ids: string[];
    vector_rescue?: { side: string; accepted: string[]; rejected: string[] };
  };
  /** Patch 9I: Capability boundary classification */
  _capabilityDecision?: import('../policy/capability-types.js').CapabilityDecision;
}

// ─── Answer Verification (Patch 7G) ─────────────────────────

/** Single issue found by deterministic answer verifier */
export interface AnswerVerificationIssue {
  code: string;
  severity: 'info' | 'warning' | 'error';
  message: string;
}

/** Result from deterministic answer verification */
export interface AnswerVerificationResult {
  ok: boolean;
  issues: AnswerVerificationIssue[];
  revised_answer?: string;
  revised_explanation?: string;
  /** 7L-D: True if verifier determined context has insufficient focus evidence */
  insufficient_evidence?: boolean;
  /** 7L-D: Citation policy when insufficient_evidence is true */
  citation_policy?: 'clear' | 'focus_positive_only';
  /** 7N-C: Years to filter from citations in comparison answers */
  comparison_noise_years?: string[];
}

// ─── Loaded Dataset ──────────────────────────────────────────

/** Complete loaded dataset in memory */
export interface LoadedDataset {
  events: Map<string, EventDocument>;
  synthesis: Map<string, SynthesisDocument>;
  disambiguationRules: Map<string, DisambiguationRuleDocument>;
  canonicalEvents: EventDocument[];
  canonicalSynthesis: SynthesisDocument[];
  canonicalDisambiguationRules: DisambiguationRuleDocument[];
  allCanonicalDocs: IndexableDocument[];
  eventEntityLinks: EventEntityLink[];
  synthesisEventLinks: SynthesisEventLink[];
  qaBenchmark: QABenchmarkEntry[];
  retrievalQueries: RetrievalQuery[];
  hardNegatives: HardNegative[];
  /** Source documents for citation provenance */
  sources: Map<string, SourceDocument>;
  /** Runtime links/provenance layer loaded from links_indexable.jsonl */
  runtimeLinks: RuntimeProvenanceLink[];
  linksByFromDocId: Map<string, RuntimeProvenanceLink[]>;
  linksByToSourceId: Map<string, RuntimeProvenanceLink[]>;
  /** Lookup: entity_id → event records linked to it */
  entityToEvents: Map<string, string[]>;
  /** Lookup: synthesis_id → linked event_ids */
  synthesisToEvents: Map<string, string[]>;
  /** Lookup: event_id → synthesis_ids that reference it */
  eventToSynthesis: Map<string, string[]>;
}

// ─── Scope Guard Types (Patch 7J) ────────────────────────────

/** Decision from pre-retrieval scope and ambiguity guard */
export type ScopeGuardDecision = 'in_scope' | 'out_of_scope' | 'needs_clarification';

/** Result of scope/ambiguity evaluation */
export interface ScopeGuardResult {
  decision: ScopeGuardDecision;
  reason?: string;
  confidence: 'high' | 'medium' | 'low';
  matched_patterns?: string[];
}

// ─── Evidence Selection Types (Patch 7E) ─────────────────────

/** Evidence role assigned by EvidenceSelector — query-dependent */
export type EvidenceRole = 'primary' | 'supporting' | 'contrast' | 'excluded';

/** Reason for an evidence role assignment */
export interface EvidenceSelectionReason {
  code: string;
  message: string;
  weight?: number;
}

/** Single evidence item with role annotation */
export interface EvidenceItem {
  doc_id: string;
  role: EvidenceRole;
  role_score: number;
  reasons: EvidenceSelectionReason[];
  /** Original rerank_score for ordering within role */
  rerank_score: number;
}

/** Complete evidence selection result */
export interface EvidenceSelection {
  primary: EvidenceItem[];
  supporting: EvidenceItem[];
  contrast: EvidenceItem[];
  excluded: EvidenceItem[];
  /** Ordered list: primary → supporting → contrast (excluded omitted) */
  ordered: EvidenceItem[];
  diagnostics: {
    used_query_frame: boolean;
    primary_count: number;
    supporting_count: number;
    contrast_count: number;
    excluded_count: number;
    warnings: string[];
  };
}
