/**
 * Query Normalization Types — Stage 12C
 *
 * Defines shared types for the query normalization / historical entity
 * canonicalization layer that runs before memory/follow-up rewrite,
 * answer focus and hybrid retrieval.
 */

// ─── Entity Dictionary ────────────────────────────────────────

export type EntityType =
  | 'event'
  | 'person'
  | 'organization'
  | 'document'
  | 'date'
  | 'topic'
  | 'alias';

export interface HistoricalEntityEntry {
  canonical: string;
  type: EntityType;
  aliases: string[];
  no_accent: string;
  no_accent_aliases?: string[];
  keywords?: string[];
  confidence_boost?: number;
  notes?: string;
}

// ─── Match Result ─────────────────────────────────────────────

export interface EntityMatch {
  canonical: string;
  type: EntityType;
  matched_alias: string;
  match_method: 'exact_canonical' | 'no_accent_exact' | 'alias_exact' | 'fuzzy_token' | 'date_alias';
  confidence: number;
  evidence_score?: number;
}

// ─── Normalization Status ─────────────────────────────────────

export type NormalizationStatus =
  | 'canonicalized'
  | 'candidate_augmented'
  | 'unchanged_low_confidence'
  | 'ambiguous_needs_clarification'
  | 'oos_unchanged'
  | 'followup_passthrough';

// ─── Normalization Result ─────────────────────────────────────

export interface QueryNormalizationResult {
  /** User's raw input query */
  original_query: string;
  /** Cleaned query: lowercased, collapsed spaces, punctuation trimmed */
  normalized_query: string;
  /** No-accent form of the normalized query */
  no_accent_query: string;
  /** Best canonical query — equal to original_query if no rewrite */
  canonical_query: string;
  /** Query to send to retrieval (may be augmented) */
  retrieval_query: string;
  /** Whether a canonical rewrite was applied */
  normalization_status: NormalizationStatus;
  rewrite_applied: boolean;
  confidence: number;
  evidence_score?: number;
  hard_canonicalize_allowed?: boolean;
  canonical_candidate?: string;
  ambiguity_reason?: string;
  /** Entities matched against the historical entity dictionary */
  matched_entities: EntityMatch[];
  /** Competing candidates if ambiguous */
  ambiguous_candidates: EntityMatch[];
  /** Latency in milliseconds for normalization */
  latency_ms: number;
  warnings: string[];
}
