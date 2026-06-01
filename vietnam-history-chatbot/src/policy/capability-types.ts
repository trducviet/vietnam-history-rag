/**
 * Capability Boundary Types — PATCH 9I
 *
 * Defines the classification buckets, answer policies, evidence quality
 * levels, and upgrade signals for the capability-aware RAG system.
 */

// ─── Capability Bucket ───────────────────────────────────────

/** What category of question/capability this query falls into */
export type CapabilityBucket =
  | 'SAFE_DIRECT_FACT'
  | 'SAFE_DATE_LOOKUP'
  | 'SAFE_ACTOR_LOOKUP'
  | 'SAFE_LOCATION_LOOKUP'
  | 'SAFE_TIMELINE'
  | 'SAFE_COMPARISON'
  | 'SAFE_DISAMBIGUATION'
  | 'HONEST_PARTIAL_REQUIRED'
  | 'LOW_EVIDENCE_FREEFORM'
  | 'NEEDS_CLARIFICATION'
  | 'OUT_OF_SCOPE'
  | 'CORPUS_GAP'
  | 'RETRIEVAL_WEAK';

// ─── Answer Policy ───────────────────────────────────────────

/** How the system should answer given the capability classification */
export type AnswerPolicy =
  | 'FULL_ANSWER'
  | 'FULL_ANSWER_OR_CAUTION'
  | 'HONEST_PARTIAL'
  | 'LOW_EVIDENCE_CAUTION'
  | 'ASK_CLARIFICATION'
  | 'REFUSE_OOS'
  | 'CORPUS_GAP_NOTICE';

// ─── Evidence Quality ────────────────────────────────────────

/** Quality assessment of the retrieved evidence for this query */
export type EvidenceQuality =
  | 'STRONG_DIRECT'
  | 'DIRECT_BUT_NARROW'
  | 'ONE_SIDE_MISSING'
  | 'LOW_RELEVANCE'
  | 'WRONG_SIDE_RISK'
  | 'CORPUS_GAP'
  | 'OUT_OF_SCOPE'
  | 'AMBIGUOUS_QUERY';

// ─── Upgrade Signal ──────────────────────────────────────────

/** What capability upgrade would most improve this query's answer */
export type UpgradeSignal =
  | 'needs_vector_retrieval'
  | 'needs_llm_synthesis'
  | 'needs_corpus_expansion'
  | 'needs_structured_role_relation'
  | 'needs_alias_expansion'
  | 'needs_side_specific_retrieval'
  | 'none';

// ─── Capability Decision ─────────────────────────────────────

/** Complete capability classification for a single query */
export interface CapabilityDecision {
  bucket: CapabilityBucket;
  policy: AnswerPolicy;
  evidenceQuality: EvidenceQuality;
  confidenceCeiling: 'high' | 'medium' | 'low';
  citationsAllowed: boolean;
  citationsRequired: boolean;
  mustAcknowledgeMissingEvidence: boolean;
  upgradeSignals: UpgradeSignal[];
  reasons: string[];
}
