/**
 * Semantic Feature Extractor (Patch 7C-1B)
 *
 * Patch 7C-1B change: Separate primary semantic text from contextual text.
 *
 * Primary actions must come from title/summary-level semantics;
 * contextual text only contributes topics.
 *
 * Rules:
 * - NO hard-coded EVT IDs.
 * - NO evidence role (primary/supporting/contrast) — query-dependent, EvidenceSelector.
 * - NO is_hard_negative — query-dependent, HardNegativeGuard.
 * - All phrase matching: normalizeVietnameseText + hasAnyPhrase (no regex \b).
 * - treaty_related ≠ treaty_signing (requires explicit signing verb in PRIMARY text).
 * - withdrawal_or_evacuation is NOT treaty_signing even if context mentions Paris.
 * - accession requires named international org phrase in PRIMARY text.
 *
 * Two-layer text architecture:
 *   primarySemanticText  = title + summary + metadata labels
 *                          → drives: actions, primary flags
 *   contextualSemanticText = title + summary + text_for_embedding + keywords + aliases
 *                            → drives: topics, treaty/campaign/movement names,
 *                              context-level flags (is_treaty_related)
 */

import type { SemanticFeatures, SemanticDomain, SemanticAction, AnswerAffordance } from './semantic-taxonomy.js';
import {
  normalizeVietnameseText,
  hasAnyPhrase,
  CONTROLLED_PHRASES,
  DOC_TYPE_TO_DOMAIN_MAP,
  DOC_KIND_TO_AFFORDANCE_MAP,
} from './semantic-taxonomy.js';

// ─── Input Shape ──────────────────────────────────────────────────────────────

export interface FeatureExtractorInput {
  doc_id?: string;
  title: string;
  summary?: string;
  text_for_embedding?: string;
  doc_type?: string;
  doc_kind?: string;
  doc_source?: 'event' | 'synthesis' | string;
  event_status?: string;
  year?: number | null;
  end_year?: number | null;
  period_label?: string | null;
  verification_status?: string;
  significance_level?: string;
  source_ids?: string[];
  retrieval_keywords_vi?: string[];
  aliases?: string[];
  classification?: {
    category_l1?: string;
    category_l2?: string;
    category_l3?: string;
  };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * PRIMARY semantic text.
 * Used to determine the document's own primary actions.
 *
 * Sources: title + summary + doc_type/doc_kind/classification labels ONLY.
 * Excluded: text_for_embedding, retrieval_keywords_vi, aliases.
 *
 * Rationale: text_for_embedding and keyword lists often include contextual
 * mentions of related events. Those mentions must NOT promote an action to
 * "primary" for this document.
 */
function buildPrimarySemanticText(input: FeatureExtractorInput): string {
  const parts: string[] = [];
  if (input.title) parts.push(input.title);
  if (input.summary) parts.push(input.summary);
  // Include classification labels as lightweight metadata hints
  if (input.classification?.category_l2) parts.push(input.classification.category_l2);
  if (input.classification?.category_l3) parts.push(input.classification.category_l3);
  if (input.doc_type) parts.push(input.doc_type);
  if (input.doc_kind) parts.push(input.doc_kind);
  return normalizeVietnameseText(parts.join(' '));
}

/**
 * CONTEXTUAL semantic text.
 * Used for topics, treaty/campaign/movement names, and broad topic flags.
 *
 * Sources: all available text including embedding text, keywords, aliases.
 * These contribute context signals but do NOT drive primary action extraction.
 */
function buildContextualSemanticText(input: FeatureExtractorInput): string {
  const parts: string[] = [];
  if (input.title) parts.push(input.title);
  if (input.summary) parts.push(input.summary);
  if (input.text_for_embedding) parts.push(input.text_for_embedding);
  if (input.retrieval_keywords_vi?.length) {
    parts.push(input.retrieval_keywords_vi.join(' '));
  }
  if (input.aliases?.length) {
    parts.push(input.aliases.join(' '));
  }
  return normalizeVietnameseText(parts.join(' '));
}

/** Deduplicate an array while preserving order */
function dedup<T>(arr: T[]): T[] {
  return [...new Set(arr)];
}

// ─── Domain Extraction ────────────────────────────────────────────────────────

function extractDomains(input: FeatureExtractorInput): SemanticDomain[] {
  const domains: SemanticDomain[] = [];

  // Priority 1: classification.category_l1 (from full corpus enrichment)
  const l1 = input.classification?.category_l1?.toLowerCase();
  if (l1) {
    const l1Map: Record<string, SemanticDomain> = {
      military: 'military',
      diplomacy: 'diplomacy',
      politics: 'politics',
      administration: 'administration',
      economy: 'economy',
      society: 'society',
    };
    const mapped = l1Map[l1];
    if (mapped) domains.push(mapped);
  }

  // Priority 2: doc_type / doc_kind mapping
  const typeKey = input.doc_type?.toLowerCase() ?? '';
  const kindKey = input.doc_kind?.toLowerCase() ?? '';
  const fromType = DOC_TYPE_TO_DOMAIN_MAP[typeKey] ?? DOC_TYPE_TO_DOMAIN_MAP[kindKey] ?? [];
  for (const d of fromType) {
    if (!domains.includes(d)) domains.push(d);
  }

  // Fallback: synthesis source
  if (domains.length === 0 && input.doc_source === 'synthesis') {
    domains.push('politics');
  }

  return dedup(domains) as SemanticDomain[];
}

// ─── Action Extraction ────────────────────────────────────────────────────────

/**
 * Extract primary actions.
 *
 * All phrase-based action detection uses PRIMARY text only (title + summary + metadata).
 * Contextual text (embedding, keywords, aliases) does NOT trigger primary actions.
 *
 * doc_type/doc_kind/classification still provide strong structural signals.
 */
function extractActions(
  input: FeatureExtractorInput,
  primaryText: string
): SemanticAction[] {
  const actions: SemanticAction[] = [];
  const docType = input.doc_type?.toLowerCase() ?? '';
  const docKind = input.doc_kind?.toLowerCase() ?? '';

  // ── Structural mapping from doc_type / doc_kind ──
  const typeActionMap: Record<string, SemanticAction> = {
    campaign: 'campaign',
    battle: 'battle',
    uprising: 'uprising',
    process: 'movement',
    reform: 'reform',
    program: 'economic_program',
    treaty: 'treaty_related',
    conference: 'conference',
    institution_founding: 'institution_founding',
    organization_founding: 'organization_founding',
    law_constitution: 'law_constitution',
    policy: 'policy',
    period_overview: 'period_overview',
    timeline_summary: 'timeline_synthesis',
  };

  const fromType = typeActionMap[docType] ?? typeActionMap[docKind];
  if (fromType) actions.push(fromType);

  // ── classification.category_l2/l3 supplemental mapping ──
  const l2 = input.classification?.category_l2?.toLowerCase() ?? '';
  const l3 = input.classification?.category_l3?.toLowerCase() ?? '';
  const l2ActionMap: Record<string, SemanticAction> = {
    treaty: 'treaty_related',
    battle: 'battle',
    campaign: 'campaign',
    military_campaign: 'campaign',
    revolution: 'uprising',
    political_event: 'policy',
    territorial_change: 'territorial_change',
    institution_founding: 'institution_founding',
    institution_foundation: 'institution_founding',
    organization_founding: 'organization_founding',
    conference: 'conference',
    reform: 'reform',
    economic_program: 'economic_program',
    election: 'election_referendum',
    boundary: 'boundary_or_division',
    normalization: 'normalization',
    accession: 'accession',
  };

  const fromL2 = l2ActionMap[l2] ?? l2ActionMap[l3];
  if (fromL2 && !actions.includes(fromL2)) actions.push(fromL2);

  // ── Synthesis doc_kind mapping ──
  if (input.doc_source === 'synthesis') {
    const kindSynthMap: Record<string, SemanticAction> = {
      period_overview: 'period_overview',
      timeline_summary: 'timeline_synthesis',
      comparison: 'comparison_synthesis',
      person_profile: 'person_profile',
    };
    const synAction = kindSynthMap[docKind];
    if (synAction && !actions.includes(synAction)) actions.push(synAction);
  }

  // ── Phrase-level detection (PRIMARY TEXT ONLY) ──
  // These match on title/summary/metadata labels — NOT on keywords or aliases.

  // treaty_signing: verb phrase in title/summary indicates this doc IS about signing
  if (hasAnyPhrase(primaryText, CONTROLLED_PHRASES.action.treatySigning)) {
    if (!actions.includes('treaty_signing')) actions.push('treaty_signing');
    if (!actions.includes('treaty_related')) actions.push('treaty_related');
  }

  // treaty_clause: clause-specific content in title/summary
  if (hasAnyPhrase(primaryText, CONTROLLED_PHRASES.action.treatyClause)) {
    if (!actions.includes('treaty_clause')) actions.push('treaty_clause');
  }

  // withdrawal_or_evacuation: title/summary explicitly about rut quan
  if (hasAnyPhrase(primaryText, CONTROLLED_PHRASES.action.withdrawalOrEvacuation)) {
    if (!actions.includes('withdrawal_or_evacuation')) actions.push('withdrawal_or_evacuation');
  }

  // uprising / resistance
  if (hasAnyPhrase(primaryText, CONTROLLED_PHRASES.action.uprising)) {
    if (!actions.includes('uprising')) actions.push('uprising');
  }

  // accession: title/summary directly says gia nhap [org]
  if (hasAnyPhrase(primaryText, CONTROLLED_PHRASES.action.accession)) {
    if (!actions.includes('accession')) actions.push('accession');
  }

  // normalization of diplomatic relations
  if (hasAnyPhrase(primaryText, CONTROLLED_PHRASES.action.normalization)) {
    if (!actions.includes('normalization')) actions.push('normalization');
  }

  // campaign_start: title/summary says mo man / phat dong chien dich
  if (hasAnyPhrase(primaryText, CONTROLLED_PHRASES.action.campaignStart)) {
    if (!actions.includes('campaign_start')) actions.push('campaign_start');
  }

  // victory_or_end: title/summary says chien thang / giai phong / dau hang
  if (hasAnyPhrase(primaryText, CONTROLLED_PHRASES.action.victoryOrEnd)) {
    if (!actions.includes('victory_or_end')) actions.push('victory_or_end');
  }

  // state founding / independence
  if (hasAnyPhrase(primaryText, CONTROLLED_PHRASES.action.foundingState)) {
    if (!actions.includes('state_founding')) actions.push('state_founding');
    if (!actions.includes('independence_declaration')) actions.push('independence_declaration');
  }

  // boundary_or_division: title/summary directly mentions vi tuyen / gioi tuyen
  // Conservative: only from primary text so contextual mentions in embedding don't trigger it
  const boundaryTerms = ['vi tuyen', 'gioi tuyen', 'ranh gioi', 'chia cat'];
  if (boundaryTerms.some(t => primaryText.includes(t))) {
    if (!actions.includes('boundary_or_division')) actions.push('boundary_or_division');
  }

  return dedup(actions) as SemanticAction[];
}

// ─── Affordance Extraction ────────────────────────────────────────────────────

/**
 * Affordances use actions (already filtered to primary) + contextual text
 * for disambiguation signals (Paris/Genève mentions in any text).
 */
function extractAffordances(
  input: FeatureExtractorInput,
  contextualText: string,
  actions: SemanticAction[]
): AnswerAffordance[] {
  const affordances: AnswerAffordance[] = [];

  // doc_source base affordances
  if (input.doc_source === 'event') {
    affordances.push('fact_lookup', 'date_lookup', 'actor_lookup', 'location_lookup');
  } else if (input.doc_source === 'synthesis') {
    affordances.push('explanation', 'cause_effect', 'timeline');
  }

  // doc_kind synthesis mapping
  const kindMap = DOC_KIND_TO_AFFORDANCE_MAP[input.doc_kind?.toLowerCase() ?? ''] ?? [];
  for (const a of kindMap) {
    if (!affordances.includes(a)) affordances.push(a);
  }

  // Action-driven affordances
  if (actions.includes('treaty_related') || actions.includes('treaty_signing')) {
    if (!affordances.includes('treaty_lookup')) affordances.push('treaty_lookup');
  }
  if (actions.includes('treaty_clause')) {
    if (!affordances.includes('clause_lookup')) affordances.push('clause_lookup');
  }
  if (actions.includes('conference')) {
    if (!affordances.includes('conference_lookup')) affordances.push('conference_lookup');
  }
  if (actions.includes('campaign') || actions.includes('campaign_start')) {
    if (!affordances.includes('campaign_lookup')) affordances.push('campaign_lookup');
  }
  if (actions.includes('movement') || actions.includes('uprising')) {
    if (!affordances.includes('movement_lookup')) affordances.push('movement_lookup');
  }
  if (
    actions.includes('institution_founding') ||
    actions.includes('organization_founding') ||
    actions.includes('state_founding')
  ) {
    if (!affordances.includes('organization_lookup')) affordances.push('organization_lookup');
  }
  if (actions.includes('comparison_synthesis')) {
    if (!affordances.includes('comparison')) affordances.push('comparison');
  }
  if (actions.includes('boundary_or_division') || actions.includes('treaty_clause')) {
    if (!affordances.includes('sub_event_lookup')) affordances.push('sub_event_lookup');
  }

  // Disambiguation signal: contextual text may mention Paris/Genève
  // (OK to use contextual here — disambiguation is about what topics this doc RELATES TO)
  if (
    hasAnyPhrase(contextualText, CONTROLLED_PHRASES.topic.parisAgreement) ||
    hasAnyPhrase(contextualText, CONTROLLED_PHRASES.topic.genevaAccords)
  ) {
    if (!affordances.includes('disambiguation')) affordances.push('disambiguation');
  }

  return dedup(affordances) as AnswerAffordance[];
}

// ─── Flag Extraction ──────────────────────────────────────────────────────────

/**
 * Flags:
 * - Primary action flags (is_treaty_signing, is_withdrawal_or_evacuation, etc.)
 *   come entirely from actions[] which was built from PRIMARY text.
 * - is_treaty_related is broader: set if actions say so, OR if contextual text
 *   mentions treaty topics (this is intentional — a doc contextually about a
 *   treaty is marked treaty_related even if its primary action is e.g. withdrawal).
 * - is_boundary_or_division: conservative — only from actions[] (primary text).
 */
function extractFlags(
  contextualText: string,
  actions: SemanticAction[],
  event_status?: string
): SemanticFeatures['flags'] {
  return {
    is_planned_not_executed: event_status === 'planned_not_executed',

    // Context-level flag: may fire even when doc's primary action is not treaty_signing
    is_treaty_related:
      actions.includes('treaty_related') ||
      actions.includes('treaty_signing') ||
      hasAnyPhrase(contextualText, CONTROLLED_PHRASES.topic.treatyRelated),

    // Primary-action flags: only from actions[] (which came from primary text)
    is_treaty_signing: actions.includes('treaty_signing'),
    is_treaty_clause: actions.includes('treaty_clause'),
    is_withdrawal_or_evacuation: actions.includes('withdrawal_or_evacuation'),
    is_foundation:
      actions.includes('institution_founding') ||
      actions.includes('organization_founding') ||
      actions.includes('state_founding'),
    is_accession: actions.includes('accession'),
    is_normalization: actions.includes('normalization'),
    is_boundary_or_division: actions.includes('boundary_or_division'),
    is_campaign_start: actions.includes('campaign_start'),
    is_victory_or_end: actions.includes('victory_or_end'),
  };
}

// ─── Topics & Aliases ─────────────────────────────────────────────────────────

/**
 * Topics use CONTEXTUAL text — keywords, aliases, embedding text all contribute.
 * Topics represent "what this document relates to", not "what this document IS about".
 */
function extractTopics(
  input: FeatureExtractorInput,
  contextualText: string
): string[] {
  const topics: string[] = [];

  // Raw keywords_vi (unnormalized, for BM25 boosting)
  if (input.retrieval_keywords_vi?.length) {
    topics.push(...input.retrieval_keywords_vi);
  }

  // Named topic signals from contextual text
  if (hasAnyPhrase(contextualText, CONTROLLED_PHRASES.topic.parisAgreement)) {
    topics.push('Hiệp định Paris');
  }
  if (hasAnyPhrase(contextualText, CONTROLLED_PHRASES.topic.genevaAccords)) {
    topics.push('Hiệp định Genève');
  }
  if (hasAnyPhrase(contextualText, CONTROLLED_PHRASES.topic.usWithdrawal)) {
    topics.push('Mỹ rút quân');
  }
  if (hasAnyPhrase(contextualText, CONTROLLED_PHRASES.topic.doiMoi)) {
    topics.push('Đổi mới');
  }

  return dedup(topics);
}

// ─── Main Extractor ───────────────────────────────────────────────────────────

/**
 * Extract SemanticFeatures for a single document.
 *
 * Deterministic: same input → same output.
 * No external calls. No hard-coded EVT IDs.
 *
 * Two-layer architecture (Patch 7C-1B):
 *   primaryText    → actions (what this doc IS about)
 *   contextualText → topics, flags.is_treaty_related (what this doc RELATES TO)
 */
export function extractSemanticFeatures(input: FeatureExtractorInput): SemanticFeatures {
  const primaryText = buildPrimarySemanticText(input);
  const contextualText = buildContextualSemanticText(input);

  const domains = extractDomains(input);
  const actions = extractActions(input, primaryText);
  const answer_affordances = extractAffordances(input, contextualText, actions);
  const flags = extractFlags(contextualText, actions, input.event_status);
  const topics = extractTopics(input, contextualText);
  const aliases = input.aliases ?? [];

  // treaty_names / campaign_names / movement_names
  // These use flags/actions (already primary-text-filtered)
  const treaty_names: string[] = [];
  const campaign_names: string[] = [];
  const movement_names: string[] = [];

  if (flags.is_treaty_related && input.title) {
    treaty_names.push(input.title);
  }
  if (actions.includes('campaign') && input.title) {
    campaign_names.push(input.title);
  }
  if ((actions.includes('movement') || actions.includes('uprising')) && input.title) {
    movement_names.push(input.title);
  }

  return {
    domains,
    actions,
    answer_affordances,
    topics,
    aliases,
    ...(treaty_names.length > 0 && { treaty_names }),
    ...(campaign_names.length > 0 && { campaign_names }),
    ...(movement_names.length > 0 && { movement_names }),
    time: {
      year_min: input.year ?? null,
      year_max: input.end_year ?? input.year ?? null,
      period_label: input.period_label ?? null,
    },
    evidence_quality: {
      verification_status:
        (input.verification_status as SemanticFeatures['evidence_quality']['verification_status']) ??
        undefined,
      significance_level: input.significance_level ?? undefined,
      has_sources: (input.source_ids?.length ?? 0) > 0,
    },
    flags,
  };
}
