/**
 * Query Normalizer — Stage 12C
 *
 * Main entry point for historical query normalization.
 * Runs before memory/follow-up rewrite, answer focus and hybrid retrieval.
 *
 * Pipeline position:
 *   raw user query
 *   → normalizeQuery()          ← THIS FILE
 *   → memory/follow-up rewrite
 *   → answer focus
 *   → hybrid retrieval
 *
 * Principles:
 * - Wrong rewrite is worse than no rewrite.
 * - OOS/ambiguous must not become answerable due to normalization.
 * - Follow-up pronouns must not be normalized away.
 * - All changes are deterministic — no LLM, no network.
 */

import type { QueryNormalizationResult, EntityMatch, NormalizationStatus } from './query-normalization-types.js';
import {
  normalizeText,
  removeVietnameseAccents,
  collapseRepeatedChars,
  generateQueryVariants,
  fuzzyScore,
  tokenJaccard,
} from './vietnamese-normalize.js';
import { HISTORICAL_ENTITY_DICTIONARY, buildAliasIndex } from './historical-entity-dictionary.js';

// ─── Constants ────────────────────────────────────────────────

const HIGH_CONFIDENCE_THRESHOLD = 0.85;
const MEDIUM_CONFIDENCE_THRESHOLD = 0.70;
const CANONICALIZE_EVIDENCE_THRESHOLD = 0.50;
const AMBIGUITY_GAP_THRESHOLD = 0.05;  // If top-2 within this gap → ambiguous

// ─── OOS Patterns (no rewrite) ────────────────────────────────

const OOS_PATTERNS = [
  /\bgi[aá] v[aà]ng\b/i,
  /\bth[oờ]i ti[eế]t\b/i,
  /\bb[oó]ng [dđ][aá]\b/i,
  /\bbitcoin\b/i,
  /\bch[uứ]ng kho[aá]n\b/i,
  /\bt[yỷ] gi[aá]\b/i,
  /\bh[oỏ]i h[aà]ng\b/i,
  /\bgi[aá] x[aă]ng\b/i,
];

// ─── Follow-up Pronoun Patterns (do not rewrite away) ─────────

const FOLLOWUP_PRONOUN_PATTERNS = [
  /\b(n[oó]|[oô]ng [aấ]y|b[aà] [aấ]y|h[oọ]|t[oổ] ch[uứ]c [dđ][oó]|s[uự] ki[eệ]n [dđ][oó]|chi[eế]n d[iị]ch [dđ][oó]|[aấ]y|v[iị] [dđ][oó])\b/i,
  /\b(no|ong ay|ba ay|to chuc do|su kien do|chien dich do|nguon nao noi vay|nguon cua phan do|o dau noi vay|ay)\b/i,
];

const YEAR_PATTERN = /\b(18|19|20)\d{2}\b/;
const DATE_STANDALONE_PATTERN = /^(?:\d{1,2}\/\d{1,2}|\d{1,2}-\d{1,2})$/;
const ENTITY_TYPE_CUE_PATTERN =
  /\b(hi[eệ]p\s*(d[iị]nh|din|u[oơ]c)|hiep\s*(dinh|din|uoc)|chi[eế]n\s*d[iị]ch|chien\s*dich|c[aá]ch\s*m[aạ]ng|cach\s*mang|cm\s*(thang|t)\s*\d?|tuy[eê]n\s*ng[oô]n|dang\s*cong\s*san|[dđ]ảng\s*c[oộ]ng\s*s[aả]n|t[oổ]\s*ch[uứ]c|to\s*chuc|phong\s*trao|phong\s*tr[aà]o)\b/i;
const ACTION_CUE_PATTERN =
  /\b([yý]\s*ngh[iĩ]a|noi\s*dung|n[oộ]i\s*dung|so\s*s[aá]nh|s[oọ]\s*s[aá]nh|kh[aá]c|vai\s*tr[oò]|di[eễ]n\s*bi[eế]n|dien\s*bien|l[aà]\s*g[iì]|la\s*gi|th[aà]nh\s*l[aậ]p|thanh\s*lap|khi\s*n[aà]o|ngu[oồ]n\s*n[aà]o|nguon\s*nao|d[uự]a\s*v[aà]o|dua\s*vao|ph[aâ]n\s*t[ií]ch|phan\s*tich|nh[aậ]n\s*x[eé]t|nhan\s*xet)\b/i;

// ─── Alias index (built once) ─────────────────────────────────

const ALIAS_INDEX = buildAliasIndex(HISTORICAL_ENTITY_DICTIONARY);

function tokenCount(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function containsAliasPhrase(form: string, alias: string): boolean {
  if (tokenCount(alias) < 2) return false;
  return new RegExp(`(^|\\s)${escapeRegExp(alias)}($|\\s)`, 'i').test(form);
}

function exactAliasTokenCount(match: EntityMatch): number {
  return tokenCount(match.matched_alias);
}

function computeEvidenceScore(
  rawQuery: string,
  normalized: string,
  noAccent: string,
  match: EntityMatch,
  secondMatch: EntityMatch | undefined,
  hasFollowUpContext: boolean,
): { score: number; reasons: string[] } {
  let score = 0;
  const reasons: string[] = [];
  const queryTokenCount = tokenCount(noAccent);

  if (YEAR_PATTERN.test(rawQuery) || YEAR_PATTERN.test(noAccent)) {
    score += 0.30;
    reasons.push('year_present');
  }
  if (ENTITY_TYPE_CUE_PATTERN.test(rawQuery) || ENTITY_TYPE_CUE_PATTERN.test(noAccent)) {
    score += 0.25;
    reasons.push('entity_type_cue');
  }
  if (ACTION_CUE_PATTERN.test(rawQuery) || ACTION_CUE_PATTERN.test(noAccent)) {
    score += 0.20;
    reasons.push('action_or_question_cue');
  }
  if (exactAliasTokenCount(match) >= 2) {
    score += 0.20;
    reasons.push('multi_token_alias');
  }
  if (exactAliasTokenCount(match) >= 3 && match.match_method !== 'fuzzy_token') {
    score += 0.25;
    reasons.push('curated_multi_token_alias');
  }
  if (queryTokenCount >= 3) {
    score += 0.10;
    reasons.push('query_length_support');
  }
  if (hasFollowUpContext) {
    score += 0.20;
    reasons.push('prior_memory_context');
  }
  if (queryTokenCount <= 1 || DATE_STANDALONE_PATTERN.test(noAccent)) {
    score -= 0.30;
    reasons.push('standalone_short_alias_penalty');
  }
  if (secondMatch && match.confidence - secondMatch.confidence < 0.10) {
    score -= 0.30;
    reasons.push('multiple_close_candidates_penalty');
  }

  return { score: Math.max(0, Math.min(1, Number(score.toFixed(2)))), reasons };
}

function buildCandidateAugmentedResult(
  rawQuery: string,
  normalized: string,
  noAccent: string,
  matches: EntityMatch[],
  topMatch: EntityMatch,
  secondMatch: EntityMatch | undefined,
  evidenceScore: number,
  warnings: string[],
  startedAt: number,
  reason: string,
): QueryNormalizationResult {
  return {
    original_query: rawQuery,
    normalized_query: normalized,
    no_accent_query: noAccent,
    canonical_query: rawQuery,
    retrieval_query: `${normalized} ${topMatch.canonical}`,
    normalization_status: 'candidate_augmented',
    rewrite_applied: false,
    confidence: topMatch.confidence,
    evidence_score: evidenceScore,
    hard_canonicalize_allowed: false,
    canonical_candidate: topMatch.canonical,
    ambiguity_reason: reason,
    matched_entities: matches,
    ambiguous_candidates: secondMatch ? [secondMatch] : [],
    latency_ms: Date.now() - startedAt,
    warnings: [...warnings, reason],
  };
}

// ─── Core: match against dictionary ──────────────────────────

function matchEntities(query: string): EntityMatch[] {
  const variants = generateQueryVariants(query);
  const collapsed = collapseRepeatedChars(removeVietnameseAccents(normalizeText(query)));
  const allForms = Array.from(new Set([...variants, collapsed]));

  const matchMap = new Map<string, EntityMatch>();

  for (const form of allForms) {
    // 1. Exact alias / no-accent match
    const exactEntry = ALIAS_INDEX.get(form);
    if (exactEntry) {
      const key = exactEntry.canonical;
      const existing = matchMap.get(key);
      const baseConf = 0.92 + (exactEntry.confidence_boost ?? 0);
      const conf = Math.min(1.0, baseConf);
      if (!existing || existing.confidence < conf) {
        matchMap.set(key, {
          canonical: exactEntry.canonical,
          type: exactEntry.type,
          matched_alias: form,
          match_method: exactEntry.no_accent === form ? 'no_accent_exact' : 'alias_exact',
          confidence: conf,
        });
      }
    }

    // 1b. Exact alias contained inside a longer natural-language query.
    // This recovers "viet minh la gi" / "cm thang 8 la gi" without allowing
    // single-token standalone aliases to hard-canonicalize.
    for (const entry of HISTORICAL_ENTITY_DICTIONARY) {
      const aliasForms = Array.from(new Set([entry.no_accent, ...entry.aliases]));
      const containedAlias = aliasForms
        .filter(alias => containsAliasPhrase(form, alias))
        .sort((a, b) => tokenCount(b) - tokenCount(a))[0];

      if (containedAlias) {
        const key = entry.canonical;
        const existing = matchMap.get(key);
        const aliasTokens = tokenCount(containedAlias);
        const baseConf = 0.91 + Math.min(0.04, aliasTokens * 0.005) + (entry.confidence_boost ?? 0);
        const conf = Math.min(1.0, baseConf);
        if (!existing || existing.confidence < conf) {
          matchMap.set(key, {
            canonical: entry.canonical,
            type: entry.type,
            matched_alias: containedAlias,
            match_method: 'alias_exact',
            confidence: conf,
          });
        }
      }
    }

    // 2. Fuzzy token match against all dictionary aliases
    for (const entry of HISTORICAL_ENTITY_DICTIONARY) {
      if (matchMap.has(entry.canonical)) continue;  // already matched exactly

      // Try against canonical no_accent form
      let best = fuzzyScore(form, entry.no_accent);
      let bestAlias = entry.no_accent;

      for (const alias of entry.aliases) {
        const s = fuzzyScore(form, alias);
        if (s > best) { best = s; bestAlias = alias; }
      }

      if (best >= MEDIUM_CONFIDENCE_THRESHOLD) {
        const conf = Math.min(1.0, best * 0.95 + (entry.confidence_boost ?? 0));
        const existing = matchMap.get(entry.canonical);
        if (!existing || existing.confidence < conf) {
          matchMap.set(entry.canonical, {
            canonical: entry.canonical,
            type: entry.type,
            matched_alias: bestAlias,
            match_method: 'fuzzy_token',
            confidence: conf,
          });
        }
      }
    }
  }

  // Sort descending by confidence
  return Array.from(matchMap.values()).sort((a, b) => b.confidence - a.confidence);
}

// ─── Main normalizer ──────────────────────────────────────────

/**
 * Normalize a raw user query against the historical entity dictionary.
 *
 * @param rawQuery Raw user input (may be unaccented, typo, or alias)
 * @param hasFollowUpContext Whether prior session memory has active entities
 *   (if true, follow-up patterns are preserved)
 */
export function normalizeQuery(
  rawQuery: string,
  hasFollowUpContext = false,
): QueryNormalizationResult {
  const t0 = Date.now();
  const warnings: string[] = [];

  const normalized = normalizeText(rawQuery);
  const noAccent = removeVietnameseAccents(normalized);

  // ── Guard 1: OOS detection ────────────────────────────────
  for (const pat of OOS_PATTERNS) {
    if (pat.test(rawQuery)) {
      return {
        original_query: rawQuery,
        normalized_query: normalized,
        no_accent_query: noAccent,
        canonical_query: rawQuery,
        retrieval_query: rawQuery,
        normalization_status: 'oos_unchanged',
        rewrite_applied: false,
        confidence: 0,
        evidence_score: 0,
        hard_canonicalize_allowed: false,
        matched_entities: [],
        ambiguous_candidates: [],
        latency_ms: Date.now() - t0,
        warnings: ['oos_detected_no_rewrite'],
      };
    }
  }

  // ── Guard 2: Follow-up pronoun passthrough ─────────────────
  // If query is mostly pronoun/ellipsis with no explicit historical token
  const isFollowUpPronoun = FOLLOWUP_PRONOUN_PATTERNS.some(p => p.test(rawQuery));
  if (isFollowUpPronoun) {
    // Follow-up ellipsis is resolved by session memory, not by entity canonicalization.
    return {
      original_query: rawQuery,
      normalized_query: normalized,
      no_accent_query: noAccent,
      canonical_query: rawQuery,
      retrieval_query: rawQuery,
      normalization_status: 'followup_passthrough',
      rewrite_applied: false,
      confidence: 0,
      evidence_score: hasFollowUpContext ? 0.20 : 0,
      hard_canonicalize_allowed: false,
      ambiguity_reason: hasFollowUpContext ? 'followup_resolved_by_memory_layer' : 'followup_without_memory',
      matched_entities: [],
      ambiguous_candidates: [],
      latency_ms: Date.now() - t0,
      warnings: [hasFollowUpContext ? 'followup_pronoun_passthrough' : 'followup_without_memory_passthrough'],
    };
  }

  // ── Entity matching ────────────────────────────────────────
  const matches = matchEntities(rawQuery);
  const topMatch = matches[0];
  const secondMatch = matches[1];

  // ── Ambiguity check ────────────────────────────────────────
  const isAmbiguous =
    topMatch &&
    secondMatch &&
    topMatch.confidence - secondMatch.confidence < AMBIGUITY_GAP_THRESHOLD &&
    topMatch.confidence < HIGH_CONFIDENCE_THRESHOLD;

  // ── No match ───────────────────────────────────────────────
  if (!topMatch || topMatch.confidence < MEDIUM_CONFIDENCE_THRESHOLD) {
    return {
      original_query: rawQuery,
      normalized_query: normalized,
      no_accent_query: noAccent,
      canonical_query: rawQuery,
      retrieval_query: normalized,
      normalization_status: 'unchanged_low_confidence',
      rewrite_applied: false,
      confidence: topMatch?.confidence ?? 0,
      evidence_score: 0,
      hard_canonicalize_allowed: false,
      matched_entities: matches,
      ambiguous_candidates: [],
      latency_ms: Date.now() - t0,
      warnings: topMatch ? ['low_confidence_no_rewrite'] : ['no_entity_match'],
    };
  }

  // ── Ambiguous → clarification ──────────────────────────────
  if (isAmbiguous) {
    return {
      original_query: rawQuery,
      normalized_query: normalized,
      no_accent_query: noAccent,
      canonical_query: rawQuery,
      retrieval_query: normalized,
      normalization_status: 'ambiguous_needs_clarification',
      rewrite_applied: false,
      confidence: topMatch.confidence,
      evidence_score: 0,
      hard_canonicalize_allowed: false,
      canonical_candidate: topMatch.canonical,
      ambiguity_reason: 'ambiguous_top_candidates',
      matched_entities: [topMatch],
      ambiguous_candidates: matches.slice(1, 4),
      latency_ms: Date.now() - t0,
      warnings: ['ambiguous_top_candidates'],
    };
  }

  const evidence = computeEvidenceScore(rawQuery, normalized, noAccent, topMatch, secondMatch, hasFollowUpContext);
  topMatch.evidence_score = evidence.score;
  const hardCanonicalizeAllowed = topMatch.confidence >= HIGH_CONFIDENCE_THRESHOLD
    && evidence.score >= CANONICALIZE_EVIDENCE_THRESHOLD;

  // ── High confidence + sufficient evidence → canonical rewrite
  if (hardCanonicalizeAllowed) {
    return {
      original_query: rawQuery,
      normalized_query: normalized,
      no_accent_query: noAccent,
      canonical_query: topMatch.canonical,
      retrieval_query: topMatch.canonical,
      normalization_status: 'canonicalized',
      rewrite_applied: true,
      confidence: topMatch.confidence,
      evidence_score: evidence.score,
      hard_canonicalize_allowed: true,
      canonical_candidate: topMatch.canonical,
      matched_entities: matches,
      ambiguous_candidates: [],
      latency_ms: Date.now() - t0,
      warnings: [...warnings, ...evidence.reasons.map(reason => `evidence_${reason}`)],
    };
  }

  // ── High confidence but weak evidence → cautious candidate only
  if (topMatch.confidence >= HIGH_CONFIDENCE_THRESHOLD) {
    return buildCandidateAugmentedResult(
      rawQuery,
      normalized,
      noAccent,
      matches,
      topMatch,
      secondMatch,
      evidence.score,
      warnings,
      t0,
      'insufficient_evidence_for_hard_canonicalization',
    );
  }

  // ── Medium confidence → augmented retrieval ────────────────
  const augmented = `${topMatch.canonical} ${normalized}`;
  return {
    original_query: rawQuery,
    normalized_query: normalized,
    no_accent_query: noAccent,
    canonical_query: rawQuery,
    retrieval_query: augmented,
    normalization_status: 'candidate_augmented',
    rewrite_applied: false,
    confidence: topMatch.confidence,
    evidence_score: evidence.score,
    hard_canonicalize_allowed: false,
    canonical_candidate: topMatch.canonical,
    ambiguity_reason: 'medium_confidence_augmented',
    matched_entities: matches,
    ambiguous_candidates: secondMatch ? [secondMatch] : [],
    latency_ms: Date.now() - t0,
    warnings: ['medium_confidence_augmented'],
  };
}
