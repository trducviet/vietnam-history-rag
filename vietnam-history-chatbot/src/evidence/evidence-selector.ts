/**
 * Evidence Selector (Patch 7E + 7E-1)
 *
 * Assigns evidence roles (primary/supporting/contrast/excluded) to
 * reranked retrieval candidates based on QueryFrame semantics.
 *
 * Design principles:
 * - ZERO document-ID hard-coding.
 * - ZERO benchmark label leakage.
 * - Uses QueryFrame.answer_focus + contrast_focus + constraints.
 * - Uses SemanticFeatures.actions + flags + topics + treaty/campaign names.
 * - Intent-specific policies for disambiguation/comparison/timeline/etc.
 * - Fully deterministic for the same input.
 *
 * Patch 7E-1 fixes:
 * - Normalized rerank/BM25 score to 0–1 before using as tiebreaker.
 * - Retrieval score no longer saturates primary_score to 1.0.
 * - Disambiguation strict policy: semantic match only, must_not_be_about blocks primary.
 * - Fallback primary safety: won't promote contrast docs for disambiguation.
 */

import type {
  QueryFrame,
  QueryFocus,
  IndexableDocument,
  RerankedResult,
  LoadedDataset,
} from '../shared/types.js';
import type {
  EvidenceRole,
  EvidenceSelectionReason,
  EvidenceItem,
  EvidenceSelection,
} from '../shared/types.js';
import type { SemanticFeatures } from '../indexing/semantic-taxonomy.js';
import { normalizeVietnameseText } from '../indexing/semantic-taxonomy.js';
import { extractQueryFocus, scoreDocumentFocus, detectTreatySubtopicFocus, normalizeForFocus, type DocFocusScore } from './focus-precision.js';
import { scoreEntityCollisionPenalty, scoreEntityAliasBoost, getAllEntityProfiles } from '../routing/entity-collision-map.js';

// Patch 9E: Cache entity profiles by ID for sync lookup
const _entityProfileCache = new Map<string, any>();
function getEntityProfileById(id: string) {
  if (_entityProfileCache.size === 0) {
    for (const p of getAllEntityProfiles()) _entityProfileCache.set(p.id, p);
  }
  return _entityProfileCache.get(id);
}

// ─── Candidate Resolution ────────────────────────────────────────────────────

/** Resolved candidate: RerankedResult enriched with full document data */
interface ResolvedCandidate {
  result: RerankedResult;
  doc: IndexableDocument;
  sf: SemanticFeatures | undefined;
}

/** Resolve reranked results to full documents */
function resolveCandidates(
  results: RerankedResult[],
  dataset: LoadedDataset
): ResolvedCandidate[] {
  const resolved: ResolvedCandidate[] = [];
  for (const r of results) {
    const doc = dataset.events.get(r.doc_id) ?? dataset.synthesis.get(r.doc_id);
    if (!doc || !doc.canonical) continue;
    resolved.push({
      result: r,
      doc,
      sf: doc.semantic_features,
    });
  }
  return resolved;
}

// ─── Score Normalization ─────────────────────────────────────────────────────

/**
 * Normalize a retrieval/rerank score to 0–1 range.
 *
 * BM25 raw scores can be very large (e.g., 53, 57). LLM reranker outputs 0–1.
 * Passthrough mode uses raw BM25 as rerank_score.
 *
 * This function ensures we get a consistent 0–1 tiebreaker:
 * - Scores already in [0, 1]: returned as-is.
 * - Scores > 1: compressed via log1p to ~0–1 range.
 */
function normalizeRetrievalScore(score: number): number {
  if (score <= 0) return 0;
  if (score <= 1) return score;
  // log1p(100) ≈ 4.615 → scores up to ~100 map to ~1.0
  return Math.min(1, Math.log1p(score) / Math.log1p(100));
}

/**
 * Check if a candidate matches any action in must_not_be_about.
 * Used to block primary role for disambiguation/misconception intents.
 */
function matchesMustNotBeAbout(
  candidate: ResolvedCandidate,
  frame: QueryFrame
): boolean {
  if (!frame.constraints?.must_not_be_about?.length) return false;
  const sf = candidate.sf;
  if (!sf) return false;
  for (const forbidden of frame.constraints.must_not_be_about) {
    const flagKey = ACTION_TO_FLAG[forbidden];
    if (flagKey && sf.flags?.[flagKey]) return true;
    if (sf.actions?.includes(forbidden as any)) return true;
  }
  return false;
}

// ─── Scoring Helpers ─────────────────────────────────────────────────────────

/** Map QueryFocus.action to a SemanticFeatures flag name */
const ACTION_TO_FLAG: Record<string, keyof NonNullable<SemanticFeatures['flags']>> = {
  'withdrawal_or_evacuation': 'is_withdrawal_or_evacuation',
  'treaty_signing': 'is_treaty_signing',
  'treaty_clause': 'is_treaty_clause',
  'treaty_related': 'is_treaty_related',
  'accession': 'is_accession',
  'normalization': 'is_normalization',
  'boundary_or_division': 'is_boundary_or_division',
  'campaign_start': 'is_campaign_start',
  'victory_or_end': 'is_victory_or_end',
  'organization_founding': 'is_foundation',
  'invasion_or_attack': 'is_campaign_start', // Patch 7L-A: invasion maps to campaign_start flag
};

/**
 * Score how well a candidate matches a QueryFocus (answer or contrast).
 * Returns a 0–1 score + reasons.
 */
function scoreFocusMatch(
  candidate: ResolvedCandidate,
  focus: QueryFocus | undefined,
  label: 'answer' | 'contrast'
): { score: number; reasons: EvidenceSelectionReason[] } {
  if (!focus) return { score: 0, reasons: [] };

  let score = 0;
  const reasons: EvidenceSelectionReason[] = [];
  const sf = candidate.sf;

  // 1. Action match via SemanticFeatures flags
  if (focus.action && sf) {
    const flagKey = ACTION_TO_FLAG[focus.action];
    if (flagKey && sf.flags?.[flagKey]) {
      score += 0.35;
      reasons.push({
        code: `${label}_action_flag_match`,
        message: `Flag ${flagKey} is true → matches ${label}.action=${focus.action}`,
        weight: 0.35,
      });
    }
    // Also check actions array
    if (sf.actions?.includes(focus.action as any)) {
      score += 0.15;
      reasons.push({
        code: `${label}_action_array_match`,
        message: `actions[] contains ${focus.action}`,
        weight: 0.15,
      });
    }
  }

  // 2. Action match via title/summary text fallback (when SF missing)
  if (focus.action && !sf) {
    const normTitle = normalizeVietnameseText(candidate.doc.title);
    const normSummary = normalizeVietnameseText(candidate.doc.summary || '');
    const actionPhrases = getActionPhrases(focus.action);
    if (actionPhrases.some(p => normTitle.includes(p) || normSummary.includes(p))) {
      score += 0.25;
      reasons.push({
        code: `${label}_action_text_match`,
        message: `Title/summary contains text matching ${focus.action}`,
        weight: 0.25,
      });
    }
  }

  // 3. Treaty name match
  if (focus.treaty_names?.length && sf?.treaty_names?.length) {
    const overlap = focus.treaty_names.filter(t =>
      sf.treaty_names!.some(st => st.includes(t) || t.includes(st))
    );
    if (overlap.length > 0) {
      score += 0.15;
      reasons.push({
        code: `${label}_treaty_match`,
        message: `Treaty names overlap: ${overlap.join(', ')}`,
        weight: 0.15,
      });
    }
  }

  // 4. Campaign name match
  if (focus.campaign_names?.length && sf?.campaign_names?.length) {
    const overlap = focus.campaign_names.filter(c =>
      sf.campaign_names!.some(sc => sc.includes(c) || c.includes(sc))
    );
    if (overlap.length > 0) {
      score += 0.15;
      reasons.push({
        code: `${label}_campaign_match`,
        message: `Campaign names overlap: ${overlap.join(', ')}`,
        weight: 0.15,
      });
    }
  }

  // 5. Movement name match
  if (focus.movement_names?.length && sf?.movement_names?.length) {
    const overlap = focus.movement_names.filter(m =>
      sf.movement_names!.some(sm => sm.includes(m) || m.includes(sm))
    );
    if (overlap.length > 0) {
      score += 0.10;
      reasons.push({
        code: `${label}_movement_match`,
        message: `Movement names overlap: ${overlap.join(', ')}`,
        weight: 0.10,
      });
    }
  }

  // 6. Actor match via people_labels
  if (focus.actor?.length) {
    const normFocusActors = focus.actor.map(a => normalizeVietnameseText(a));
    const normDocActors = candidate.doc.people_labels.map(p => normalizeVietnameseText(p));
    const actorOverlap = normFocusActors.filter(fa =>
      normDocActors.some(da => da.includes(fa) || fa.includes(da))
    );
    if (actorOverlap.length > 0) {
      score += 0.10;
      reasons.push({
        code: `${label}_actor_match`,
        message: `Actor overlap found`,
        weight: 0.10,
      });
    }
  }

  // 7. Organization match
  if (focus.organization?.length) {
    const normFocusOrgs = focus.organization.map(o => normalizeVietnameseText(o));
    const normDocOrgs = candidate.doc.organization_labels.map(o => normalizeVietnameseText(o));
    const orgOverlap = normFocusOrgs.filter(fo =>
      normDocOrgs.some(doo => doo.includes(fo) || fo.includes(doo))
    );
    if (orgOverlap.length > 0) {
      score += 0.10;
      reasons.push({
        code: `${label}_org_match`,
        message: `Organization overlap found`,
        weight: 0.10,
      });
    }
  }

  // 8. Year constraint match
  if (focus.time?.explicit_years?.length && candidate.doc.year) {
    if (focus.time.explicit_years.includes(candidate.doc.year)) {
      score += 0.05;
      reasons.push({
        code: `${label}_year_match`,
        message: `Year ${candidate.doc.year} matches focus`,
        weight: 0.05,
      });
    }
  }

  return { score: Math.min(score, 1.0), reasons };
}

/** Get normalized Vietnamese phrases for a given action */
function getActionPhrases(action: string): string[] {
  const map: Record<string, string[]> = {
    'withdrawal_or_evacuation': ['rut quan', 'rut khoi', 'rut toan bo'],
    'treaty_signing': ['ky hiep dinh', 'ky ket', 'duoc ky'],
    'treaty_clause': ['dieu khoan', 'cam ket', 'thoa thuan'],
    'accession': ['gia nhap', 'tro thanh thanh vien'],
    'normalization': ['binh thuong hoa', 'lap lai quan he'],
    'boundary_or_division': ['vi tuyen', 'chia cat', 'chia doi'],
    'campaign_start': ['bat dau chien dich', 'mo man'],
    'victory_or_end': ['chien thang', 'giai phong', 'dau hang'],
    'organization_founding': ['thanh lap', 'khai sinh', 'ra doi'],
    'independence_declaration': ['tuyen ngon doc lap', 'tuyen bo doc lap'],
    'conference': ['hoi nghi'],
    'reform': ['doi moi', 'cai cach'],
    'invasion_or_attack': ['no sung', 'xam luoc', 'tan cong', 'mo dau xam luoc'], // Patch 7L-A
  };
  return map[action] ?? [];
}

// ─── Role Scoring ────────────────────────────────────────────────────────────

/**
 * Score candidate for PRIMARY role.
 *
 * Patch 7E-1: retrieval score normalized to 0–1 and used as tiny tiebreaker only.
 * Semantic focus match is the dominant signal.
 */
function scorePrimary(
  candidate: ResolvedCandidate,
  frame: QueryFrame
): { score: number; semanticScore: number; reasons: EvidenceSelectionReason[] } {
  const { score: focusScore, reasons } = scoreFocusMatch(candidate, frame.answer_focus, 'answer');

  let total = focusScore;
  const semanticScore = focusScore; // pure semantic match, no retrieval contamination

  // Boost for verified docs
  if (candidate.doc.verification_status === 'verified') {
    total += 0.05;
    reasons.push({
      code: 'source_quality_boost',
      message: 'Verified document',
      weight: 0.05,
    });
  }

  // Boost for synthesis docs in timeline/explanation intents
  if (
    candidate.doc.doc_source === 'synthesis' &&
    ['timeline', 'explanation', 'cause_effect', 'comparison', 'significance_lookup'].includes(frame.intent)
  ) {
    total += 0.10;
    reasons.push({
      code: 'timeline_synthesis_boost',
      message: `Synthesis doc boosted for ${frame.intent} intent`,
      weight: 0.10,
    });
  }

  // Retrieval score: normalized tiebreaker only (Patch 7E-1)
  const normRS = normalizeRetrievalScore(candidate.result.rerank_score);
  total += normRS * 0.05;
  reasons.push({
    code: 'retrieval_score_tiebreak_only',
    message: `Normalized rerank=${normRS.toFixed(3)} (raw=${candidate.result.rerank_score.toFixed(3)}), contrib=+${(normRS * 0.05).toFixed(4)}`,
    weight: normRS * 0.05,
  });

  // Patch 9E: Entity collision penalty/boost
  // NOTE: frame.entity_profile is a subset. Use getEntityProfileById for full profile with forbidden_aliases.
  if (frame.entity_profile) {
    const fullProfile = getEntityProfileById(frame.entity_profile.id);
    if (fullProfile) {
      const docText = candidate.doc.text_for_embedding || candidate.doc.summary || '';
      const docYear = candidate.doc.year ?? undefined;
      const penalty = scoreEntityCollisionPenalty(
        candidate.doc.title, docText, docYear, fullProfile
      );
      if (penalty > 0) {
        total -= penalty;
        reasons.push({
          code: 'entity_collision_penalty',
          message: `Entity collision: doc matches forbidden aliases for ${frame.entity_profile.id}, penalty=-${penalty.toFixed(2)}`,
          weight: -penalty,
        });
      }
      const boost = scoreEntityAliasBoost(
        candidate.doc.title, docText, docYear, fullProfile
      );
      if (boost > 0) {
        total += boost;
        reasons.push({
          code: 'entity_alias_boost',
          message: `Entity alias match for ${frame.entity_profile.id}, boost=+${boost.toFixed(2)}`,
          weight: boost,
        });
      }
    }
  }

  return { score: Math.min(Math.max(total, 0), 1.0), semanticScore, reasons };
}

/**
 * Score candidate for CONTRAST role.
 *
 * Patch 7E-1: also returns semanticScore for clean comparison.
 */
function scoreContrast(
  candidate: ResolvedCandidate,
  frame: QueryFrame
): { score: number; semanticScore: number; reasons: EvidenceSelectionReason[] } {
  if (!frame.contrast_focus) return { score: 0, semanticScore: 0, reasons: [] };

  const { score: contrastFocusScore, reasons } = scoreFocusMatch(
    candidate, frame.contrast_focus, 'contrast'
  );

  const semanticScore = contrastFocusScore;

  // Boost contrast score if constraints require contrast
  let total = contrastFocusScore;
  if (frame.constraints?.requires_contrast) {
    total *= 1.2;
  }

  return { score: Math.min(total, 1.0), semanticScore, reasons };
}

function scoreExclusion(
  candidate: ResolvedCandidate,
  frame: QueryFrame
): { score: number; reasons: EvidenceSelectionReason[] } {
  const reasons: EvidenceSelectionReason[] = [];
  let score = 0;

  // Exclusion if must_not_be_about and candidate strongly matches that
  if (frame.constraints?.must_not_be_about?.length) {
    const sf = candidate.sf;
    for (const forbidden of frame.constraints.must_not_be_about) {
      const flagKey = ACTION_TO_FLAG[forbidden];
      if (flagKey && sf?.flags?.[flagKey]) {
        score += 0.3;
        reasons.push({
          code: 'excluded_must_not_be_about',
          message: `Doc has ${flagKey}=true, which is in must_not_be_about`,
          weight: 0.3,
        });
      }
    }
  }

  // Low retrieval score → higher exclusion (use normalized score)
  const normRS = normalizeRetrievalScore(candidate.result.rerank_score);
  if (normRS < 0.15) {
    score += 0.2;
    reasons.push({
      code: 'excluded_low_relevance',
      message: `Very low normalized rerank_score: ${normRS.toFixed(3)} (raw=${candidate.result.rerank_score.toFixed(3)})`,
      weight: 0.2,
    });
  }

  return { score: Math.min(score, 1.0), reasons };
}

// ─── Role Decision Logic ─────────────────────────────────────────────────────

interface RoleDecision {
  role: EvidenceRole;
  role_score: number;
  reasons: EvidenceSelectionReason[];
}

/**
 * Decide role for a single candidate using intent-specific policy.
 *
 * Patch 7E-1: decisions use semanticScore (no retrieval contamination)
 * for primary/contrast comparison. Retrieval score only affects role_score
 * for ordering within the same role.
 */
function decideRole(
  candidate: ResolvedCandidate,
  frame: QueryFrame,
  currentPrimaryCount: number,
  currentContrastCount: number
): RoleDecision {
  const primary = scorePrimary(candidate, frame);
  const contrast = scoreContrast(candidate, frame);
  const exclusion = scoreExclusion(candidate, frame);

  const allReasons = [...primary.reasons, ...contrast.reasons, ...exclusion.reasons];

  // Use SEMANTIC scores for role decisions — not inflated total scores
  const pSemantic = primary.semanticScore;
  const cSemantic = contrast.semanticScore;
  const isForbidden = matchesMustNotBeAbout(candidate, frame);

  // ── Exclusion check first ──
  if (exclusion.score > 0.5) {
    return { role: 'excluded', role_score: exclusion.score, reasons: allReasons };
  }

  // ── Intent-specific logic ──
  switch (frame.intent) {
    case 'disambiguation': {
      // STRICT disambiguation policy (Patch 7E-1):
      // 1. must_not_be_about blocks primary unconditionally
      // 2. contrast semantic match → contrast role (not primary)
      // 3. only candidates with strong answer_focus semantic match → primary
      // 4. pure retrieval score never makes a doc primary

      // BLOCK: must_not_be_about flags → never primary
      if (isForbidden) {
        allReasons.push({
          code: 'must_not_be_about_blocks_primary',
          message: 'Doc matches must_not_be_about → blocked from primary in disambiguation',
        });
        // If it matches contrast_focus, it's contrast; otherwise supporting
        if (cSemantic >= 0.3) {
          allReasons.push({
            code: 'semantic_contrast_action_match',
            message: `Contrast semantic=${cSemantic.toFixed(3)}, doc demoted to contrast`,
          });
          return { role: 'contrast', role_score: contrast.score, reasons: allReasons };
        }
        return { role: 'supporting', role_score: primary.score, reasons: allReasons };
      }

      // CONTRAST: semantic contrast match >= 0.35 AND contrast >= answer
      if (cSemantic >= 0.35 && cSemantic >= pSemantic) {
        allReasons.push({
          code: 'contrast_action_blocks_primary',
          message: `Contrast semantic=${cSemantic.toFixed(3)} >= answer semantic=${pSemantic.toFixed(3)} → contrast`,
        });
        return { role: 'contrast', role_score: contrast.score, reasons: allReasons };
      }

      // PRIMARY: needs strong answer_focus semantic match
      if (pSemantic >= 0.35 && pSemantic > cSemantic) {
        allReasons.push({
          code: 'semantic_answer_action_match',
          message: `Answer semantic=${pSemantic.toFixed(3)} > contrast=${cSemantic.toFixed(3)} → primary`,
        });
        return { role: 'primary', role_score: primary.score, reasons: allReasons };
      }

      // AMBIGUOUS: both match or neither matches strongly
      if (pSemantic > 0 && pSemantic === cSemantic) {
        // Tie: use must_not_be_about to break tie (already checked above)
        // Fallback to supporting to avoid contamination
        allReasons.push({
          code: 'disambiguation_requires_answer_action',
          message: `Tied semantic scores (${pSemantic.toFixed(3)}), defaulting to supporting`,
        });
        return { role: 'supporting', role_score: primary.score, reasons: allReasons };
      }

      // No strong semantic match → supporting (retrieval score alone is NOT enough)
      if (pSemantic < 0.2) {
        allReasons.push({
          code: 'disambiguation_requires_answer_action',
          message: `Answer semantic=${pSemantic.toFixed(3)} too weak for primary in disambiguation`,
        });
        return { role: 'supporting', role_score: primary.score, reasons: allReasons };
      }

      // Moderate answer match, no contrast → primary if slot available
      if (currentPrimaryCount < 2) {
        return { role: 'primary', role_score: primary.score, reasons: allReasons };
      }
      return { role: 'supporting', role_score: primary.score, reasons: allReasons };
    }

    case 'comparison': {
      // Comparison: need both sides. Keep contrast docs, don't exclude them.
      // must_not_be_about does NOT block contrast in comparison (both sides needed).

      // 7N: Noise-year demotion — if comparison has specific years in both sides,
      // demote docs whose title references a year NOT in either side
      if (frame.comparison_sides) {
        const sideAYears = (frame.comparison_sides.side_a.match(/\b(1[89]\d{2}|20\d{2})\b/g) ?? []);
        const sideBYears = (frame.comparison_sides.side_b.match(/\b(1[89]\d{2}|20\d{2})\b/g) ?? []);
        const allowedYears = new Set([...sideAYears, ...sideBYears, '1858', '1945', '1954', '1975']);
        if (sideAYears.length > 0 && sideBYears.length > 0) {
          const titleYears = (candidate.doc.title.match(/\b(1[89]\d{2}|20\d{2})\b/g) ?? []);
          const hasNoiseYear = titleYears.some(y => !allowedYears.has(y));
          if (hasNoiseYear) {
            allReasons.push({
              code: 'comparison_noise_year_demotion',
              message: 'Demoted: doc title contains a year outside the compared sides',
              weight: -0.3,
            });
            return { role: 'supporting', role_score: primary.score * 0.3, reasons: allReasons };
          }
        }
      }

      if (cSemantic > pSemantic && currentContrastCount < 2) {
        return { role: 'contrast', role_score: contrast.score, reasons: allReasons };
      }
      if (pSemantic > 0.15 || primary.score > 0.2) {
        return { role: currentPrimaryCount < 2 ? 'primary' : 'supporting', role_score: primary.score, reasons: allReasons };
      }
      return { role: 'supporting', role_score: primary.score, reasons: allReasons };
    }

    case 'misconception_check': {
      // Misconception: docs that correct the misconception → primary
      // must_not_be_about blocks primary (similar to disambiguation)
      if (isForbidden) {
        if (cSemantic >= 0.3) {
          return { role: 'contrast', role_score: contrast.score, reasons: allReasons };
        }
        return { role: 'supporting', role_score: primary.score, reasons: allReasons };
      }
      if (cSemantic > pSemantic && cSemantic > 0.3) {
        return { role: 'contrast', role_score: contrast.score, reasons: allReasons };
      }
      if (pSemantic > 0.15 || primary.score > 0.2) {
        return { role: 'primary', role_score: primary.score, reasons: allReasons };
      }
      return { role: 'supporting', role_score: primary.score, reasons: allReasons };
    }

    case 'timeline': {
      // Timeline: synthesis timeline docs → primary if topic matches
      if (
        candidate.doc.doc_source === 'synthesis' &&
        primary.score > 0.25
      ) {
        return { role: 'primary', role_score: primary.score, reasons: allReasons };
      }
      if (primary.score > 0.15) {
        return { role: currentPrimaryCount < 1 ? 'primary' : 'supporting', role_score: primary.score, reasons: allReasons };
      }
      return { role: 'supporting', role_score: primary.score, reasons: allReasons };
    }

    default: {
      // date/fact/actor/location/org lookup — conservative
      if (primary.score > 0.2 && currentPrimaryCount < 2) {
        return { role: 'primary', role_score: primary.score, reasons: allReasons };
      }
      if (primary.score > 0.1) {
        return { role: 'supporting', role_score: primary.score, reasons: allReasons };
      }
      return { role: 'supporting', role_score: primary.score, reasons: allReasons };
    }
  }
}

// ─── Main Selector ───────────────────────────────────────────────────────────

/**
 * Select evidence roles for reranked candidates.
 *
 * If queryFrame is undefined, falls back to simple score-based assignment:
 * top candidate → primary, next → supporting.
 */
export function selectEvidence(
  candidates: RerankedResult[],
  dataset: LoadedDataset,
  queryFrame?: QueryFrame,
  query?: string  // Patch 7L-A: pass query for focus-precision
): EvidenceSelection {
  const resolved = resolveCandidates(candidates, dataset);
  const warnings: string[] = [];

  // ── No QueryFrame fallback ──
  if (!queryFrame) {
    return fallbackSelection(resolved, warnings);
  }

  // ── QueryFrame-based selection ──
  const items: EvidenceItem[] = [];
  let primaryCount = 0;
  let contrastCount = 0;

  for (const c of resolved) {
    const decision = decideRole(c, queryFrame, primaryCount, contrastCount);
    items.push({
      doc_id: c.doc.doc_id,
      role: decision.role,
      role_score: decision.role_score,
      reasons: decision.reasons,
      rerank_score: c.result.rerank_score,
    });
    if (decision.role === 'primary') primaryCount++;
    if (decision.role === 'contrast') contrastCount++;
  }

  // ── Patch 7L-E2 + 7M-A: Focus precision — with treaty subtopic override ──
  // When focus profile is active and a focus-positive candidate exists:
  //   1. bestFocusItem.role = 'primary'
  //   2. bestFocusItem.role_score = max(all primaries) + 1.0
  //   3. bestFocusItem.rerank_score = max(all primaries) + 1.0
  //   4. For lookup-like intents: demote all OTHER primaries to supporting
  //   5. Tag: focus_profile_positive + focus_primary_forced
  //
  // Patch 7M-A: When a SPECIFIC TreatySubtopicFocus exists (not general/unknown),
  // SUPPRESS the generic treaty focus profile (treaty_geneve_focus / treaty_paris_focus)
  // and use subtopic-aware primary selection instead.
  const lookupLikeIntents = [
    'fact_lookup', 'date_lookup', 'actor_lookup', 'location_lookup',
    'entity_profile', 'treaty_lookup', 'clause_lookup', 'misconception_check',
  ];
  if (lookupLikeIntents.includes(queryFrame.intent)) {
    const queryFocus = extractQueryFocus({ query: query ?? '', queryFrame });
    const isDisambigProtected = queryFrame.intent === 'disambiguation';
    const isComparisonIntent = queryFrame.intent === 'comparison';

    // Patch 7M-A: Detect treaty subtopic — if specific, suppress generic treaty focus
    const treatySubtopic = detectTreatySubtopicFocus(query ?? '');
    const hasSpecificSubtopic = treatySubtopic != null &&
      treatySubtopic.subtopic !== 'general_treaty' &&
      treatySubtopic.subtopic !== 'unknown';
    const isGenericTreatyProfile = queryFocus.focus_profile?.id === 'treaty_geneve_focus' ||
      queryFocus.focus_profile?.id === 'treaty_paris_focus';
    // Suppress generic treaty profile when a specific subtopic is active
    const suppressGenericProfile = hasSpecificSubtopic && isGenericTreatyProfile;

    if (queryFocus.focus_profile && !isDisambigProtected && !isComparisonIntent && !suppressGenericProfile) {
      // ── STANDARD focus profile force (non-treaty-subtopic) ──
      // Step 1: Find the best focus-positive candidate across ALL items
      let bestFocusItem: typeof items[0] | null = null;
      let bestFocusScore: DocFocusScore | null = null;

      for (const item of items) {
        if (item.role === 'excluded') continue;
        if (item.role === 'contrast') continue;
        if (item.reasons.some(r => r.code === 'hard_negative_excluded' || r.code === 'forbidden')) continue;

        const doc = resolved.find(r => r.doc.doc_id === item.doc_id)?.doc;
        if (!doc) continue;

        const focusScore = scoreDocumentFocus(doc, queryFocus);
        const hasProfilePositive = focusScore.matched_terms.some(t => t.startsWith('profile+:'));
        if (!hasProfilePositive) continue;
        if (focusScore.penalties.length > 0) continue;

        if (!bestFocusItem || focusScore.score > (bestFocusScore?.score ?? -1)) {
          bestFocusItem = item;
          bestFocusScore = focusScore;
        }
      }

      if (bestFocusItem && bestFocusScore) {
        // Step 2: Compute max scores across all current primaries
        const allPrimaries = items.filter(i => i.role === 'primary');
        const maxRoleScore = allPrimaries.length > 0
          ? Math.max(...allPrimaries.map(i => i.role_score))
          : 0;
        const maxRerankScore = allPrimaries.length > 0
          ? Math.max(...allPrimaries.map(i => i.rerank_score))
          : 0;

        // Step 3: Force bestFocusItem as primary with deterministically highest scores
        bestFocusItem.role = 'primary';
        bestFocusItem.role_score = maxRoleScore + 1.0;
        bestFocusItem.rerank_score = maxRerankScore + 1.0;

        // Tag with reason codes
        if (!bestFocusItem.reasons.some(r => r.code === 'focus_profile_positive')) {
          bestFocusItem.reasons.push({
            code: 'focus_profile_positive',
            message: `Focus profile match: matched=[${bestFocusScore.matched_terms.join(',')}]`,
          });
        }
        bestFocusItem.reasons.push({
          code: 'focus_primary_forced',
          message: `Forced primary: profile=${queryFocus.focus_profile.id}, focus_score=${bestFocusScore.score.toFixed(3)}, role_score=${bestFocusItem.role_score.toFixed(3)}`,
        });

        // Step 4: For lookup-like intents, demote ALL other primaries to supporting
        for (const other of allPrimaries) {
          if (other.doc_id === bestFocusItem.doc_id) continue;
          other.role = 'supporting';
          other.reasons.push({
            code: 'focus_precision_demoted',
            message: `Demoted: non-focus primary displaced by focus-forced ${bestFocusItem.doc_id}`,
          });
        }

        primaryCount = items.filter(i => i.role === 'primary').length;

        warnings.push(`Focus precision FORCED: ${bestFocusItem.doc_id} is now sole primary (profile=${queryFocus.focus_profile.id}, focus_score=${bestFocusScore.score.toFixed(3)})`);
        console.log(`   🎯 Focus FORCED: profile=${queryFocus.focus_profile.id}, primary=${bestFocusItem.doc_id}, focus_score=${bestFocusScore.score.toFixed(3)}, demoted=${allPrimaries.filter(p => p.doc_id !== bestFocusItem!.doc_id).map(p => p.doc_id).join(',') || 'none'}`);
      } else {
        // No focus-positive candidate found
        const finalPrimaryId = items.find(i => i.role === 'primary')?.doc_id ?? 'none';
        console.log(`   🎯 Focus profile ${queryFocus.focus_profile.id}: primary=${finalPrimaryId}, NO focus-positive candidate found`);
        warnings.push(`FOCUS_ALIGNMENT_CHECK: profile=${queryFocus.focus_profile.id}, no focus-positive candidate in items`);
      }
    } else if (suppressGenericProfile && treatySubtopic && !isDisambigProtected && !isComparisonIntent) {
      // ── Patch 7M-A + 7M-B: SUBTOPIC-AWARE force — replaces generic treaty force ──
      // Find best candidate matching SUBTOPIC positive terms (not generic treaty terms)
      // 7M-B: Add Vietnam-scope guard and strong-term requirement for regrouping
      let bestSubtopicItem: typeof items[0] | null = null;
      let bestSubtopicPosCount = 0;
      let bestSubtopicStrongCount = 0;
      let bestSubtopicRerankScore = -1;

      // 7M-B: Vietnam-scope guard — check if query mentions Lào/Campuchia
      const queryNormScope = normalizeForFocus(query ?? '');
      const queryMentionsLao = queryNormScope.includes('lào') || queryNormScope.includes('laos');
      const queryMentionsCambodia = queryNormScope.includes('campuchia') || queryNormScope.includes('cambodia');

      for (const item of items) {
        if (item.role === 'excluded' || item.role === 'contrast') continue;
        if (item.reasons.some(r => r.code === 'hard_negative_excluded' || r.code === 'forbidden')) continue;

        const doc = resolved.find(r => r.doc.doc_id === item.doc_id)?.doc;
        if (!doc) continue;

        const docText = normalizeForFocus(
          `${doc.title} ${doc.summary} ${doc.text_for_embedding || ''}`
        );

        // 7M-B: Vietnam-scope guard — demote non-Vietnam Genève docs
        // If query is about Vietnamese treaty and doesn't mention Lào/Campuchia,
        // skip docs primarily about Lào/Campuchia for primary force
        const docTitleNorm = normalizeForFocus(doc.title || '');
        const docIsLao = docTitleNorm.includes('lào') || docTitleNorm.includes('laos');
        const docIsCambodia = docTitleNorm.includes('campuchia') || docTitleNorm.includes('cambodia');
        if (docIsLao && !queryMentionsLao) {
          console.log(`   🏛️ Treaty scope guard: skipping "${doc.title}" (Lào doc for Vietnam query)`);
          continue;
        }
        if (docIsCambodia && !queryMentionsCambodia) {
          console.log(`   🏛️ Treaty scope guard: skipping "${doc.title}" (Campuchia doc for Vietnam query)`);
          continue;
        }

        // Count subtopic positive matches
        const posCount = treatySubtopic.positive_terms.filter(t =>
          docText.includes(normalizeForFocus(t))
        ).length;

        // 7M-B: Count strong positive matches (for regrouping subtopic)
        const strongTerms = treatySubtopic.strong_positive_terms ?? treatySubtopic.positive_terms;
        const strongCount = strongTerms.filter(t =>
          docText.includes(normalizeForFocus(t))
        ).length;

        // Check if doc has ONLY subtopic negative terms (no positive)
        const hasOnlyNeg = posCount === 0 &&
          treatySubtopic.negative_terms.some(t => docText.includes(normalizeForFocus(t)));

        if (posCount > 0 && !hasOnlyNeg) {
          // 7M-B: For regrouping, require at least 1 strong term
          if (treatySubtopic.strong_positive_terms && strongCount === 0) {
            // Doc has only weak terms (e.g., "đình chiến") — not eligible for primary force
            continue;
          }

          // Prefer higher strongCount, then posCount, then rerank_score
          const betterStrong = strongCount > bestSubtopicStrongCount;
          const sameStrongBetterPos = strongCount === bestSubtopicStrongCount && posCount > bestSubtopicPosCount;
          const sameAllBetterRerank = strongCount === bestSubtopicStrongCount && posCount === bestSubtopicPosCount && item.rerank_score > bestSubtopicRerankScore;

          if (betterStrong || sameStrongBetterPos || sameAllBetterRerank) {
            bestSubtopicItem = item;
            bestSubtopicPosCount = posCount;
            bestSubtopicStrongCount = strongCount;
            bestSubtopicRerankScore = item.rerank_score;
          }
        }
      }

      if (bestSubtopicItem && bestSubtopicPosCount >= 1) {
        // Force subtopic-positive doc as primary
        const allPrimaries = items.filter(i => i.role === 'primary');
        const maxRoleScore = allPrimaries.length > 0
          ? Math.max(...allPrimaries.map(i => i.role_score))
          : 0;
        const maxRerankScore = allPrimaries.length > 0
          ? Math.max(...allPrimaries.map(i => i.rerank_score))
          : 0;

        bestSubtopicItem.role = 'primary';
        bestSubtopicItem.role_score = maxRoleScore + 1.0;
        bestSubtopicItem.rerank_score = maxRerankScore + 1.0;
        bestSubtopicItem.reasons.push({
          code: 'treaty_subtopic_positive',
          message: `Subtopic match: ${treatySubtopic.subtopic}, pos_count=${bestSubtopicPosCount}, strong_count=${bestSubtopicStrongCount}`,
        });
        bestSubtopicItem.reasons.push({
          code: 'treaty_subtopic_primary_forced',
          message: `Forced primary: subtopic=${treatySubtopic.subtopic}, generic profile suppressed=${queryFocus.focus_profile?.id}`,
        });

        // Demote all other primaries
        for (const other of allPrimaries) {
          if (other.doc_id === bestSubtopicItem.doc_id) continue;
          other.role = 'supporting';
          other.reasons.push({
            code: 'treaty_subtopic_demoted',
            message: `Demoted: generic treaty primary displaced by subtopic-forced ${bestSubtopicItem.doc_id} (${treatySubtopic.subtopic})`,
          });
        }

        primaryCount = items.filter(i => i.role === 'primary').length;
        warnings.push(`Treaty subtopic FORCED: ${treatySubtopic.subtopic}, primary=${bestSubtopicItem.doc_id}, generic profile ${queryFocus.focus_profile?.id} suppressed`);
        console.log(`   🏛️ Treaty subtopic FORCED: ${treatySubtopic.subtopic}, primary=${bestSubtopicItem.doc_id}, pos=${bestSubtopicPosCount}, strong=${bestSubtopicStrongCount}, suppressed=${queryFocus.focus_profile?.id}`);
      } else {
        // No subtopic-positive candidate — do NOT fall back to generic treaty force
        const finalPrimaryId = items.find(i => i.role === 'primary')?.doc_id ?? 'none';
        console.log(`   🏛️ Treaty subtopic ${treatySubtopic.subtopic}: primary=${finalPrimaryId}, NO subtopic-positive candidate found (generic profile suppressed)`);
        warnings.push(`TREATY_SUBTOPIC_NO_MATCH: subtopic=${treatySubtopic.subtopic}, generic profile ${queryFocus.focus_profile?.id} suppressed, no subtopic-positive candidate`);
      }
    } else if (!isDisambigProtected && !isComparisonIntent) {
      // No focus profile — run legacy penalty-based swap
      const queryFocusLegacy = extractQueryFocus({ query: query ?? '', queryFrame });
      const primaryItems = items.filter(i => i.role === 'primary');
      const swapCandidates = items.filter(i => i.role !== 'excluded' && i.role !== 'primary');

      if (primaryItems.length > 0 && swapCandidates.length > 0) {
        const primaryDoc = resolved.find(r => r.doc.doc_id === primaryItems[0].doc_id)?.doc;
        if (primaryDoc) {
          const primaryFocusScore = scoreDocumentFocus(primaryDoc, queryFocusLegacy);

          if (primaryFocusScore.penalties.length > 0 && primaryFocusScore.score < 0) {
            let bestSwap: { item: typeof swapCandidates[0]; score: DocFocusScore } | null = null;
            for (const candItem of swapCandidates) {
              if (candItem.role === 'contrast') continue;
              const candDoc = resolved.find(r => r.doc.doc_id === candItem.doc_id)?.doc;
              if (!candDoc) continue;
              const candFocusScore = scoreDocumentFocus(candDoc, queryFocusLegacy);
              if (candFocusScore.score > primaryFocusScore.score + 0.05 && candFocusScore.penalties.length === 0) {
                if (!bestSwap || candFocusScore.score > bestSwap.score.score) {
                  bestSwap = { item: candItem, score: candFocusScore };
                }
              }
            }
            if (bestSwap) {
              primaryItems[0].role = 'supporting';
              primaryItems[0].reasons.push({ code: 'focus_precision_demoted', message: `Demoted: penalties` });
              bestSwap.item.role = 'primary';
              bestSwap.item.reasons.push({ code: 'focus_precision_promoted', message: `Promoted: penalty-swap` });
              bestSwap.item.role_score = Math.max(bestSwap.item.role_score, primaryItems[0].role_score + 0.01);
              bestSwap.item.rerank_score = Math.max(bestSwap.item.rerank_score, primaryItems[0].rerank_score + 0.01);
              primaryCount = items.filter(i => i.role === 'primary').length;
            }
          }
        }
      }
    }
  }

  // ── Safety: ensure at least 1 primary ──
  if (primaryCount === 0 && items.length > 0) {
    // For disambiguation/misconception: DON'T promote contrast docs to primary.
    // Find best non-excluded AND non-contrast candidate.
    const isDisambigLike = queryFrame.intent === 'disambiguation' || queryFrame.intent === 'misconception_check';

    const eligibleForPromotion = items.filter(i => {
      if (i.role === 'excluded') return false;
      if (isDisambigLike && i.role === 'contrast') return false;
      return true;
    });

    const best = eligibleForPromotion.sort((a, b) => b.rerank_score - a.rerank_score)[0];
    if (best) {
      best.role = 'primary';
      best.reasons.push({
        code: 'fallback_primary_promotion',
        message: 'No semantic-primary found — promoted top eligible (non-contrast) to primary',
      });
      warnings.push('No candidate matched answer_focus well — promoted top eligible result to primary.');
    } else if (!isDisambigLike) {
      // For non-disambiguation, can promote any non-excluded as last resort
      const lastResort = items.filter(i => i.role !== 'excluded')
        .sort((a, b) => b.rerank_score - a.rerank_score)[0];
      if (lastResort) {
        lastResort.role = 'primary';
        lastResort.reasons.push({
          code: 'fallback_primary_promotion',
          message: 'No primary found — last-resort promotion to primary',
        });
        warnings.push('No suitable primary candidate — last-resort promotion.');
      }
    }
  }

  return buildSelection(items, true, warnings);
}

// ─── Fallback (no QueryFrame) ────────────────────────────────────────────────

function fallbackSelection(
  resolved: ResolvedCandidate[],
  warnings: string[]
): EvidenceSelection {
  const items: EvidenceItem[] = [];

  for (let i = 0; i < resolved.length; i++) {
    const c = resolved[i];
    const role: EvidenceRole = i === 0 ? 'primary' : 'supporting';
    items.push({
      doc_id: c.doc.doc_id,
      role,
      role_score: c.result.rerank_score,
      reasons: [{ code: 'fallback_score_order', message: `Score-ordered position ${i}` }],
      rerank_score: c.result.rerank_score,
    });
  }

  return buildSelection(items, false, warnings);
}

// ─── Selection Assembly ──────────────────────────────────────────────────────

function buildSelection(
  items: EvidenceItem[],
  usedQueryFrame: boolean,
  warnings: string[]
): EvidenceSelection {
  const primary = items.filter(i => i.role === 'primary').sort((a, b) => b.role_score - a.role_score);
  const supporting = items.filter(i => i.role === 'supporting').sort((a, b) => b.role_score - a.role_score);
  const contrast = items.filter(i => i.role === 'contrast').sort((a, b) => b.role_score - a.role_score);
  const excluded = items.filter(i => i.role === 'excluded');

  // Ordered: primary → supporting → contrast (excluded not in ordered)
  const ordered = [...primary, ...supporting, ...contrast];

  return {
    primary,
    supporting,
    contrast,
    excluded,
    ordered,
    diagnostics: {
      used_query_frame: usedQueryFrame,
      primary_count: primary.length,
      supporting_count: supporting.length,
      contrast_count: contrast.length,
      excluded_count: excluded.length,
      warnings,
    },
  };
}
