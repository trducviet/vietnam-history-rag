/**
 * Context Builder — assembles curated context bundles from reranked retrieval
 * results. NEVER sends raw top-k to LLM.
 *
 * Enforced rules (from implementation plan v2):
 * 1. canonical > duplicate — only canonical docs in context
 * 2. verified > reviewed > unverified — priority by verification_status
 * 3. same-cluster / linked docs preferred
 * 4. planned_not_executed must be distinguished from actual events
 * 5. Raw top-k NEVER goes directly to the LLM
 *
 * PATCH 3 additions:
 * - Primary docs use text_for_embedding for richer content
 * - Supporting docs use summary for token efficiency
 * - Disambiguation notes for hard-negative / same-cluster confusion
 * - Citation plan for generator guidance
 */

import type {
  QueryIntent,
  RerankedResult,
  IndexableDocument,
  LoadedDataset,
  ContextBundle,
  DisambiguationNote,
  CitationPlanItem,
  QueryFrame,
} from '../shared/types.js';
import { expandComparisonSideTerms, normalizeVietnamesePhrase } from '../routing/query-frame-builder.js';
import type { BM25Index } from '../retrieval/bm25-index.js';

// Patch 9C: Optional BM25 indexes for side-specific rescue queries
export interface SideRescueBM25 {
  eventBM25: BM25Index;
  synthesisBM25: BM25Index;
}

// ─── Constants ───────────────────────────────────────────────

const MAX_PRIMARY_TEXT_CHARS = 1800;
const MAX_SUPPORTING_TEXT_CHARS = 700;
const MAX_LINKED_TEXT_CHARS = 500;
const MAX_DISAMBIGUATION_NOTES = 3;

// ─── Context Template Sizes ──────────────────────────────────

/** Min docs per intent type */
const CONTEXT_TEMPLATES: Record<QueryIntent, { primary: number; supporting: number }> = {
  fact_lookup:     { primary: 1, supporting: 1 },
  date_lookup:     { primary: 1, supporting: 1 },
  actor_lookup:    { primary: 1, supporting: 1 },
  location_lookup: { primary: 1, supporting: 1 },
  entity_profile:  { primary: 1, supporting: 3 },
  explanation:     { primary: 1, supporting: 2 },
  comparison:      { primary: 2, supporting: 2 },
  timeline:        { primary: 1, supporting: 5 },
  cause_effect:    { primary: 1, supporting: 2 },
  multi_hop:       { primary: 2, supporting: 3 },
};

// ─── Verification Priority ──────────────────────────────────

const VERIFICATION_ORDER: Record<string, number> = {
  verified: 0,
  reviewed: 1,
  unverified: 2,
};

function verificationPriority(status: string): number {
  return VERIFICATION_ORDER[status] ?? 3;
}

// ─── Text Helpers ────────────────────────────────────────────

/** Truncate text to a character limit without cutting mid-word */
function truncateText(text: string, limit: number): string {
  if (!text) return '';
  if (text.length <= limit) return text;
  return text.slice(0, limit).trimEnd() + '...';
}

/**
 * Get the appropriate context text for a document based on its role.
 * - Primary: uses text_for_embedding (richer), falls back to summary
 * - Supporting/linked: uses summary (compact), falls back to text_for_embedding
 */
function getDocContextText(
  doc: IndexableDocument,
  role: 'primary' | 'supporting' | 'linked'
): string {
  const limit =
    role === 'primary'
      ? MAX_PRIMARY_TEXT_CHARS
      : role === 'supporting'
        ? MAX_SUPPORTING_TEXT_CHARS
        : MAX_LINKED_TEXT_CHARS;

  const raw =
    role === 'primary'
      ? (doc.text_for_embedding || doc.summary || '')
      : (doc.summary || doc.text_for_embedding || '');

  return truncateText(raw, limit);
}

// ─── Candidate Doc Type (Patch 7E-2) ─────────────────────────

/** Resolved candidate with optional evidence role annotations */
interface CandidateDoc {
  doc: IndexableDocument;
  rerankScore: number;
  reason?: string;
  evidenceRole?: string;
  evidenceRoleScore?: number;
  /** 7L-E: Evidence reason codes from EvidenceSelector */
  evidenceReasons?: string[];
}

// ─── Context Builder ─────────────────────────────────────────

/**
 * Build a curated context bundle from reranked results.
 *
 * ⚠️ NEVER send raw top-k results directly to the LLM.
 * This function enforces priority rules, cluster coherence,
 * and planned_not_executed distinction.
 *
 * Patch 7E-2: Evidence-aware mode — when candidates carry evidence_role
 * annotations, primary_docs only come from evidence_role='primary'.
 * Contrast docs are omitted from citation context for disambiguation.
 */
export function buildContextBundle(
  intent: QueryIntent,
  rerankedResults: RerankedResult[],
  dataset: LoadedDataset,
  queryFrame?: QueryFrame,
  bm25Indexes?: SideRescueBM25   // Patch 9C: optional BM25 for side rescue
): ContextBundle {
  const template = CONTEXT_TEMPLATES[intent];
  const warnings: string[] = [];

  // Detect evidence-aware mode (Patch 7E-2)
  const hasEvidenceRoles = rerankedResults.some(r => r.evidence_role != null);

  // Step 1: Resolve reranked IDs to full documents, enforce CANONICAL only
  const candidateDocs: CandidateDoc[] = [];

  for (const result of rerankedResults) {
    const doc = resolveDocument(result.doc_id, dataset);
    if (!doc) {
      warnings.push(`Document ${result.doc_id} not found in dataset`);
      continue;
    }
    if (!doc.canonical) {
      warnings.push(`Skipped non-canonical document ${result.doc_id}`);
      continue;
    }
    candidateDocs.push({
      doc,
      rerankScore: result.rerank_score,
      reason: result.reason,
      evidenceRole: result.evidence_role,
      evidenceRoleScore: result.evidence_role_score,
      evidenceReasons: result.evidence_reasons,
    });
  }

  // ── Evidence-aware mode (Patch 7E-2+7F) ──
  if (hasEvidenceRoles) {
    return buildEvidenceAwareBundle(intent, template, candidateDocs, dataset, warnings, queryFrame, bm25Indexes);
  }

  // ── Legacy mode (no evidence annotations) ──
  return buildLegacyBundle(intent, template, candidateDocs, dataset, warnings);
}

// ─── Evidence-Aware Bundle (Patch 7E-2) ──────────────────────

/**
 * Build context bundle using evidence role annotations.
 *
 * Rules:
 * 1. primary_docs: ONLY from evidenceRole='primary'. Max template.primary.
 *    Do NOT fill from supporting/contrast just to meet template quota.
 * 2. supporting_docs: from evidenceRole='supporting'.
 *    For comparison: also include evidenceRole='contrast' after supporting.
 *    For disambiguation/misconception: EXCLUDE contrast docs entirely.
 * 3. excluded: never included.
 * 4. Contrast docs omitted from citation context for disambiguation/misconception.
 */
function buildEvidenceAwareBundle(
  intent: QueryIntent,
  template: { primary: number; supporting: number },
  candidateDocs: CandidateDoc[],
  dataset: LoadedDataset,
  warnings: string[],
  queryFrame?: QueryFrame,
  bm25Indexes?: SideRescueBM25   // Patch 9C
): ContextBundle {
  const evidenceWarnings: string[] = [];
  evidenceWarnings.push('Evidence-aware mode active.');

  const isDisambigLike = ['multi_hop', 'disambiguation', 'misconception_check'].includes(intent);
  const isComparison = intent === 'comparison';
  const isTimeline = intent === 'timeline';
  const isLookup = ['fact_lookup', 'date_lookup', 'actor_lookup', 'location_lookup', 'entity_profile'].includes(intent);

  // Sort within each group by evidenceRoleScore DESC, then rerankScore DESC
  const sortByScore = (
    a: { evidenceRoleScore?: number; rerankScore: number },
    b: { evidenceRoleScore?: number; rerankScore: number }
  ) => {
    const rsDiff = (b.evidenceRoleScore ?? 0) - (a.evidenceRoleScore ?? 0);
    if (Math.abs(rsDiff) > 0.001) return rsDiff;
    return b.rerankScore - a.rerankScore;
  };

  // Partition by evidence role
  const evPrimary = candidateDocs.filter(c => c.evidenceRole === 'primary').sort(sortByScore);
  const evSupporting = candidateDocs.filter(c => c.evidenceRole === 'supporting').sort(sortByScore);
  const evContrast = candidateDocs.filter(c => c.evidenceRole === 'contrast').sort(sortByScore);

  // ── Primary docs: ONLY from evidence_role='primary' ──
  const primaryDocs: IndexableDocument[] = [];
  const usedIds = new Set<string>();

  // For timeline/explanation: prefer synthesis primary docs first
  const synthPrimary = evPrimary.filter(c => c.doc.doc_source === 'synthesis');
  const eventPrimary = evPrimary.filter(c => c.doc.doc_source !== 'synthesis');
  let orderedPrimary = (intent === 'timeline' || intent === 'explanation' || intent === 'cause_effect')
    ? [...synthPrimary, ...eventPrimary]
    : [...evPrimary];

  // 7L-E2: Focus-priority reorder — if any primary has focus reason codes, move it to the front.
  // Recognizes: focus_primary_forced (7L-E2: deterministically forced),
  //             focus_precision_promoted (swapped from supporting),
  //             focus_profile_positive (matched profile+ terms),
  //             focus_primary_preferred (was primary but boosted).
  // Does NOT apply to comparison (needs both sides) or disambiguation (protected).
  if (!isComparison && !isDisambigLike) {
    const FOCUS_REASON_CODES = ['focus_primary_forced', 'focus_precision_promoted', 'focus_profile_positive', 'focus_primary_preferred'];
    const focusPromotedIdx = orderedPrimary.findIndex(c =>
      c.evidenceReasons?.some(r => FOCUS_REASON_CODES.includes(r))
    );
    if (focusPromotedIdx > 0) {
      const [promoted] = orderedPrimary.splice(focusPromotedIdx, 1);
      orderedPrimary.unshift(promoted);
      evidenceWarnings.push(`Focus primary forced to front: ${promoted.doc.doc_id}`);
    }
  }

  // Patch 7K: For queries with sides, ensure primary docs cover both sides
  // Patch 7K-A: NEVER apply two-sided logic for disambiguation — contrast must be omitted
  // Patch 8D: Apply for comparison and multi_hop/fact_lookup with comparison_sides
  //           but NOT disambiguation with 'khác với' pattern (which needs primary-focused context)
  //           Disambiguation with 'có phải' pattern (identity questions) DO need two-sided
  //           Note: routing maps disambiguation→multi_hop, so check both routing intent AND frame intent
  const disambigFrameIntents = ['disambiguation', 'misconception_check'];
  const frameIntent = queryFrame?.intent ?? '';
  const isDisambigFrame = disambigFrameIntents.includes(intent) ||
    (isDisambigLike && disambigFrameIntents.includes(frameIntent));
  // 'khác với' disambiguation = primary-focused (exclude sideB from context)
  // 'có phải' disambiguation = identity check (include both sides)
  const sideMarker = queryFrame?.comparison_sides?.marker ?? '';
  const isPrimaryFocusedDisambig = isDisambigFrame && !['có phải', 'có phải cùng', 'có giống nhau'].includes(sideMarker);
  const compSides = (!isPrimaryFocusedDisambig) ? queryFrame?.comparison_sides : undefined;
  const isTwoSided = !!compSides;
  if (isTwoSided && compSides) {
    const sideATerms = expandComparisonSideTerms(compSides.side_a);
    const sideBTerms = expandComparisonSideTerms(compSides.side_b);

    // Patch 8D-A: Discriminator-scored side classification.
    // Returns score 0..100: 0=no match, higher=stronger match.
    // When multi-word discriminator phrases are available (e.g. "biên giới 1950",
    // "chiến dịch biên giới"), at least ONE phrase must match (title or body)
    // for the doc to be considered a side match. Token-only matches like
    // "chiến", "dịch", "biên" are too generic and cause false positives
    // (e.g. "chiến dịch Điện Biên Phủ" incorrectly matching sideA "Biên giới").
    const scoreSideMatch = (doc: IndexableDocument, sideTerms: string[]): number => {
      const titleText = doc.title.toLowerCase().normalize('NFKC');
      const fullText = `${doc.title} ${doc.summary} ${doc.text_for_embedding || ''}`.toLowerCase().normalize('NFKC');
      const multiWordTerms = sideTerms.filter(t => t.length > 5 && t.includes(' '));
      const longTokens = sideTerms.filter(t => t.length > 3);

      // Count multi-word phrase matches
      const titleMultiMatches = multiWordTerms.filter(t => titleText.includes(t)).length;
      const bodyMultiMatches = multiWordTerms.filter(t => fullText.includes(t)).length;
      const totalMultiMatches = bodyMultiMatches; // body includes title

      // GATE: If multi-word discriminators exist but NONE match anywhere,
      // this doc is NOT about this side — return 0 regardless of token matches.
      if (multiWordTerms.length > 0 && totalMultiMatches === 0) {
        return 0;
      }

      let score = 0;

      // Title multi-word matches are strong (doc is ABOUT this topic)
      score += titleMultiMatches * 30;

      // Body-only multi-word matches are weak (doc merely mentions topic)
      const bodyOnlyMulti = bodyMultiMatches - titleMultiMatches;
      score += bodyOnlyMulti * 3;

      // Token matches add supplementary signal
      const titleTokenMatches = longTokens.filter(t => titleText.includes(t)).length;
      score += titleTokenMatches * 5;

      return score;
    };

    // Binary convenience wrapper using minimum score threshold
    const matchesSideExpanded = (doc: IndexableDocument, sideTerms: string[]): boolean => {
      return scoreSideMatch(doc, sideTerms) >= 5;
    };

    // 7N-B: Noise-year filter — if both sides have specific years,
    // skip docs whose title contains year-specific entities NOT in either side
    const sideAYears = (compSides.side_a.match(/\b(1[89]\d{2}|20\d{2})\b/g) ?? []);
    const sideBYears = (compSides.side_b.match(/\b(1[89]\d{2}|20\d{2})\b/g) ?? []);
    const allowedYears = new Set([...sideAYears, ...sideBYears, '1858', '1945', '1954', '1975']);
    const hasStrictYearScope = sideAYears.length > 0 && sideBYears.length > 0;

    const isNoiseYearDoc = (doc: IndexableDocument): boolean => {
      if (!hasStrictYearScope) return false;
      const titleYears = (doc.title.match(/\b(1[89]\d{2}|20\d{2})\b/g) ?? []);
      return titleYears.some(y => !allowedYears.has(y));
    };

    // Find best candidate for each side, using discriminator score
    // Patch 8D-A: Use highest-score selection instead of first-match
    const allCandidates = [...orderedPrimary, ...evSupporting, ...evContrast];
    let bestA: CandidateDoc | undefined;
    let bestAScore = 0;
    let bestB: CandidateDoc | undefined;
    let bestBScore = 0;

    for (const c of allCandidates) {
      if (isNoiseYearDoc(c.doc)) continue;
      const sa = scoreSideMatch(c.doc, sideATerms);
      const sb = scoreSideMatch(c.doc, sideBTerms);
      const ma = sa >= 5;
      const mb = sb >= 5;
      if (ma && mb) {
        // Synthesis doc covering both sides → ideal primary
        primaryDocs.push(c.doc);
        usedIds.add(c.doc.doc_id);
        break;
      }
      // Patch 8D-A: Prefer doc with highest discriminator score for each side
      if (ma && sa > bestAScore) { bestA = c; bestAScore = sa; }
      if (mb && sb > bestBScore) { bestB = c; bestBScore = sb; }
    }

    // If no single doc covers both, add one per side
    if (primaryDocs.length === 0) {
      if (bestA) { primaryDocs.push(bestA.doc); usedIds.add(bestA.doc.doc_id); }
      if (bestB && !usedIds.has(bestB.doc.doc_id)) { primaryDocs.push(bestB.doc); usedIds.add(bestB.doc.doc_id); }
    }

    // Fallback: if still no primary, use top-scored (but skip noise docs)
    if (primaryDocs.length === 0) {
      for (const c of orderedPrimary) {
        if (primaryDocs.length >= template.primary) break;
        if (!isNoiseYearDoc(c.doc)) {
          primaryDocs.push(c.doc);
          usedIds.add(c.doc.doc_id);
        }
      }
    }

    // Warn if only one side found
    const hasSideA = primaryDocs.some(d => matchesSideExpanded(d, sideATerms));
    const hasSideB = primaryDocs.some(d => matchesSideExpanded(d, sideBTerms));
    if (!hasSideA) evidenceWarnings.push(`Comparison: missing evidence for side A "${compSides.side_a}"`);
    if (!hasSideB) evidenceWarnings.push(`Comparison: missing evidence for side B "${compSides.side_b}"`);
  } else {
    for (const c of orderedPrimary) {
      if (primaryDocs.length >= template.primary) break;
      primaryDocs.push(c.doc);
      usedIds.add(c.doc.doc_id);
    }
  }

  // ── Supporting docs with semantic guard ──
  const supportingDocs: IndexableDocument[] = [];
  let omittedSupportingCount = 0;

  // Patch 8D: For two-sided queries, increase supporting capacity to ensure side coverage
  const supportingCap = isTwoSided ? Math.max(template.supporting, 4) : template.supporting;

  for (const c of evSupporting) {
    if (supportingDocs.length >= supportingCap) break;
    if (usedIds.has(c.doc.doc_id)) continue;

    // Patch 8D: Skip semantic guard for two-sided queries — side docs must be preserved
    if (!isTwoSided && !shouldIncludeSupportingDoc(c, queryFrame, intent, isDisambigLike, isLookup)) {
      omittedSupportingCount++;
      continue;
    }

    supportingDocs.push(c.doc);
    usedIds.add(c.doc.doc_id);
  }

  // Patch 8D: Include contrast docs as supporting for ALL two-sided queries (not just comparison)
  if (isTwoSided || isComparison) {
    for (const c of evContrast) {
      if (supportingDocs.length >= supportingCap + 2) break;
      if (usedIds.has(c.doc.doc_id)) continue;
      supportingDocs.push(c.doc);
      usedIds.add(c.doc.doc_id);
    }
  }

  // ── Patch 8D: Side promotion — ensure both sides are represented ──
  // Patch 8D-A: Use title-priority scoring for coverage check (same as primary selection)
  if (isTwoSided && compSides) {
    const sideATerms = expandComparisonSideTerms(compSides.side_a);
    const sideBTerms = expandComparisonSideTerms(compSides.side_b);
    // Reuse same discriminator-scored logic as primary selection
    // Patch 9C-R Final: use normalizeVietnamesePhrase for hyphen/spacing normalization
    const matchSideTitle = (doc: IndexableDocument, terms: string[]): boolean => {
      const titleText = normalizeVietnamesePhrase(doc.title);
      const multiWordTerms = terms.filter(t => t.length > 5 && t.includes(' '));
      // Strong match: title contains a discriminator phrase
      if (multiWordTerms.some(t => titleText.includes(t))) return true;
      // Medium match: title contains 2+ long tokens
      const longTokens = terms.filter(t => t.length > 3);
      const titleMatches = longTokens.filter(t => titleText.includes(t)).length;
      return titleMatches >= 2;
    };
    const matchSideAny = (doc: IndexableDocument, terms: string[]): boolean => {
      const fullText = normalizeVietnamesePhrase(`${doc.title} ${doc.summary} ${doc.text_for_embedding || ''}`);
      const multiWordTerms = terms.filter(t => t.length > 5 && t.includes(' '));
      if (multiWordTerms.length > 0) return multiWordTerms.some(t => fullText.includes(t));
      const longTokens = terms.filter(t => t.length > 3);
      return longTokens.filter(t => fullText.includes(t)).length >= 2;
    };
    const allCtx = [...primaryDocs, ...supportingDocs];
    // Patch 8D-A: Check title-level coverage first; fall back to body-level
    const ctxHasSideA = allCtx.some(d => matchSideTitle(d, sideATerms)) || allCtx.some(d => matchSideAny(d, sideATerms));
    const ctxHasSideB = allCtx.some(d => matchSideTitle(d, sideBTerms)) || allCtx.some(d => matchSideAny(d, sideBTerms));

    // Promote missing side from evidence pool
    // Patch 8D-A: Prefer title-matching docs over body-only matching docs
    const evidencePool = [...evSupporting, ...evContrast];
    if (!ctxHasSideA) {
      const bestATitle = evidencePool.find(c => !usedIds.has(c.doc.doc_id) && matchSideTitle(c.doc, sideATerms));
      const bestABody = !bestATitle ? evidencePool.find(c => !usedIds.has(c.doc.doc_id) && matchSideAny(c.doc, sideATerms)) : undefined;
      const bestA = bestATitle ?? bestABody;
      if (bestA) {
        supportingDocs.push(bestA.doc);
        usedIds.add(bestA.doc.doc_id);
        evidenceWarnings.push(`Side-promoted sideA doc: ${bestA.doc.doc_id}`);
      }
    }
    if (!ctxHasSideB) {
      const bestBTitle = evidencePool.find(c => !usedIds.has(c.doc.doc_id) && matchSideTitle(c.doc, sideBTerms));
      const bestBBody = !bestBTitle ? evidencePool.find(c => !usedIds.has(c.doc.doc_id) && matchSideAny(c.doc, sideBTerms)) : undefined;
      const bestB = bestBTitle ?? bestBBody;
      if (bestB) {
        supportingDocs.push(bestB.doc);
        usedIds.add(bestB.doc.doc_id);
        evidenceWarnings.push(`Side-promoted sideB doc: ${bestB.doc.doc_id}`);
      }
    }

    // ── Patch 9C: BM25 side-specific rescue ──
    // If after side promotion, one side is still missing, query BM25 directly
    // using the side's text. No API call, no vector search — BM25 in-memory only.
    if (bm25Indexes) {
      const allCtxAfterPromo = [...primaryDocs, ...supportingDocs];
      const promoHasSideA = allCtxAfterPromo.some(d => matchSideAny(d, sideATerms));
      const promoHasSideB = allCtxAfterPromo.some(d => matchSideAny(d, sideBTerms));

      const rescueSide = (sideLabel: string, sideTerms: string[], sideName: string) => {
        // Use multi-word phrases as BM25 query for precision
        const multiPhrases = sideTerms.filter(t => t.includes(' ') && t.length > 5);
        const queryText = multiPhrases.length > 0 ? multiPhrases.join(' ') : sideName;
        
        // Search both event and synthesis BM25 indexes
        const eventResults = bm25Indexes.eventBM25.search(queryText, 3);
        const synthResults = bm25Indexes.synthesisBM25.search(queryText, 3);
        const allBm25 = [...eventResults, ...synthResults]
          .sort((a, b) => b.score - a.score);

        for (const bm25Hit of allBm25) {
          if (usedIds.has(bm25Hit.doc_id)) continue;
          const doc = dataset.events.get(bm25Hit.doc_id) ?? dataset.synthesis.get(bm25Hit.doc_id);
          if (!doc || !doc.canonical) continue;
          // Verify the doc actually matches this side
          if (!matchSideAny(doc, sideTerms)) continue;
          
          supportingDocs.push(doc);
          usedIds.add(doc.doc_id);
          evidenceWarnings.push(`BM25_SIDE_RESCUE: rescued ${sideLabel} doc ${doc.doc_id} ("${doc.title}") via BM25 query "${queryText.substring(0, 50)}"`);
          return true;
        }
        evidenceWarnings.push(`BM25_SIDE_RESCUE: no suitable ${sideLabel} doc found via BM25 for "${sideName}"`);
        return false;
      };

      if (!promoHasSideA) rescueSide('sideA', sideATerms, compSides.side_a);
      if (!promoHasSideB) rescueSide('sideB', sideBTerms, compSides.side_b);
    }

    // ── Patch 9C: Final side coverage verdict ──
    const finalCtx = [...primaryDocs, ...supportingDocs];
    const finalHasSideA = finalCtx.some(d => matchSideAny(d, sideATerms));
    const finalHasSideB = finalCtx.some(d => matchSideAny(d, sideBTerms));
    const hasBothDoc = finalCtx.some(d => matchSideAny(d, sideATerms) && matchSideAny(d, sideBTerms));

    let coverageVerdict: string;
    if (finalHasSideA && finalHasSideB && !hasBothDoc) {
      coverageVerdict = 'FULL_TWO_SIDED';
    } else if (hasBothDoc) {
      coverageVerdict = 'BOTH_DOC_COVERS';
    } else if (finalHasSideA && !finalHasSideB) {
      coverageVerdict = 'PARTIAL_INSUFFICIENT_MISSING_SIDE_B';
    } else if (!finalHasSideA && finalHasSideB) {
      coverageVerdict = 'PARTIAL_INSUFFICIENT_MISSING_SIDE_A';
    } else {
      coverageVerdict = 'NEEDS_REVIEW';
    }
    evidenceWarnings.push(`SIDE_COVERAGE_VERDICT: ${coverageVerdict}`);
  }

  // Track omitted contrast doc IDs
  const omittedDocIds: string[] = [];
  if (isDisambigLike && evContrast.length > 0) {
    for (const c of evContrast) {
      omittedDocIds.push(c.doc.doc_id);
    }
    evidenceWarnings.push(
      `Omitted ${evContrast.length} contrast document(s) from disambiguation citation context.`
    );
  }
  if (omittedSupportingCount > 0) {
    evidenceWarnings.push(
      `Omitted ${omittedSupportingCount} low-relevance supporting doc(s) from citation plan.`
    );
  }

  // ── Planned not executed ──
  const allDocs = [...primaryDocs, ...supportingDocs];
  const plannedDocs = allDocs.filter(d => d.event_status === 'planned_not_executed');
  if (plannedDocs.length > 0) {
    warnings.push(
      `${plannedDocs.length} document(s) có trạng thái "planned_not_executed" — ` +
      `sự kiện này được lên kế hoạch nhưng chưa/không thực hiện: ` +
      plannedDocs.map(d => d.doc_id).join(', ')
    );
  }

  // ── Patch 9G-R / 9G-R3: Supporting citation relevance post-filter ──
  // Remove supporting docs that have no focus overlap with primary/query
  // Patch 9G-R3: Stricter caps — strict lookups get 0, fact_lookup gets 1
  if (primaryDocs.length > 0 && supportingDocs.length > 0 && !isTwoSided && !isComparison) {
    const primaryTitle = primaryDocs[0].title.toLowerCase().normalize('NFKC');
    const focusTopic = (queryFrame?.answer_focus?.topic ?? '').toLowerCase().normalize('NFKC');
    const focusTokens = focusTopic.split(/\s+/).filter(t => t.length > 3);

    // Build query entity terms (from focus + primary title)
    const entityTerms = new Set<string>();
    focusTokens.forEach(t => entityTerms.add(t));
    primaryTitle.split(/\s+/).filter(t => t.length > 3).forEach(t => entityTerms.add(t));

    // Patch 9G-R3: Stricter caps to reduce citation pollution
    const isStrictLookup = ['date_lookup', 'actor_lookup', 'location_lookup', 'clause_lookup'].includes(intent);
    const maxSupporting = isStrictLookup ? 0 : (intent === 'fact_lookup' ? 1 : (isTimeline ? 3 : 2));

    // Score each supporting doc by entity overlap
    const scored = supportingDocs.map(doc => {
      const docText = `${doc.title} ${doc.summary ?? ''}`.toLowerCase().normalize('NFKC');
      let overlap = 0;
      for (const term of entityTerms) {
        if (docText.includes(term)) overlap++;
      }
      return { doc, overlap };
    });

    // Sort by overlap (desc), then take top N
    scored.sort((a, b) => b.overlap - a.overlap);

    // Patch 9G-R3: Require minimum 3 entity term overlaps (stricter than 2)
    const minOverlap = entityTerms.size >= 4 ? 3 : 2;
    const filtered = scored
      .filter(s => s.overlap >= minOverlap || entityTerms.size < 3)
      .slice(0, maxSupporting)
      .map(s => s.doc);

    // Replace supporting docs if we filtered anything
    if (filtered.length < supportingDocs.length) {
      const removedCount = supportingDocs.length - filtered.length;
      supportingDocs.length = 0;
      filtered.forEach(d => supportingDocs.push(d));
      evidenceWarnings.push(`Patch 9G-R: Filtered ${removedCount} low-relevance supporting doc(s).`);
    }
  }

  // ── Timeline sort ──
  if (isTimeline) {
    supportingDocs.sort((a, b) => (a.year ?? 9999) - (b.year ?? 9999));
  }

  // ── Disambiguation notes ──
  const candidateDocsLegacy = candidateDocs.map(c => ({ doc: c.doc, rerankScore: c.rerankScore, reason: c.reason }));
  const disambiguationNotes = buildDisambiguationNotes(intent, candidateDocsLegacy, usedIds, dataset);

  // ── Citation plan (role-aware strict, Patch 7F-1) ──
  // Patch 8D: Pass isTwoSided for side-aware citation cap
  const citationPlan = buildRoleAwareCitationPlan(
    primaryDocs, supportingDocs, isComparison || isTwoSided,
    evContrast.map(c => c.doc),
    dataset, intent, queryFrame, evSupporting, isTwoSided
  );
  const citedCount = citationPlan.filter(i => i.citation_role !== 'excluded').length;
  if (isLookup && !isTwoSided && primaryDocs.length >= 1) {
    evidenceWarnings.push(`Citation plan strict lookup mode: P=${primaryDocs.length} cited=${citedCount}`);
  } else {
    evidenceWarnings.push(`Citation plan: P=${primaryDocs.length} S=${supportingDocs.length} cited=${citedCount} omitted=${omittedDocIds.length}`);
  }

  // Quality warnings
  if (primaryDocs.length === 0) {
    warnings.push('Evidence-aware mode: no primary evidence documents found.');
  }

  // ── Context text ──
  const contextText = assembleContextText(
    intent, primaryDocs, supportingDocs, warnings, disambiguationNotes, citationPlan
  );

  return {
    intent,
    primary_docs: primaryDocs,
    supporting_docs: supportingDocs,
    planned_not_executed_docs: plannedDocs,
    context_text: contextText,
    included_doc_ids: [...usedIds],
    warnings,
    disambiguation_notes: disambiguationNotes.length > 0 ? disambiguationNotes : undefined,
    citation_plan: citationPlan.length > 0 ? citationPlan : undefined,
    omitted_doc_ids: omittedDocIds.length > 0 ? omittedDocIds : undefined,
    evidence_warnings: evidenceWarnings,
  };
}

// ─── Legacy Bundle (pre-7E-2) ────────────────────────────────

/** Legacy context building without evidence role awareness */
function buildLegacyBundle(
  intent: QueryIntent,
  template: { primary: number; supporting: number },
  candidateDocs: CandidateDoc[],
  dataset: LoadedDataset,
  warnings: string[]
): ContextBundle {
  // Step 2: Sort by priority: rerank_score DESC, verification ASC, year ASC
  candidateDocs.sort((a, b) => {
    const scoreDiff = b.rerankScore - a.rerankScore;
    if (Math.abs(scoreDiff) > 0.01) return scoreDiff;
    const verDiff = verificationPriority(a.doc.verification_status)
                  - verificationPriority(b.doc.verification_status);
    if (verDiff !== 0) return verDiff;
    return (a.doc.year ?? 9999) - (b.doc.year ?? 9999);
  });

  // Step 3: Select primary docs
  const primaryDocs: IndexableDocument[] = [];
  const usedIds = new Set<string>();

  if (['explanation', 'comparison', 'timeline', 'cause_effect', 'entity_profile'].includes(intent)) {
    const synthCandidates = candidateDocs.filter(c => c.doc.doc_source === 'synthesis');
    const eventCandidates = candidateDocs.filter(c => c.doc.doc_source === 'event');
    for (const c of synthCandidates) {
      if (primaryDocs.length >= template.primary) break;
      primaryDocs.push(c.doc);
      usedIds.add(c.doc.doc_id);
    }
    for (const c of eventCandidates) {
      if (primaryDocs.length >= template.primary) break;
      primaryDocs.push(c.doc);
      usedIds.add(c.doc.doc_id);
    }
  } else {
    for (const c of candidateDocs) {
      if (primaryDocs.length >= template.primary) break;
      primaryDocs.push(c.doc);
      usedIds.add(c.doc.doc_id);
    }
  }

  // Step 4: Select supporting docs
  const supportingDocs: IndexableDocument[] = [];
  const linkedDocIds = collectLinkedDocIds(primaryDocs, dataset);
  const linkedCandidates = candidateDocs.filter(
    c => !usedIds.has(c.doc.doc_id) && linkedDocIds.has(c.doc.doc_id)
  );
  const unlinkedCandidates = candidateDocs.filter(
    c => !usedIds.has(c.doc.doc_id) && !linkedDocIds.has(c.doc.doc_id)
  );
  for (const c of linkedCandidates) {
    if (supportingDocs.length >= template.supporting) break;
    supportingDocs.push(c.doc);
    usedIds.add(c.doc.doc_id);
  }
  for (const c of unlinkedCandidates) {
    if (supportingDocs.length >= template.supporting) break;
    supportingDocs.push(c.doc);
    usedIds.add(c.doc.doc_id);
  }

  // Step 4B (Stage 7E2): Rule context guarantee for disambiguation/comparison intents
  // Force-include best rule doc if comparison/multi_hop/cause_effect and none in context
  if (['comparison', 'multi_hop', 'cause_effect'].includes(intent)) {
    const hasRuleInContext = [...primaryDocs, ...supportingDocs].some(
      d => d.doc_type === 'disambiguation_rule' || d.doc_type === 'comparison_note'
    );
    if (!hasRuleInContext) {
      const bestRule = candidateDocs.find(
        c => !usedIds.has(c.doc.doc_id) &&
             (c.doc.doc_type === 'disambiguation_rule' || c.doc.doc_type === 'comparison_note')
      );
      if (bestRule) {
        supportingDocs.push(bestRule.doc);
        usedIds.add(bestRule.doc.doc_id);
        warnings.push(`Stage 7E2: Force-included rule doc ${bestRule.doc.doc_id} for disambiguation context.`);
      }
    }
  }

  // Step 5: planned_not_executed
  const allDocs = [...primaryDocs, ...supportingDocs];
  const plannedDocs = allDocs.filter(d => d.event_status === 'planned_not_executed');
  if (plannedDocs.length > 0) {
    warnings.push(
      `${plannedDocs.length} document(s) có trạng thái "planned_not_executed" — ` +
      `sự kiện này được lên kế hoạch nhưng chưa/không thực hiện: ` +
      plannedDocs.map(d => d.doc_id).join(', ')
    );
  }

  // Step 6: Timeline sort
  if (intent === 'timeline') {
    supportingDocs.sort((a, b) => (a.year ?? 9999) - (b.year ?? 9999));
  }

  // Step 7: Disambiguation notes
  const disambiguationNotes = buildDisambiguationNotes(intent, candidateDocs, usedIds, dataset);

  // Step 8: Citation plan
  const citationPlan = buildCitationPlan(primaryDocs, supportingDocs, dataset);

  // Step 9: Context text
  const contextText = assembleContextText(
    intent, primaryDocs, supportingDocs, warnings, disambiguationNotes, citationPlan
  );

  if (primaryDocs.length < template.primary) {
    warnings.push(`Chỉ tìm được ${primaryDocs.length}/${template.primary} tài liệu chính`);
  }
  if (supportingDocs.length < template.supporting) {
    warnings.push(`Chỉ tìm được ${supportingDocs.length}/${template.supporting} tài liệu hỗ trợ`);
  }

  return {
    intent,
    primary_docs: primaryDocs,
    supporting_docs: supportingDocs,
    planned_not_executed_docs: plannedDocs,
    context_text: contextText,
    included_doc_ids: [...usedIds],
    warnings,
    disambiguation_notes: disambiguationNotes.length > 0 ? disambiguationNotes : undefined,
    citation_plan: citationPlan.length > 0 ? citationPlan : undefined,
  };
}

// ─── Document Resolution ─────────────────────────────────────

/** Resolve a doc_id to full document, handling both events and synthesis */
function resolveDocument(docId: string, dataset: LoadedDataset): IndexableDocument | null {
  return dataset.events.get(docId) ?? dataset.synthesis.get(docId) ?? dataset.disambiguationRules.get(docId) ?? null;
}

// ─── Link Traversal ──────────────────────────────────────────

/** Collect doc IDs linked to the given primary docs via synthesis-event and entity links */
function collectLinkedDocIds(
  primaryDocs: IndexableDocument[],
  dataset: LoadedDataset
): Set<string> {
  const linked = new Set<string>();

  for (const doc of primaryDocs) {
    // Related events listed in the document itself
    for (const relId of doc.related_event_ids) {
      linked.add(relId);
    }

    // If it's a synthesis doc, get linked events
    const synthEvents = dataset.synthesisToEvents.get(doc.doc_id);
    if (synthEvents) {
      for (const eventId of synthEvents) linked.add(eventId);
    }

    // If it's an event, get linked synthesis docs
    const eventSynths = dataset.eventToSynthesis.get(doc.doc_id);
    if (eventSynths) {
      for (const synthId of eventSynths) linked.add(synthId);
    }

    // Same-cluster docs (rag_cluster)
    if (doc.rag_cluster) {
      const allDocs = [...dataset.events.values(), ...dataset.synthesis.values(), ...dataset.disambiguationRules.values()];
      for (const d of allDocs) {
        if (d.rag_cluster === doc.rag_cluster && d.doc_id !== doc.doc_id && d.canonical) {
          linked.add(d.doc_id);
        }
      }
    }
  }

  return linked;
}

// ─── Disambiguation Notes ────────────────────────────────────

/**
 * Build disambiguation notes to prevent generator confusion.
 * Checks for hard-negative flags, same-cluster near-misses,
 * and planned_not_executed documents in candidates.
 */
function buildDisambiguationNotes(
  _intent: QueryIntent,
  candidateDocs: Array<{ doc: IndexableDocument; rerankScore: number; reason?: string }>,
  selectedDocIds: Set<string>,
  dataset: LoadedDataset
): DisambiguationNote[] {
  const notes: DisambiguationNote[] = [];
  const seenDocIds = new Set<string>();

  for (const candidate of candidateDocs) {
    if (notes.length >= MAX_DISAMBIGUATION_NOTES) break;
    if (selectedDocIds.has(candidate.doc.doc_id)) continue;
    if (seenDocIds.has(candidate.doc.doc_id)) continue;

    // Hard-negative flag from guard
    if (candidate.reason && (
      candidate.reason.includes('Hard-negative') ||
      candidate.reason.includes('hard negative') ||
      candidate.reason.includes('known hard negative')
    )) {
      notes.push({
        doc_id: candidate.doc.doc_id,
        title: candidate.doc.title,
        risk_type: 'hard_negative',
        reason: `Tài liệu này là hard-negative: ${candidate.reason}`,
      });
      seenDocIds.add(candidate.doc.doc_id);
      continue;
    }

    // Planned-not-executed in non-selected candidates
    if (candidate.doc.event_status === 'planned_not_executed') {
      notes.push({
        doc_id: candidate.doc.doc_id,
        title: candidate.doc.title,
        risk_type: 'planned_not_executed',
        reason: 'Sự kiện này được lên kế hoạch nhưng chưa/không thực hiện. Không nói như đã xảy ra.',
      });
      seenDocIds.add(candidate.doc.doc_id);
      continue;
    }

    // Same cluster as a primary doc but not selected
    const primaryClusters = [...selectedDocIds]
      .map(id => {
        const d = dataset.events.get(id) ?? dataset.synthesis.get(id);
        return d?.rag_cluster;
      })
      .filter(Boolean);

    if (
      candidate.doc.rag_cluster &&
      primaryClusters.includes(candidate.doc.rag_cluster) &&
      candidate.rerankScore > 0.3
    ) {
      notes.push({
        doc_id: candidate.doc.doc_id,
        title: candidate.doc.title,
        risk_type: 'same_cluster',
        reason: `Cùng nhóm sự kiện (${candidate.doc.rag_cluster}) — có thể bị nhầm lẫn với tài liệu chính.`,
      });
      seenDocIds.add(candidate.doc.doc_id);
    }
  }

  return notes;
}

// ─── Semantic Supporting Doc Guard (Patch 7F) ────────────────

/**
 * Decide whether a supporting candidate should be included in the citation plan.
 *
 * Conservative: only demote when we have clear semantic mismatch signals.
 * - Comparison: always include (both sides needed).
 * - Timeline: always include (chronological coverage needed).
 * - Explanation/cause_effect: always include (context needed).
 * - Lookup/disambiguation: demote if evidenceRoleScore is very low AND
 *   the doc has no topic/title/text overlap with the query frame.
 */
function shouldIncludeSupportingDoc(
  candidate: CandidateDoc,
  queryFrame: QueryFrame | undefined,
  intent: string,
  isDisambigLike: boolean,
  isLookup: boolean
): boolean {
  // Never filter for intents that need broad context
  if (intent === 'comparison' || intent === 'timeline' ||
      intent === 'explanation' || intent === 'cause_effect') {
    return true;
  }

  // If no QueryFrame, keep legacy behavior
  if (!queryFrame) return true;

  // High-confidence supporting docs always pass
  const score = candidate.evidenceRoleScore ?? 0;
  if (score >= 0.25) return true;

  // Very low score: check if there's any semantic overlap with answer_focus
  if (score < 0.1) {
    // For lookup or disambiguation, skip very weak supporting docs
    if (isLookup || isDisambigLike) {
      const focus = queryFrame.answer_focus;

      // Allow if doc title/summary contains the topic
      const topicLower = (focus.topic ?? '').toLowerCase();
      if (topicLower.length > 3) {
        const docText = (candidate.doc.title + ' ' + (candidate.doc.summary ?? '')).toLowerCase();
        if (docText.includes(topicLower)) return true;
      }

      // Allow if treaty_names / campaign_names / movement_names overlap
      const treatyNames = focus.treaty_names ?? [];
      const campaignNames = focus.campaign_names ?? [];
      const movementNames = focus.movement_names ?? [];
      const allNames = [...treatyNames, ...campaignNames, ...movementNames].map(n => n.toLowerCase());
      if (allNames.length > 0) {
        const docText = (candidate.doc.title + ' ' + (candidate.doc.summary ?? '')).toLowerCase();
        if (allNames.some(n => n.length > 3 && docText.includes(n))) return true;
      }

      // No overlap — demote
      return false;
    }
  }

  return true;
}

// ─── Strict Citation Supporting Guard (Patch 7F-1) ───────────

/**
 * Decide whether a supporting doc should enter the CITATION PLAN.
 * Stricter than shouldIncludeSupportingDoc (which gates context inclusion).
 *
 * This gates citation — the final list of records the generator will cite.
 */
function shouldCiteSupportingDoc(
  doc: IndexableDocument,
  candidate: CandidateDoc,
  queryFrame: QueryFrame | undefined,
  intent: QueryIntent,
  isDisambigLike: boolean,
  isLookup: boolean
): boolean {
  // Comparison/timeline/explanation always allow supporting citations
  if (intent === 'comparison') return true;
  if (intent === 'timeline') return true;
  if (intent === 'explanation' || intent === 'cause_effect') return true;

  // Hard-negative-flagged candidates: skip for lookup/disambiguation
  const reason = candidate.reason ?? '';
  if ((isLookup || isDisambigLike) &&
      (reason.includes('hard-negative') || reason.includes('Hard-negative') ||
       reason.includes('penalized') || reason.includes('known hard negative'))) {
    return false;
  }

  // No QueryFrame — conservative: allow if score good
  if (!queryFrame) {
    return (candidate.evidenceRoleScore ?? 0) >= 0.2;
  }

  const score = candidate.evidenceRoleScore ?? 0;
  const focus = queryFrame.answer_focus;

  // Strong score → always cite
  if (score >= 0.35) return true;

  // Medium score → check semantic overlap
  if (score >= 0.2) {
    // Topic overlap
    const topicLower = (focus.topic ?? '').toLowerCase();
    if (topicLower.length > 3) {
      const docText = (doc.title + ' ' + (doc.summary ?? '')).toLowerCase();
      if (docText.includes(topicLower)) return true;
    }
    // Treaty/campaign/movement name overlap
    const allNames = [
      ...(focus.treaty_names ?? []),
      ...(focus.campaign_names ?? []),
      ...(focus.movement_names ?? []),
    ].map(n => n.toLowerCase());
    if (allNames.length > 0) {
      const docText = (doc.title + ' ' + (doc.summary ?? '')).toLowerCase();
      if (allNames.some(n => n.length > 3 && docText.includes(n))) return true;
    }
    // For lookup with actor/location/org overlap
    if (isLookup) {
      const actorNames = (focus.actor ?? []).map(a => a.toLowerCase());
      const locNames = (focus.location ?? []).map(l => l.toLowerCase());
      const orgNames = (focus.organization ?? []).map(o => o.toLowerCase());
      const allEntityNames = [...actorNames, ...locNames, ...orgNames];
      if (allEntityNames.length > 0) {
        const docText = (doc.title + ' ' + (doc.summary ?? '')).toLowerCase();
        if (allEntityNames.some(n => n.length > 3 && docText.includes(n))) return true;
      }
    }
    // Medium score, no name overlap — allow for non-lookup/non-disambiguation
    return !isLookup && !isDisambigLike;
  }

  // Low score (< 0.2): only cite if strong topic/name match
  const topicLower = (focus.topic ?? '').toLowerCase();
  if (topicLower.length > 3) {
    const docText = (doc.title + ' ' + (doc.summary ?? '')).toLowerCase();
    if (docText.includes(topicLower)) return !isLookup; // allow for non-lookup only
  }

  return false;
}

// ─── Role-Aware Citation Plan (Patch 7F-1) ────────────────────

/**
 * Build citation plan with strict per-intent citation caps and role-aware filtering.
 *
 * Citation caps:
 * - lookup (fact/date/actor/location/entity): 1 primary + max 1 strong supporting
 * - disambiguation/misconception: 1 primary + filtered supporting, no contrast
 * - comparison: primary + supporting + contrast (both sides needed)
 * - timeline: primary + up to template.supporting (broad)
 * - explanation/cause_effect: primary + up to 2 supporting
 * - other (multi_hop): primary + filtered supporting
 */
function buildRoleAwareCitationPlan(
  primaryDocs: IndexableDocument[],
  supportingDocs: IndexableDocument[],
  isComparison: boolean,
  contrastDocs: IndexableDocument[],
  _dataset: LoadedDataset,
  intent: QueryIntent,
  queryFrame?: QueryFrame,
  supportingCandidates?: CandidateDoc[],
  isTwoSided?: boolean
): CitationPlanItem[] {
  const plan: CitationPlanItem[] = [];
  let priority = 0;

  const isLookup = ['fact_lookup', 'date_lookup', 'actor_lookup', 'location_lookup', 'entity_profile'].includes(intent);
  const isDisambigLike = ['multi_hop', 'disambiguation', 'misconception_check'].includes(intent);
  const isTimeline = intent === 'timeline';
  const isExplanation = intent === 'explanation' || intent === 'cause_effect';

  // Citation cap for supporting docs by intent
  let maxSupportingCitations: number;
  if (isLookup) {
    maxSupportingCitations = 1;       // at most 1 supporting for lookup
  } else if (isDisambigLike) {
    maxSupportingCitations = 2;       // limited supporting for disambiguation
  } else if (isTimeline) {
    maxSupportingCitations = 6;       // broad for timeline
  } else if (isExplanation) {
    maxSupportingCitations = 2;       // synthesis + 2 events for explanation
  } else if (isComparison) {
    maxSupportingCitations = 4;       // both sides
  } else {
    maxSupportingCitations = 2;
  }

  // Patch 8D: Override cap for two-sided queries
  if (isTwoSided && maxSupportingCitations < 4) {
    maxSupportingCitations = 4;
  }
  for (const doc of primaryDocs) {
    plan.push({
      doc_id: doc.doc_id,
      citation_role: 'primary',
      source_ids: doc.source_ids ?? [],
      reason: 'evidence_role=primary',
      priority: priority++,
    });
  }

  // If only primary needed for lookup and we already have primary, log strict mode
  if (isLookup && primaryDocs.length >= 1 && maxSupportingCitations === 1) {
    // strict lookup: only add supporting if it passes the strict gate
  }

  // Supporting docs — apply strict citation guard
  let supportingCited = 0;
  for (const doc of supportingDocs) {
    if (supportingCited >= maxSupportingCitations) break;

    // Find matching candidate for score info
    const matchingCandidate = supportingCandidates?.find(c => c.doc.doc_id === doc.doc_id);
    const candidateProxy: CandidateDoc = matchingCandidate ?? {
      doc,
      rerankScore: 0,
      evidenceRoleScore: 0,
    };

    // Check if this is a contrast doc for comparison intent
    const isContrast = isComparison && contrastDocs.some(c => c.doc_id === doc.doc_id);

    // For comparison contrast docs: always cite
    if (isContrast) {
      plan.push({
        doc_id: doc.doc_id,
        citation_role: 'contrast',
        source_ids: doc.source_ids ?? [],
        reason: 'evidence_role=contrast (comparison)',
        priority: priority++,
      });
      supportingCited++;
      continue;
    }

    // Patch 8D: Skip strict citation guard for two-sided queries — side docs must be cited
    if (!isTwoSided && !shouldCiteSupportingDoc(doc, candidateProxy, queryFrame, intent, isDisambigLike, isLookup)) {
      continue;
    }

    plan.push({
      doc_id: doc.doc_id,
      citation_role: 'supporting',
      source_ids: doc.source_ids ?? [],
      reason: 'evidence_role=supporting',
      priority: priority++,
    });
    supportingCited++;
  }

  return plan;
}

// ─── Citation Plan (legacy) ───────────────────────────────────

/** Build a citation plan mapping docs to their citation roles (legacy, used by buildLegacyBundle) */
function buildCitationPlan(
  primaryDocs: IndexableDocument[],
  supportingDocs: IndexableDocument[],
  _dataset: LoadedDataset
): CitationPlanItem[] {
  const plan: CitationPlanItem[] = [];

  for (const doc of primaryDocs) {
    plan.push({
      doc_id: doc.doc_id,
      citation_role: 'primary',
      source_ids: doc.source_ids ?? [],
    });
  }

  for (const doc of supportingDocs) {
    plan.push({
      doc_id: doc.doc_id,
      citation_role: 'supporting',
      source_ids: doc.source_ids ?? [],
    });
  }

  return plan;
}

// ─── Context Text Assembly ───────────────────────────────────

/** Format documents into structured context text for the LLM */
function assembleContextText(
  intent: QueryIntent,
  primaryDocs: IndexableDocument[],
  supportingDocs: IndexableDocument[],
  warnings: string[],
  disambiguationNotes: DisambiguationNote[],
  citationPlan: CitationPlanItem[]
): string {
  const sections: string[] = [];

  // Header
  sections.push(`=== CONTEXT BUNDLE (intent: ${intent}) ===\n`);

  // Warnings
  if (warnings.length > 0) {
    sections.push(`⚠️ LƯU Ý:\n${warnings.map(w => `- ${w}`).join('\n')}\n`);
  }

  // Primary documents (richer content)
  sections.push('--- TÀI LIỆU CHÍNH ---');
  for (const doc of primaryDocs) {
    sections.push(formatDocForContext(doc, 'primary'));
  }

  // Supporting documents (compact)
  if (supportingDocs.length > 0) {
    sections.push('\n--- TÀI LIỆU HỖ TRỢ ---');
    for (const doc of supportingDocs) {
      sections.push(formatDocForContext(doc, 'supporting'));
    }
  }

  // Disambiguation notes
  if (disambiguationNotes.length > 0) {
    sections.push('\n=== DISAMBIGUATION NOTES ===');
    sections.push('Không nhầm với:');
    for (const note of disambiguationNotes) {
      sections.push(`- [${note.doc_id}] ${note.title} — ${note.reason}`);
    }
    sections.push('(Không dùng disambiguation notes như bằng chứng chính.)');
  }

  // Citation plan
  if (citationPlan.length > 0) {
    sections.push('\n=== CITATION PLAN ===');
    for (const item of citationPlan) {
      const sourceNote = item.source_ids.length > 0
        ? ` (sources: ${item.source_ids.join(', ')})`
        : '';
      sections.push(`- Cite ${item.doc_id} as ${item.citation_role} evidence.${sourceNote}`);
    }
    sections.push('- Do not cite disambiguation notes as primary evidence.');
  }

  return sections.join('\n');
}

/** Format a single document for inclusion in context */
function formatDocForContext(doc: IndexableDocument, role: 'primary' | 'supporting'): string {
  const status = doc.event_status === 'planned_not_executed'
    ? ' ⚠️ [KẾ HOẠCH CHƯA THỰC HIỆN]'
    : '';
  const verified = doc.verification_status === 'verified' ? '✓' : '○';
  const yearStr = doc.year ? `(${doc.year}${doc.end_year ? '–' + doc.end_year : ''})` : '';

  const lines = [
    `\n[${role === 'primary' ? 'PRIMARY' : 'SUPPORTING'}] [${verified} ${doc.doc_id}] ${doc.title} ${yearStr}${status}`,
    `Loại: ${doc.doc_source}/${doc.doc_type} | Xác minh: ${doc.verification_status} | Vai trò: ${role}`,
  ];

  // Content text based on role
  const contentText = getDocContextText(doc, role);
  if (role === 'primary') {
    if (doc.summary && contentText !== doc.summary) {
      lines.push(`Tóm tắt: ${truncateText(doc.summary, MAX_SUPPORTING_TEXT_CHARS)}`);
    }
    if (contentText) {
      lines.push(`Nội dung liên quan:\n${contentText}`);
    }
  } else {
    if (contentText) {
      lines.push(`Tóm tắt: ${contentText}`);
    }
  }

  if (doc.people_labels.length > 0) {
    lines.push(`Nhân vật: ${doc.people_labels.join(', ')}`);
  }

  if (doc.place_labels.length > 0) {
    lines.push(`Địa điểm: ${doc.place_labels.join(', ')}`);
  }

  // Source IDs for provenance tracing
  if (doc.source_ids && doc.source_ids.length > 0) {
    lines.push(`Source IDs: ${doc.source_ids.join(', ')}`);
  }

  return lines.join('\n');
}
