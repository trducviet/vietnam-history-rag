/**
 * Confidence Scorer — computes confidence level from 4 signals:
 *   1. retrievalScoreGap — gap between top-1 and top-2 (clear winner vs ambiguous)
 *   2. verifiedRatio — fraction of verified docs in context
 *   3. ambiguityScore — score variance among top candidates
 *   4. hardNegativeRisk — query-specific hard-negative overlap
 *
 * PATCH 2: Replaced dead canonicalRatio (always 1.0) with retrievalScoreGap.
 *          hardNegativeRisk is now query-specific, not global.
 *
 * Formula: overall = 0.30×scoreGap + 0.30×verified + 0.20×(1-ambiguity) + 0.20×(1-hardNeg)
 *   overall ≥ 0.75 → high | overall ≥ 0.45 → medium | else → low
 */

import type {
  ConfidenceSignals,
  ConfidenceLevel,
  ContextBundle,
  HybridSearchResult,
  LoadedDataset,
} from '../shared/types.js';
import { getQuerySpecificHardNegativeIds } from '../reranking/hard-negative-guard.js';

// ─── Signal Computation ─────────────────────────────────────

/**
 * Compute the gap between top-1 and top-2 retrieval scores.
 * A large gap means the top result is clearly differentiated.
 * Returns 0 (no gap / single result) to 1 (clear winner).
 */
function computeRetrievalScoreGap(retrievalResults: HybridSearchResult[]): number {
  if (retrievalResults.length === 0) return 0;
  if (retrievalResults.length === 1) return 1;

  const top1 = retrievalResults[0].combined_score;
  const top2 = retrievalResults[1].combined_score;
  const maxScore = Math.max(top1, 1e-6);

  return Math.max(0, Math.min(1, (top1 - top2) / maxScore));
}

/** Compute the ratio of verified docs in the context bundle */
function computeVerifiedRatio(bundle: ContextBundle): number {
  const allDocs = [...bundle.primary_docs, ...bundle.supporting_docs];
  if (allDocs.length === 0) return 0;
  const verifiedCount = allDocs.filter(d => d.verification_status === 'verified').length;
  return verifiedCount / allDocs.length;
}

/**
 * Compute ambiguity score based on how close the top retrieval scores are.
 * If scores are tightly clustered → high ambiguity (hard to distinguish).
 * If there's a clear winner → low ambiguity.
 *
 * Uses coefficient of variation of top-5 scores (inverted).
 * Returns 0 (clear winner) to 1 (very ambiguous).
 */
function computeAmbiguityScore(retrievalResults: HybridSearchResult[]): number {
  const topScores = retrievalResults.slice(0, 5).map(r => r.combined_score);
  if (topScores.length < 2) return 0;

  const mean = topScores.reduce((a, b) => a + b, 0) / topScores.length;
  if (mean === 0) return 1;

  const variance = topScores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / topScores.length;
  const stdDev = Math.sqrt(variance);
  const cv = stdDev / mean;

  // Low CV = scores are similar = ambiguous → return high value
  // High CV = clear differentiation → return low value
  return Math.max(0, Math.min(1, 1 - cv * 2));
}

/**
 * Compute query-specific hard-negative risk: what fraction of the context
 * bundle's doc_ids appear in the hard-negatives for similar benchmark queries.
 * Returns 0 (no risk) to 1 (high risk).
 *
 * PATCH 2: Uses getQuerySpecificHardNegativeIds instead of global scan.
 */
function computeHardNegativeRisk(
  bundle: ContextBundle,
  query: string,
  dataset: LoadedDataset
): number {
  const bundleIds = new Set(bundle.included_doc_ids);
  if (bundleIds.size === 0) return 0;

  const queryHardNegIds = getQuerySpecificHardNegativeIds(query, dataset);
  if (queryHardNegIds.size === 0) return 0;

  let hitCount = 0;
  for (const docId of bundleIds) {
    if (queryHardNegIds.has(docId)) hitCount++;
  }

  return hitCount / bundleIds.size;
}

// ─── Main Scorer ─────────────────────────────────────────────

/**
 * Compute all confidence signals and the final confidence level.
 */
export function computeConfidence(
  bundle: ContextBundle,
  retrievalResults: HybridSearchResult[],
  query: string,
  dataset: LoadedDataset
): { level: ConfidenceLevel; signals: ConfidenceSignals } {
  const retrievalScoreGap = computeRetrievalScoreGap(retrievalResults);
  const verifiedRatio = computeVerifiedRatio(bundle);
  const ambiguityScore = computeAmbiguityScore(retrievalResults);
  const hardNegativeRisk = computeHardNegativeRisk(bundle, query, dataset);

  const signals: ConfidenceSignals = {
    retrievalScoreGap,
    verifiedRatio,
    ambiguityScore,
    hardNegativeRisk,
    // deprecated backward compat
    canonicalRatio: 1.0,
  };

  const overall =
    retrievalScoreGap * 0.30 +
    verifiedRatio * 0.30 +
    (1 - ambiguityScore) * 0.20 +
    (1 - hardNegativeRisk) * 0.20;

  let level: ConfidenceLevel;
  if (overall >= 0.75) level = 'high';
  else if (overall >= 0.45) level = 'medium';
  else level = 'low';

  return { level, signals };
}

// ─── Patch 9F: Answer-Quality-Aware Confidence Clamping ──────

/**
 * Clamp confidence level based on answer content quality signals.
 * The base confidence from computeConfidence only uses retrieval signals.
 * This function downgrades confidence when the answer/citation quality
 * is weaker than retrieval signals suggest.
 *
 * Rules:
 *  1. HONEST_PARTIAL phrases → max 'medium'
 *  2. Missing evidence / insufficient phrases → max 'low'
 *  3. No citations for in-scope factual answer → max 'medium'
 *  4. "kết luận một phần" → max 'medium'
 *  5. OOS refusal → 'low' (correct behavior, but low factual confidence)
 */
export function clampConfidenceByAnswerQuality(
  baseLevel: ConfidenceLevel,
  answer: string,
  citationCount: number,
): ConfidenceLevel {
  const an = answer.toLowerCase().normalize('NFKC');

  // Rule 5: OOS refusal → always low
  if (an.includes('nằm ngoài phạm vi') || an.includes('ngoài phạm vi dữ liệu')) {
    return 'low';
  }

  // Rule 5b: Ambiguous/clarification → always low
  if (an.includes('thiếu ngữ cảnh') || an.includes('nêu rõ tên sự kiện') ||
      an.includes('chưa đủ thông tin để xác định')) {
    return 'low';
  }

  // Rule 2: Missing evidence / insufficient phrases → max 'low'
  const insufficientPhrases = [
    'chưa có tài liệu',
    'không tìm thấy tài liệu phù hợp',
    'chưa tìm thấy đủ bằng chứng',
    'thiếu thông tin',
    'chưa nêu rõ người',
  ];
  if (insufficientPhrases.some(p => an.includes(p))) {
    return 'low';
  }

  // Rule 1: Honest partial phrases → max 'medium'
  const honestPartialPhrases = [
    'chưa có đủ bằng chứng',
    'chưa đủ bằng chứng trực tiếp',
    'thiếu bằng chứng',
    'kết luận một phần',
    'chưa thể so sánh đầy đủ',
    'chưa có đủ thông tin trực tiếp',
    'chỉ đủ bằng chứng cho một vế',
  ];
  if (honestPartialPhrases.some(p => an.includes(p))) {
    if (baseLevel === 'high') return 'medium';
    return baseLevel;
  }

  // Rule 3: No citations for in-scope factual answer → max 'medium'
  if (citationCount === 0 && !an.includes('nằm ngoài phạm vi') && !an.includes('thiếu ngữ cảnh')) {
    if (baseLevel === 'high') return 'medium';
  }

  return baseLevel;
}
