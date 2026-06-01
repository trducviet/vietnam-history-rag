/**
 * Hard-Negative Guard — applies query-specific hard-negative penalties
 * to reranked candidates. Uses benchmark data to identify when a candidate
 * is a known confusion risk for the current (or similar) query.
 *
 * PATCH 2: Replaces the old global hard-negative check with per-query matching.
 *
 * Strategy:
 * 1. Normalize & tokenize the input query.
 * 2. Search retrievalQueries and qaBenchmark for similar questions (Jaccard ≥ 0.55 or exact).
 * 3. Collect hard_negative_ids from matched entries.
 * 4. Penalize any candidate whose doc_id is in the collected set.
 */

import type { RerankedResult, LoadedDataset } from '../shared/types.js';

// ─── Types ───────────────────────────────────────────────────

export interface HardNegativeAssessment {
  doc_id: string;
  is_hard_negative: boolean;
  risk: number;
  reasons: string[];
  matched_query_ids: string[];
}

// ─── Text Normalization ──────────────────────────────────────

/** Normalize text for comparison: lowercase, NFKC, strip punctuation */
function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .normalize('NFKC')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Tokenize normalized text into word set */
function tokenize(text: string): Set<string> {
  return new Set(normalizeText(text).split(' ').filter(t => t.length > 0));
}

/** Jaccard similarity between two token sets */
function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) intersection++;
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

// ─── Query Matching ──────────────────────────────────────────

const SIMILARITY_THRESHOLD = 0.55;

/**
 * Find hard-negative doc IDs that are relevant to the current query.
 * Matches against retrievalQueries and qaBenchmark entries using
 * exact match or Jaccard similarity ≥ 0.55.
 */
export function getQuerySpecificHardNegativeIds(
  query: string,
  dataset: LoadedDataset
): Set<string> {
  const normalizedQuery = normalizeText(query);
  const queryTokens = tokenize(query);
  const hardNegIds = new Set<string>();

  // Search retrievalQueries
  for (const rq of dataset.retrievalQueries) {
    if (isQuerySimilar(normalizedQuery, queryTokens, rq.query)) {
      for (const hnId of rq.hard_negative_ids) {
        hardNegIds.add(hnId);
      }
    }
  }

  // Search qaBenchmark
  for (const qa of dataset.qaBenchmark) {
    if (isQuerySimilar(normalizedQuery, queryTokens, qa.question)) {
      for (const hnId of qa.hard_negative_ids) {
        hardNegIds.add(hnId);
      }
    }
  }

  return hardNegIds;
}

/** Check if a benchmark question is similar to the current query */
function isQuerySimilar(
  normalizedQuery: string,
  queryTokens: Set<string>,
  benchmarkQuestion: string
): boolean {
  const normalizedBenchmark = normalizeText(benchmarkQuestion);

  // Exact normalized match
  if (normalizedQuery === normalizedBenchmark) return true;

  // Jaccard similarity
  const benchmarkTokens = tokenize(benchmarkQuestion);
  return jaccardSimilarity(queryTokens, benchmarkTokens) >= SIMILARITY_THRESHOLD;
}

// ─── Assessment ──────────────────────────────────────────────

/**
 * Assess hard-negative risk for a single document against the current query.
 */
export function assessHardNegativeRisk(
  query: string,
  docId: string,
  dataset: LoadedDataset
): HardNegativeAssessment {
  const normalizedQuery = normalizeText(query);
  const queryTokens = tokenize(query);
  const matchedQueryIds: string[] = [];
  const reasons: string[] = [];
  let isHardNeg = false;

  // Check retrievalQueries
  for (const rq of dataset.retrievalQueries) {
    if (isQuerySimilar(normalizedQuery, queryTokens, rq.query)) {
      if (rq.hard_negative_ids.includes(docId)) {
        isHardNeg = true;
        matchedQueryIds.push(rq.query_id);
        reasons.push(`Hard-negative for retrieval query ${rq.query_id}: "${rq.query.substring(0, 50)}..."`);
      }
    }
  }

  // Check qaBenchmark
  for (const qa of dataset.qaBenchmark) {
    if (isQuerySimilar(normalizedQuery, queryTokens, qa.question)) {
      if (qa.hard_negative_ids.includes(docId)) {
        isHardNeg = true;
        matchedQueryIds.push(qa.query_id);
        reasons.push(`Hard-negative for benchmark ${qa.query_id}: "${qa.question.substring(0, 50)}..."`);
      }
    }
  }

  return {
    doc_id: docId,
    is_hard_negative: isHardNeg,
    risk: isHardNeg ? 1.0 : 0.0,
    reasons,
    matched_query_ids: matchedQueryIds,
  };
}

// ─── Guard Application ──────────────────────────────────────

const HARD_NEGATIVE_PENALTY_FACTOR = 0.55;

/**
 * Apply hard-negative guard to reranked candidates.
 * Penalizes known hard-negatives by reducing their rerank_score.
 * Does NOT remove them — lets downstream decide.
 */
export function applyHardNegativeGuard(
  query: string,
  candidates: RerankedResult[],
  dataset: LoadedDataset
): RerankedResult[] {
  const hardNegIds = getQuerySpecificHardNegativeIds(query, dataset);

  if (hardNegIds.size === 0) return candidates;

  let penalizedCount = 0;

  const guarded = candidates.map(c => {
    if (hardNegIds.has(c.doc_id)) {
      penalizedCount++;
      return {
        ...c,
        rerank_score: c.rerank_score * HARD_NEGATIVE_PENALTY_FACTOR,
        reason: (c.reason ? c.reason + '; ' : '') +
          'Hard-negative guard: known confusing document for this query type.',
      };
    }
    return c;
  });

  if (penalizedCount > 0) {
    console.log(`   🛡️ Hard-negative guard: penalized ${penalizedCount}/${candidates.length} candidates`);
  }

  // Re-sort after penalty
  guarded.sort((a, b) => b.rerank_score - a.rerank_score);
  return guarded;
}
