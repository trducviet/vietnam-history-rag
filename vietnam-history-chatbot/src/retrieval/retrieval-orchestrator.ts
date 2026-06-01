/**
 * Retrieval Orchestrator — routes retrieval to the correct lane(s)
 * based on router target_indexes. Enforces lane separation:
 *   - Event lane: eventBM25 + eventVectorStore
 *   - Synthesis lane: synthesisBM25 + synthesisVectorStore
 *
 * NEVER searches a mixed index. Each lane has its own BM25 and VectorStore.
 */

import type { BM25Index } from './bm25-index.js';
import type { VectorStore } from '../shared/vector-store.js';
import type {
  RoutingResult,
  MetadataFilter,
  HybridSearchResult,
  RuntimeDocSource,
} from '../shared/types.js';
import { hybridSearch, bm25OnlySearch, type HybridSearchOptions } from './hybrid-search.js';
import { config } from '../shared/config.js';
import { rerankResults, expandTeacherQuery, type RerankerDebugInfo } from './retrieval-reranker.js';

// ─── Dual Index Config ───────────────────────────────────────

/** Physical separation of Event and Synthesis indexes */
export interface DualIndexConfig {
  eventBM25: BM25Index;
  synthesisBM25: BM25Index;
  /** BM25-only lane for disambiguation rules; links remain provenance-only. */
  rulesBM25?: BM25Index;
  eventVectorStore: VectorStore | null;
  synthesisVectorStore: VectorStore | null;
}

// ─── Orchestrator ────────────────────────────────────────────

/**
 * Orchestrate retrieval across Event and Synthesis lanes based on
 * the router's target_indexes decision.
 *
 * Rules:
 * 1. Only search lanes specified by target_indexes.
 * 2. Each lane gets its own doc_source filter enforced.
 * 3. canonical_only is always true.
 * 4. Results from all active lanes are merged and sorted by combined_score.
 * 5. If target_indexes is empty/invalid, fallback to both lanes with warning.
 */
export async function orchestrateRetrieval(
  query: string,
  routing: RoutingResult,
  indexes: DualIndexConfig,
  topK: number = config.defaultTopK,
  options?: HybridSearchOptions
): Promise<HybridSearchResult[]> {
  // Stage 7E1: teacher-style query expansion (no API)
  const expandedQuery = expandTeacherQuery(query);
  if (expandedQuery !== query) {
    console.log(`   🎓 Teacher-style expansion: "${query}" → added terms`);
  }
  const searchQuery = expandedQuery;
  // Validate target_indexes, fallback to both if empty
  const targetIndexes: RuntimeDocSource[] =
    routing.target_indexes?.length > 0
      ? routing.target_indexes
      : ['event', 'synthesis'];

  if (!routing.target_indexes?.length) {
    console.warn('⚠️  Empty target_indexes from router, falling back to both lanes');
  }
  // Stage 7E1C: Always search rules lane when available.
  // Rules have very few docs (low cost) and can surface relevant context
  // for queries like Q014 ("giặc đói/dốt") that need rule docs but don't
  // have explicit comparison/disambiguation intent.
  const shouldSearchRules = !!indexes.rulesBM25;

  const allResults: HybridSearchResult[] = [];

  // Determine per-lane topK: fetch more candidates per lane for better merge
  const laneTopK = topK;

  // ── Event Lane ───────────────────────────────────────────
  if (targetIndexes.includes('event')) {
    const eventFilter: MetadataFilter = {
      ...routing.metadata_filters,
      canonical_only: true,
      doc_source: 'event',
    };

    const eventResults = await searchLane(
      searchQuery,
      indexes.eventBM25,
      indexes.eventVectorStore,
      laneTopK,
      eventFilter,
      options
    );
    allResults.push(...eventResults);

    const eventMode = indexes.eventVectorStore ? 'hybrid' : 'BM25-only';
    console.log(`   📗 Event lane (${eventMode}): ${eventResults.length} candidates`);
  }

  // ── Synthesis Lane ───────────────────────────────────────
  if (targetIndexes.includes('synthesis')) {
    const synthesisFilter: MetadataFilter = {
      ...routing.metadata_filters,
      canonical_only: true,
      doc_source: 'synthesis',
    };

    const synthesisResults = await searchLane(
      searchQuery,
      indexes.synthesisBM25,
      indexes.synthesisVectorStore,
      laneTopK,
      synthesisFilter,
      options
    );
    allResults.push(...synthesisResults);

    const synthMode = indexes.synthesisVectorStore ? 'hybrid' : 'BM25-only';
    console.log(`   📘 Synthesis lane (${synthMode}): ${synthesisResults.length} candidates`);
  }

  // ── Disambiguation Rule Lane ─────────────────────────────
  if (shouldSearchRules && indexes.rulesBM25) {
    const rulesFilter: MetadataFilter = {
      ...routing.metadata_filters,
      canonical_only: true,
      doc_source: 'disambiguation_rule',
    };

    const ruleResults = await searchLane(
      searchQuery,
      indexes.rulesBM25,
      null,
      laneTopK,
      rulesFilter,
      options
    );
    allResults.push(...ruleResults);
    console.log(`   📙 Rules lane (BM25-only): ${ruleResults.length} candidates`);
  }

  // ── Merge & Sort ─────────────────────────────────────────
  allResults.sort((a, b) => b.combined_score - a.combined_score);

  // Stage 7E1: post-retrieval reranking (no API)
  const { results: reranked, debug: rerankerDebug } = rerankResults(
    query,
    allResults.slice(0, topK * 2), // rerank top 2x candidates
    routing,
    topK
  );

  // Stage 7E1C: if reranker made no adjustments, use original sort to avoid
  // candidate-set alteration from the 2x slicing (preserves Q014-type queries)
  const finalResults = rerankerDebug.total_score_adjustments > 0
    ? reranked
    : allResults.slice(0, topK);

  if (rerankerDebug.total_score_adjustments > 0) {
    console.log(`   🔄 Reranker: ${rerankerDebug.total_score_adjustments} adjustments` +
      (rerankerDebug.teacher_style_detected ? ' [teacher]' : '') +
      (rerankerDebug.comparison_intent_detected ? ' [comparison]' : '') +
      (rerankerDebug.cross_period_pair_detected ? ` [pair:${rerankerDebug.cross_period_pair_detected}]` : '') +
      (rerankerDebug.rule_boost_applied > 0 ? ` [rules+${rerankerDebug.rule_boost_applied}]` : '')
    );
  }

  console.log(`   📊 Merged: ${allResults.length} total → final top ${finalResults.length}`);
  return finalResults;
}

// ─── Lane Search ─────────────────────────────────────────────

/**
 * Search a single lane (event or synthesis) using hybrid or BM25-only.
 */
async function searchLane(
  query: string,
  bm25Index: BM25Index,
  vectorStore: VectorStore | null,
  topK: number,
  filter: MetadataFilter,
  options?: HybridSearchOptions
): Promise<HybridSearchResult[]> {
  if (vectorStore) {
    return hybridSearch(query, vectorStore, bm25Index, topK, filter, options);
  }
  return bm25OnlySearch(query, bm25Index, topK, filter);
}
