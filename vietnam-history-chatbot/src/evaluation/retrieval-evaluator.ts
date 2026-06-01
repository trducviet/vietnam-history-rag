/**
 * Retrieval Evaluator — measures retrieval quality using:
 *   - Recall@1, Recall@3, Recall@5
 *   - Mean Reciprocal Rank (MRR)
 *   - Hard-negative confusion rate
 *
 * Uses retrieval_queries.jsonl and hard_negatives.jsonl for evaluation.
 */

import type {
  LoadedDataset,
  RetrievalQuery,
  HybridSearchResult,
  MetadataFilter,
} from '../shared/types.js';
import type { VectorStore } from '../shared/vector-store.js';
import type { BM25Index } from '../retrieval/bm25-index.js';
import { hybridSearch, bm25OnlySearch } from '../retrieval/hybrid-search.js';

// ─── Metric Types ────────────────────────────────────────────

export interface RetrievalMetrics {
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  mrr: number;
  hard_negative_confusion_rate: number;
  total_queries: number;
}

export interface PerQueryResult {
  query_id: string;
  query: string;
  query_type: string;
  difficulty: string;
  target_ids: string[];
  retrieved_ids: string[];
  recall_at_1: boolean;
  recall_at_3: boolean;
  recall_at_5: boolean;
  reciprocal_rank: number;
  hard_negative_hits: string[];
}

// ─── Evaluation Runner ──────────────────────────────────────

/**
 * Run retrieval evaluation on all queries from retrieval_queries.jsonl.
 *
 * @param mode - 'hybrid' requires vectorStore, 'bm25' uses BM25 only
 */
export async function evaluateRetrieval(
  dataset: LoadedDataset,
  bm25Index: BM25Index,
  vectorStore: VectorStore | null,
  mode: 'hybrid' | 'bm25' = 'bm25'
): Promise<{ metrics: RetrievalMetrics; perQuery: PerQueryResult[] }> {
  const queries = dataset.retrievalQueries;
  const perQuery: PerQueryResult[] = [];

  console.log(`\n📊 Running retrieval evaluation (${mode} mode) on ${queries.length} queries...\n`);

  for (const q of queries) {
    const result = await evaluateSingleQuery(q, dataset, bm25Index, vectorStore, mode);
    perQuery.push(result);

    const status = result.recall_at_5 ? '✅' : '❌';
    console.log(`  ${status} [${q.query_id}] ${q.query.substring(0, 60)}... RR=${result.reciprocal_rank.toFixed(2)}`);
  }

  // Aggregate metrics
  const metrics = aggregateMetrics(perQuery, dataset);

  console.log('\n═══════════════════════════════════════════');
  console.log('  Retrieval Evaluation Results');
  console.log('═══════════════════════════════════════════');
  console.log(`  Recall@1:   ${(metrics.recall_at_1 * 100).toFixed(1)}%`);
  console.log(`  Recall@3:   ${(metrics.recall_at_3 * 100).toFixed(1)}%`);
  console.log(`  Recall@5:   ${(metrics.recall_at_5 * 100).toFixed(1)}%`);
  console.log(`  MRR:        ${metrics.mrr.toFixed(3)}`);
  console.log(`  Hard-neg confusion: ${(metrics.hard_negative_confusion_rate * 100).toFixed(1)}%`);
  console.log(`  Total queries: ${metrics.total_queries}`);
  console.log('═══════════════════════════════════════════\n');

  return { metrics, perQuery };
}

/** Evaluate a single retrieval query */
async function evaluateSingleQuery(
  query: RetrievalQuery,
  dataset: LoadedDataset,
  bm25Index: BM25Index,
  vectorStore: VectorStore | null,
  mode: 'hybrid' | 'bm25'
): Promise<PerQueryResult> {
  // Retrieve top-5
  let results: HybridSearchResult[];

  if (mode === 'hybrid' && vectorStore) {
    results = await hybridSearch(query.query, vectorStore, bm25Index, 5);
  } else {
    results = bm25OnlySearch(query.query, bm25Index, 5);
  }

  const retrievedIds = results.map(r => r.doc_id);

  // Target = expected + acceptable
  const targetIds = new Set([
    ...query.target_record_ids,
    ...query.acceptable_record_ids,
  ]);

  // Recall@k
  const recall_at_1 = retrievedIds.slice(0, 1).some(id => targetIds.has(id));
  const recall_at_3 = retrievedIds.slice(0, 3).some(id => targetIds.has(id));
  const recall_at_5 = retrievedIds.slice(0, 5).some(id => targetIds.has(id));

  // Reciprocal rank
  let reciprocal_rank = 0;
  for (let i = 0; i < retrievedIds.length; i++) {
    if (targetIds.has(retrievedIds[i])) {
      reciprocal_rank = 1 / (i + 1);
      break;
    }
  }

  // Hard-negative hits in top-3
  const hardNegIds = new Set(query.hard_negative_ids);
  const hard_negative_hits = retrievedIds
    .slice(0, 3)
    .filter(id => hardNegIds.has(id));

  return {
    query_id: query.query_id,
    query: query.query,
    query_type: query.query_type,
    difficulty: query.difficulty,
    target_ids: [...targetIds],
    retrieved_ids: retrievedIds,
    recall_at_1,
    recall_at_3,
    recall_at_5,
    reciprocal_rank,
    hard_negative_hits,
  };
}

/** Aggregate per-query results into overall metrics */
function aggregateMetrics(
  perQuery: PerQueryResult[],
  dataset: LoadedDataset
): RetrievalMetrics {
  const n = perQuery.length;
  if (n === 0) {
    return {
      recall_at_1: 0, recall_at_3: 0, recall_at_5: 0,
      mrr: 0, hard_negative_confusion_rate: 0, total_queries: 0,
    };
  }

  const recall_at_1 = perQuery.filter(r => r.recall_at_1).length / n;
  const recall_at_3 = perQuery.filter(r => r.recall_at_3).length / n;
  const recall_at_5 = perQuery.filter(r => r.recall_at_5).length / n;
  const mrr = perQuery.reduce((sum, r) => sum + r.reciprocal_rank, 0) / n;

  // Hard-negative confusion: fraction of queries where a hard-neg appeared in top-3
  const confusedQueries = perQuery.filter(r => r.hard_negative_hits.length > 0).length;
  const hard_negative_confusion_rate = confusedQueries / n;

  return {
    recall_at_1,
    recall_at_3,
    recall_at_5,
    mrr,
    hard_negative_confusion_rate,
    total_queries: n,
  };
}

// ─── Diagnostic Helpers ──────────────────────────────────────

/** Print detailed failure analysis for debugging */
export function printFailureAnalysis(
  perQuery: PerQueryResult[],
  dataset: LoadedDataset
): void {
  const failures = perQuery.filter(r => !r.recall_at_5);

  if (failures.length === 0) {
    console.log('🎉 No failures at Recall@5!');
    return;
  }

  console.log(`\n❌ ${failures.length} queries failed at Recall@5:\n`);

  for (const f of failures) {
    console.log(`  [${f.query_id}] (${f.difficulty}) ${f.query}`);
    console.log(`    Expected: ${f.target_ids.join(', ')}`);
    console.log(`    Got:      ${f.retrieved_ids.join(', ')}`);

    if (f.hard_negative_hits.length > 0) {
      console.log(`    ⚠️ Hard-neg hits: ${f.hard_negative_hits.join(', ')}`);
    }
    console.log();
  }
}

/** Print breakdown by query type and difficulty */
export function printBreakdown(perQuery: PerQueryResult[]): void {
  // By difficulty
  const byDifficulty = new Map<string, PerQueryResult[]>();
  for (const r of perQuery) {
    const arr = byDifficulty.get(r.difficulty) || [];
    arr.push(r);
    byDifficulty.set(r.difficulty, arr);
  }

  console.log('\n📊 Breakdown by difficulty:');
  for (const [diff, results] of byDifficulty) {
    const r5 = results.filter(r => r.recall_at_5).length / results.length;
    console.log(`  ${diff}: Recall@5 = ${(r5 * 100).toFixed(1)}% (${results.length} queries)`);
  }

  // By query type
  const byType = new Map<string, PerQueryResult[]>();
  for (const r of perQuery) {
    const arr = byType.get(r.query_type) || [];
    arr.push(r);
    byType.set(r.query_type, arr);
  }

  console.log('\n📊 Breakdown by query type:');
  for (const [type, results] of byType) {
    const r5 = results.filter(r => r.recall_at_5).length / results.length;
    console.log(`  ${type}: Recall@5 = ${(r5 * 100).toFixed(1)}% (${results.length} queries)`);
  }
}
