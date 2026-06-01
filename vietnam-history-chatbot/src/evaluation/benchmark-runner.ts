/**
 * QA Benchmark Runner — replays all questions from qa_benchmark.jsonl
 * through the full RAG pipeline and measures end-to-end quality.
 *
 * PATCH 1: Uses dual BM25 indexes + orchestrateRetrieval.
 */

import type {
  LoadedDataset,
  QABenchmarkEntry,
  ChatResponse,
} from '../shared/types.js';
import type { VectorStore } from '../shared/vector-store.js';
import type { BM25Index } from '../retrieval/bm25-index.js';
import { routeQuery } from '../routing/query-router.js';
import { orchestrateRetrieval, type DualIndexConfig } from '../retrieval/retrieval-orchestrator.js';
import { rerankResults } from '../reranking/reranker.js';
import { applyHardNegativeGuard } from '../reranking/hard-negative-guard.js';
import { buildContextBundle } from '../context/context-builder.js';
import { generateAnswer } from '../generation/answer-generator.js';

// ─── Benchmark Types ─────────────────────────────────────────

export interface BenchmarkResult {
  query_id: string;
  question: string;
  question_type: string;
  difficulty: string;
  gold_answer: string;
  system_answer: string;
  citations_match: boolean;
  expected_ids: string[];
  retrieved_ids: string[];
  recall_at_5: boolean;
  confidence: string;
  response: ChatResponse;
}

export interface BenchmarkSummary {
  total: number;
  retrieval_recall_at_5: number;
  citation_match_rate: number;
  by_difficulty: Record<string, { total: number; recall5: number }>;
  by_type: Record<string, { total: number; recall5: number }>;
}

// ─── Benchmark Runner ────────────────────────────────────────

/**
 * Run the full QA benchmark through the complete RAG pipeline.
 *
 * PATCH 1: Accepts dual BM25 indexes and optional dual vector stores.
 */
export async function runQABenchmark(
  dataset: LoadedDataset,
  eventBM25: BM25Index,
  synthesisBM25: BM25Index,
  eventVectorStore: VectorStore | null,
  synthesisVectorStore: VectorStore | null,
  mode: 'full' | 'bm25' = 'bm25'
): Promise<{ summary: BenchmarkSummary; results: BenchmarkResult[] }> {
  const entries = dataset.qaBenchmark;
  const results: BenchmarkResult[] = [];

  const indexes: DualIndexConfig = {
    eventBM25,
    synthesisBM25,
    eventVectorStore,
    synthesisVectorStore,
  };

  console.log(`\n🧪 Running QA benchmark (${mode} mode, dual-index) on ${entries.length} questions...\n`);

  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    console.log(`  [${i + 1}/${entries.length}] ${entry.question.substring(0, 70)}...`);

    const result = await runSingleBenchmark(entry, dataset, indexes);
    results.push(result);

    const status = result.recall_at_5 ? '✅' : '❌';
    console.log(`    ${status} Recall@5=${result.recall_at_5} Confidence=${result.confidence}`);
  }

  const summary = buildSummary(results);
  printSummary(summary);

  return { summary, results };
}

/** Run a single benchmark question through the full pipeline */
async function runSingleBenchmark(
  entry: QABenchmarkEntry,
  dataset: LoadedDataset,
  indexes: DualIndexConfig
): Promise<BenchmarkResult> {
  // Step 1: Route query
  const routing = await routeQuery(entry.question);

  // Step 2: Retrieve via dual-lane orchestrator
  const searchResults = await orchestrateRetrieval(entry.question, routing, indexes, 10);

  // Step 3: Rerank
  const reranked = await rerankResults(entry.question, searchResults, dataset);

  // Step 3.5: Hard-negative guard
  const guarded = applyHardNegativeGuard(entry.question, reranked, dataset);

  // Step 4: Build context bundle
  const contextBundle = buildContextBundle(routing.intent, guarded, dataset);

  // Step 5: Generate answer
  const response = await generateAnswer(entry.question, contextBundle, searchResults, dataset);

  // Step 6: Evaluate
  const targetIds = new Set([
    ...entry.expected_record_ids,
    ...entry.acceptable_record_ids,
  ]);
  const retrievedIds = searchResults.map(r => r.doc_id).slice(0, 5);
  const recall_at_5 = retrievedIds.some(id => targetIds.has(id));

  const citationIds = new Set(response.citations.map(c => c.record_id));
  const citations_match = entry.expected_record_ids.some(id => citationIds.has(id));

  return {
    query_id: entry.query_id,
    question: entry.question,
    question_type: entry.question_type,
    difficulty: entry.difficulty,
    gold_answer: entry.gold_answer,
    system_answer: response.answer,
    citations_match,
    expected_ids: [...targetIds],
    retrieved_ids: retrievedIds,
    recall_at_5,
    confidence: response.confidence,
    response,
  };
}

// ─── Summary ─────────────────────────────────────────────────

function buildSummary(results: BenchmarkResult[]): BenchmarkSummary {
  const total = results.length;
  const retrieval_recall_at_5 = results.filter(r => r.recall_at_5).length / total;
  const citation_match_rate = results.filter(r => r.citations_match).length / total;

  const by_difficulty: Record<string, { total: number; recall5: number }> = {};
  const by_type: Record<string, { total: number; recall5: number }> = {};

  for (const r of results) {
    if (!by_difficulty[r.difficulty]) by_difficulty[r.difficulty] = { total: 0, recall5: 0 };
    by_difficulty[r.difficulty].total++;
    if (r.recall_at_5) by_difficulty[r.difficulty].recall5++;

    if (!by_type[r.question_type]) by_type[r.question_type] = { total: 0, recall5: 0 };
    by_type[r.question_type].total++;
    if (r.recall_at_5) by_type[r.question_type].recall5++;
  }

  return { total, retrieval_recall_at_5, citation_match_rate, by_difficulty, by_type };
}

function printSummary(summary: BenchmarkSummary): void {
  console.log('\n═══════════════════════════════════════════');
  console.log('  QA Benchmark Results (Dual-Index)');
  console.log('═══════════════════════════════════════════');
  console.log(`  Total questions:    ${summary.total}`);
  console.log(`  Retrieval R@5:      ${(summary.retrieval_recall_at_5 * 100).toFixed(1)}%`);
  console.log(`  Citation match:     ${(summary.citation_match_rate * 100).toFixed(1)}%`);

  console.log('\n  By difficulty:');
  for (const [diff, stats] of Object.entries(summary.by_difficulty)) {
    const pct = ((stats.recall5 / stats.total) * 100).toFixed(1);
    console.log(`    ${diff}: ${pct}% (${stats.recall5}/${stats.total})`);
  }

  console.log('\n  By type:');
  for (const [type, stats] of Object.entries(summary.by_type)) {
    const pct = ((stats.recall5 / stats.total) * 100).toFixed(1);
    console.log(`    ${type}: ${pct}% (${stats.recall5}/${stats.total})`);
  }
  console.log('═══════════════════════════════════════════\n');
}
