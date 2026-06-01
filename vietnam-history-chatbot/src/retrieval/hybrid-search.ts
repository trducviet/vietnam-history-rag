/**
 * Hybrid Search — combines BM25 lexical scores with vector similarity scores
 * using weighted fusion. Enforces canonical-only retrieval.
 *
 * Formula: combined = α × norm(vector_score) + (1−α) × norm(bm25_score)
 * where α = config.vectorWeight (default 0.6)
 */

import { config } from '../shared/config.js';
import type { VectorStore } from '../shared/vector-store.js';
import type { BM25Index } from './bm25-index.js';
import type { MetadataFilter, HybridSearchResult, DocMetadata } from '../shared/types.js';
import OpenAI from 'openai';
import { isCloudEmbeddingDisabled, NoCloudViolationError } from '../runtime/no-cloud-guard.js';

/** Options for hybrid search */
export interface HybridSearchOptions {
  /** Override vector weight (α). Default: config.vectorWeight */
  vectorWeight?: number;
}

// ─── Query Embedding ─────────────────────────────────────────

let openaiClient: OpenAI | null = null;

function getOpenAI(): OpenAI {
  if (isCloudEmbeddingDisabled()) {
    throw new NoCloudViolationError('hybrid vector search embedding', 'cloud embedding disabled');
  }
  if (!openaiClient) {
    if (!config.openaiApiKey) {
      throw new Error('OPENAI_API_KEY required for vector search');
    }
    openaiClient = new OpenAI({ apiKey: config.openaiApiKey });
  }
  return openaiClient;
}

/** Embed a query string for vector search */
async function embedQuery(query: string): Promise<number[]> {
  const openai = getOpenAI();
  const response = await openai.embeddings.create({
    model: config.embeddingModel,
    input: query,
  });
  return response.data[0].embedding;
}

// ─── Score Normalization ─────────────────────────────────────

/** Min-max normalize scores to [0, 1] */
function normalizeScores(scores: Map<string, number>): Map<string, number> {
  if (scores.size === 0) return scores;

  let min = Infinity;
  let max = -Infinity;
  for (const s of scores.values()) {
    if (s < min) min = s;
    if (s > max) max = s;
  }

  const range = max - min;
  const normalized = new Map<string, number>();
  for (const [id, s] of scores) {
    normalized.set(id, range === 0 ? 1 : (s - min) / range);
  }
  return normalized;
}

// ─── Hybrid Search ───────────────────────────────────────────

/**
 * Perform hybrid search combining vector similarity and BM25 lexical matching.
 *
 * CANONICAL ENFORCEMENT: filters are always set to canonical_only=true.
 * If retrieval returns any non-canonical document, it is excluded.
 */
export async function hybridSearch(
  query: string,
  vectorStore: VectorStore,
  bm25Index: BM25Index,
  topK: number = config.defaultTopK,
  filter?: MetadataFilter,
  options?: HybridSearchOptions
): Promise<HybridSearchResult[]> {
  // Force canonical-only
  const safeFilter: MetadataFilter = {
    ...filter,
    canonical_only: true,
  };

  // Run BM25 and vector search in parallel
  const expandedK = topK * 3; // fetch more candidates for better fusion

  const [vectorResults, bm25Results] = await Promise.all([
    (async () => {
      const queryEmbedding = await embedQuery(query);
      return vectorStore.search(queryEmbedding, expandedK, safeFilter);
    })(),
    Promise.resolve(bm25Index.search(query, expandedK, safeFilter)),
  ]);

  // Collect raw scores
  const vectorScores = new Map<string, number>();
  const bm25Scores = new Map<string, number>();
  const metadataMap = new Map<string, DocMetadata>();

  for (const r of vectorResults) {
    vectorScores.set(r.doc_id, r.score);
    metadataMap.set(r.doc_id, r.metadata);
  }
  for (const r of bm25Results) {
    bm25Scores.set(r.doc_id, r.score);
    if (!metadataMap.has(r.doc_id)) {
      metadataMap.set(r.doc_id, r.metadata);
    }
  }

  // Normalize scores to [0, 1]
  const normVector = normalizeScores(vectorScores);
  const normBM25 = normalizeScores(bm25Scores);

  // Fuse all candidate doc IDs
  const allDocIds = new Set([...normVector.keys(), ...normBM25.keys()]);
  const α = options?.vectorWeight ?? config.vectorWeight;

  const results: HybridSearchResult[] = [];
  for (const docId of allDocIds) {
    const vs = normVector.get(docId) ?? 0;
    const bs = normBM25.get(docId) ?? 0;
    const combined = α * vs + (1 - α) * bs;

    const metadata = metadataMap.get(docId);
    if (!metadata) continue;

    // Double-check canonical enforcement
    if (!metadata.canonical) continue;

    results.push({
      doc_id: docId,
      vector_score: vectorScores.get(docId) ?? 0,
      bm25_score: bm25Scores.get(docId) ?? 0,
      combined_score: combined,
      metadata,
    });
  }

  // Sort by combined score descending
  results.sort((a, b) => b.combined_score - a.combined_score);
  return results.slice(0, topK);
}

/**
 * BM25-only search fallback (when vector store is not available).
 * Useful for testing without embeddings.
 */
export function bm25OnlySearch(
  query: string,
  bm25Index: BM25Index,
  topK: number = config.defaultTopK,
  filter?: MetadataFilter
): HybridSearchResult[] {
  const safeFilter: MetadataFilter = { ...filter, canonical_only: true };
  const results = bm25Index.search(query, topK, safeFilter);

  return results
    .filter(r => r.metadata.canonical)
    .map(r => ({
      doc_id: r.doc_id,
      vector_score: 0,
      bm25_score: r.score,
      combined_score: r.score,
      metadata: r.metadata,
    }));
}
