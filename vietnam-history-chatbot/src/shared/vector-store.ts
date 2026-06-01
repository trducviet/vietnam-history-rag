/**
 * Vector Store Abstraction Layer.
 *
 * Provides a clean interface so the retrieval pipeline can work with
 * in-memory vectors today and migrate to Qdrant / pgvector later
 * without touching core retrieval logic.
 */

import type { DocMetadata, MetadataFilter, SearchResult } from './types.js';

// ─── Interface ───────────────────────────────────────────────

export interface VectorStore {
  /** Insert or update a vector with metadata */
  upsert(id: string, embedding: number[], metadata: DocMetadata): Promise<void>;

  /** Search for the top-k most similar vectors, optionally filtered */
  search(queryEmbedding: number[], topK: number, filter?: MetadataFilter): Promise<SearchResult[]>;

  /** Remove a vector by id */
  delete(id: string): Promise<void>;

  /** Number of stored vectors */
  size(): number;

  /** Serialize the store for caching */
  serialize(): string;

  /** Deserialize from cached data */
  deserialize(data: string): void;
}

// ─── In-Memory Implementation ────────────────────────────────

interface StoredVector {
  id: string;
  embedding: number[];
  metadata: DocMetadata;
}

/**
 * Baseline in-memory vector store using brute-force cosine similarity.
 * Sufficient for ~526 documents. Replace with QdrantVectorStore or
 * PgVectorStore via the same interface when scaling.
 */
export class InMemoryVectorStore implements VectorStore {
  private vectors: Map<string, StoredVector> = new Map();

  async upsert(id: string, embedding: number[], metadata: DocMetadata): Promise<void> {
    this.vectors.set(id, { id, embedding, metadata });
  }

  async search(
    queryEmbedding: number[],
    topK: number,
    filter?: MetadataFilter
  ): Promise<SearchResult[]> {
    const candidates: SearchResult[] = [];

    for (const stored of this.vectors.values()) {
      if (filter && !matchesFilter(stored.metadata, filter)) continue;

      const score = cosineSimilarity(queryEmbedding, stored.embedding);
      candidates.push({
        doc_id: stored.id,
        score,
        metadata: stored.metadata,
      });
    }

    candidates.sort((a, b) => b.score - a.score);
    return candidates.slice(0, topK);
  }

  async delete(id: string): Promise<void> {
    this.vectors.delete(id);
  }

  size(): number {
    return this.vectors.size;
  }

  serialize(): string {
    const entries: Array<{ id: string; embedding: number[]; metadata: DocMetadata }> = [];
    for (const v of this.vectors.values()) {
      entries.push({ id: v.id, embedding: v.embedding, metadata: v.metadata });
    }
    return JSON.stringify(entries);
  }

  deserialize(data: string): void {
    const entries = JSON.parse(data) as Array<{
      id: string;
      embedding: number[];
      metadata: DocMetadata;
    }>;
    this.vectors.clear();
    for (const entry of entries) {
      this.vectors.set(entry.id, entry);
    }
  }
}

// ─── Helpers ─────────────────────────────────────────────────

/** Cosine similarity between two vectors */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/** Check if metadata satisfies the given filter */
function matchesFilter(meta: DocMetadata, filter: MetadataFilter): boolean {
  if (filter.doc_source && meta.doc_source !== filter.doc_source) return false;
  if (filter.doc_type && meta.doc_type !== filter.doc_type) return false;
  if (filter.period_label && meta.period_label !== filter.period_label) return false;
  if (filter.event_status && meta.event_status !== filter.event_status) return false;
  if (filter.verification_status && meta.verification_status !== filter.verification_status)
    return false;
  if (filter.canonical_only && !meta.canonical) return false;

  // Year range filtering
  const docYear = meta.year;
  if (docYear !== null) {
    if (filter.year_min !== undefined && docYear < filter.year_min) return false;
    if (filter.year_max !== undefined && docYear > filter.year_max) return false;
  } else {
    // Documents without year pass unless a strict year filter is set
    if (filter.year_min !== undefined || filter.year_max !== undefined) return false;
  }

  return true;
}
