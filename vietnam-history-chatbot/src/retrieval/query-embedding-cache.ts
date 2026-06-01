/**
 * Query Embedding Cache Lookup — PATCH 10C
 * 
 * Loads query embeddings from 10B cache. NEVER calls API.
 * Returns missing status if query not found.
 */
import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..', '..');
const CACHE_DIR = resolve(ROOT, 'cache');

interface QueryCacheEntry {
  id: string;
  text_hash: string;
  embedding: number[];
  metadata: { question: string; expected_terms?: string[] };
}

interface QueryCacheFile {
  patch: string; kind: string; model: string; dimensions: number;
  entries: QueryCacheEntry[];
}

export interface QueryEmbeddingLookup {
  found: boolean;
  embedding?: number[];
  queryId?: string;
  reason?: 'query_embedding_missing' | 'cache_missing';
}

let queryCache: Map<string, QueryCacheEntry> | null = null;
let questionIndex: Map<string, QueryCacheEntry> | null = null;

function loadCache(cachePath?: string): boolean {
  if (queryCache) return true;
  const path = cachePath ?? resolve(CACHE_DIR, 'query_embeddings_10b.json');
  if (!existsSync(path)) return false;
  const data: QueryCacheFile = JSON.parse(readFileSync(path, 'utf8'));
  queryCache = new Map();
  questionIndex = new Map();
  for (const e of data.entries) {
    queryCache.set(e.id, e);
    const normQ = e.metadata.question.trim().toLowerCase();
    questionIndex.set(normQ, e);
  }
  return true;
}

/** Lookup query embedding by eval ID (e.g. 'cap9i_092') */
export function lookupByQueryId(queryId: string, cachePath?: string): QueryEmbeddingLookup {
  if (!loadCache(cachePath)) return { found: false, reason: 'cache_missing' };
  const entry = queryCache!.get(queryId);
  if (!entry) return { found: false, reason: 'query_embedding_missing' };
  return { found: true, embedding: entry.embedding, queryId: entry.id };
}

/** Lookup query embedding by exact question text */
export function lookupByQuestion(question: string, cachePath?: string): QueryEmbeddingLookup {
  if (!loadCache(cachePath)) return { found: false, reason: 'cache_missing' };
  const norm = question.trim().toLowerCase();
  const entry = questionIndex!.get(norm);
  if (!entry) return { found: false, reason: 'query_embedding_missing' };
  return { found: true, embedding: entry.embedding, queryId: entry.id };
}

/** Reset cache (for testing) */
export function resetQueryCache(): void {
  queryCache = null;
  questionIndex = null;
}
