/**
 * Vector Cache Loader — PATCH 10C
 * 
 * Loads pre-built embedding caches into InMemoryVectorStore.
 * NEVER calls API. Fails gracefully if cache missing.
 */
import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { InMemoryVectorStore } from '../shared/vector-store.js';
import type { DocMetadata } from '../shared/types.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..', '..');
const CACHE_DIR = resolve(ROOT, 'cache');

interface CacheFile {
  patch: string; kind: string; model: string; dimensions: number;
  content_hash: string; doc_count: number; entry_count: number;
  entries: Array<{ id: string; text_hash: string; embedding: number[]; metadata: Record<string, unknown> }>;
}

export interface VectorCacheLoadResult {
  eventStore?: InMemoryVectorStore;
  synthesisStore?: InMemoryVectorStore;
  eventSize: number;
  synthesisSize: number;
  dimensions: number;
  cacheLoaded: boolean;
  error?: string;
  apiAttempts: 0;
}

export async function loadVectorCaches(
  eventCachePath?: string,
  synthesisCachePath?: string
): Promise<VectorCacheLoadResult> {
  const evtPath = eventCachePath ?? resolve(CACHE_DIR, 'event_embeddings_10b.json');
  const synPath = synthesisCachePath ?? resolve(CACHE_DIR, 'synthesis_embeddings_10b.json');

  if (!existsSync(evtPath)) {
    return { eventSize: 0, synthesisSize: 0, dimensions: 0, cacheLoaded: false, error: 'event cache missing: ' + evtPath, apiAttempts: 0 };
  }
  if (!existsSync(synPath)) {
    return { eventSize: 0, synthesisSize: 0, dimensions: 0, cacheLoaded: false, error: 'synthesis cache missing: ' + synPath, apiAttempts: 0 };
  }

  const evtCache: CacheFile = JSON.parse(readFileSync(evtPath, 'utf8'));
  const synCache: CacheFile = JSON.parse(readFileSync(synPath, 'utf8'));

  const evtStore = new InMemoryVectorStore();
  for (const e of evtCache.entries) await evtStore.upsert(e.id, e.embedding, e.metadata as unknown as DocMetadata);
  const synStore = new InMemoryVectorStore();
  for (const e of synCache.entries) await synStore.upsert(e.id, e.embedding, e.metadata as unknown as DocMetadata);

  return {
    eventStore: evtStore,
    synthesisStore: synStore,
    eventSize: evtStore.size(),
    synthesisSize: synStore.size(),
    dimensions: evtCache.dimensions,
    cacheLoaded: true,
    apiAttempts: 0,
  };
}
