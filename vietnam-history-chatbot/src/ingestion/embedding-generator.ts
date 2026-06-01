/**
 * Embedding Generator — generates and caches embeddings using
 * OpenAI text-embedding-3-large for canonical documents.
 *
 * PATCH 2: Supports custom cachePath per call, enabling separate
 * event and synthesis embedding caches.
 *
 * PATCH 8B: Content-hash cache validation via SHA-256.
 * Uses buildEmbeddingInput() for enriched embedding input strings.
 *
 * PATCH 8B-A: Strict generate-mode cache validation.
 * - Requires dimensions match.
 * - Legacy caches (no content_hash) are ALWAYS stale in generate mode
 *   because Patch 8B changed embedding input format (buildEmbeddingInput).
 * - Load-only mode (loadEmbeddingCacheWithStats) still accepts legacy
 *   caches with warning for backward compatibility.
 */

import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'fs';
import { dirname } from 'path';
import { createHash } from 'crypto';
import OpenAI from 'openai';
import { config } from '../shared/config.js';
import type { IndexableDocument, DocMetadata } from '../shared/types.js';
import type { VectorStore } from '../shared/vector-store.js';
import { buildEmbeddingInput, computeEmbeddingContentString } from './embedding-input-builder.js';

// ─── Types ───────────────────────────────────────────────────

interface EmbeddingCacheEntry {
  doc_id: string;
  embedding: number[];
  metadata: DocMetadata;
}

interface EmbeddingCache {
  model: string;
  dimensions: number;
  /** SHA-256 of sorted doc_id + buildEmbeddingInput(doc). Patch 8B. */
  content_hash?: string;
  created_at: string;
  entries: EmbeddingCacheEntry[];
}

export interface EmbeddingOptions {
  /** Override cache file path (default: config.embeddingCachePath) */
  cachePath?: string;
  /** Override embedding model (default: config.embeddingModel) */
  model?: string;
  /** Force re-generation even if cache exists */
  forceRefresh?: boolean;
}

// ─── Embedding Generation ────────────────────────────────────

/** Extract DocMetadata from an IndexableDocument */
function extractMetadata(doc: IndexableDocument): DocMetadata {
  return {
    doc_id: doc.doc_id,
    doc_source: doc.doc_source,
    doc_type: doc.doc_type,
    title: doc.title,
    year: doc.year,
    end_year: doc.end_year,
    period_label: doc.period_label,
    event_status: doc.event_status,
    verification_status: doc.verification_status,
    canonical: doc.canonical,
  };
}

// ─── Content Hash (Patch 8B) ─────────────────────────────────

/**
 * Compute a deterministic SHA-256 content hash for a list of documents.
 * Hash covers doc_id + buildEmbeddingInput(doc) for each document,
 * sorted by doc_id to ensure determinism.
 */
export function computeContentHash(documents: IndexableDocument[]): string {
  const contentString = computeEmbeddingContentString(documents);
  return createHash('sha256').update(contentString).digest('hex');
}

/**
 * Generate embeddings for documents and load them into the vector store.
 *
 * PATCH 2: Accepts options.cachePath so event and synthesis embeddings
 * can be cached to separate files.
 *
 * PATCH 8B-A: Strict cache validation in generate mode:
 * - model must match
 * - dimensions must match
 * - entry count must match
 * - content_hash must be present AND match
 * Legacy caches (no content_hash) are ALWAYS regenerated because
 * Patch 8B changed embedding input format (buildEmbeddingInput).
 *
 * CANONICAL ENFORCEMENT: Only canonical documents should be passed in.
 * Duplicate/redirect documents must never be embedded.
 */
export async function generateAndLoadEmbeddings(
  documents: IndexableDocument[],
  vectorStore: VectorStore,
  options?: EmbeddingOptions
): Promise<void> {
  const cachePath = options?.cachePath ?? config.embeddingCachePath;
  const model = options?.model ?? config.embeddingModel;
  const forceRefresh = options?.forceRefresh ?? false;

  // Compute content hash for current documents (Patch 8B)
  const currentHash = computeContentHash(documents);
  console.log(`📐 Content hash: ${currentHash.substring(0, 16)}... (${documents.length} docs, model=${model}, dims=${config.embeddingDimensions})`);

  // Try loading from cache first
  if (!forceRefresh && existsSync(cachePath)) {
    console.log(`💾 Loading embeddings from cache: ${cachePath}`);
    const cached = JSON.parse(readFileSync(cachePath, 'utf-8')) as EmbeddingCache;

    // Patch 8B-A: Full validation — model + dimensions + entry count + content_hash
    const modelMatch = cached.model === model;
    const dimensionsMatch = cached.dimensions === config.embeddingDimensions;
    const countMatch = cached.entries.length === documents.length;
    const hasHash = !!cached.content_hash;
    const hashMatch = hasHash && cached.content_hash === currentHash;

    // Patch 8B-A: In generate mode, ALL five conditions must be true.
    // Legacy caches without content_hash are always stale because
    // Patch 8B changed embedding input from raw text_for_embedding
    // to enriched buildEmbeddingInput() (title + aliases + keywords).
    const isValidCache = modelMatch && dimensionsMatch && countMatch && hasHash && hashMatch;

    if (!hasHash) {
      console.warn('⚠️  Embedding cache has no content_hash (legacy format); regenerating because enriched embedding input may have changed.');
    }

    if (isValidCache) {
      // Cache is valid: load it
      for (const entry of cached.entries) {
        await vectorStore.upsert(entry.doc_id, entry.embedding, entry.metadata);
      }
      console.log(`✅ Loaded ${cached.entries.length} cached embeddings into vector store (content_hash=MATCH)`);
      return;
    }

    // Cache is stale — log detailed reasons
    const reasons: string[] = [];
    if (!modelMatch) reasons.push(`model: ${cached.model} vs ${model}`);
    if (!dimensionsMatch) reasons.push(`dimensions: ${cached.dimensions} vs ${config.embeddingDimensions}`);
    if (!countMatch) reasons.push(`entries: ${cached.entries.length} vs ${documents.length}`);
    if (!hasHash) reasons.push(`missing content_hash (legacy format)`);
    else if (!hashMatch) reasons.push(`content_hash: MISMATCH`);
    console.log(`⚠️  Cache stale (${reasons.join(', ')}). Regenerating...`);
  }

  // Generate new embeddings via OpenAI API
  if (!config.openaiApiKey) {
    throw new Error(
      'OPENAI_API_KEY is required for embedding generation. Set it in .env file.'
    );
  }

  const openai = new OpenAI({ apiKey: config.openaiApiKey });
  const cacheEntries: EmbeddingCacheEntry[] = [];
  const batchSize = 20;

  console.log(`🔄 Generating embeddings for ${documents.length} documents (model: ${model})...`);
  console.log(`   Using buildEmbeddingInput() for enriched input (title + aliases + keywords)`);

  for (let i = 0; i < documents.length; i += batchSize) {
    const batch = documents.slice(i, i + batchSize);
    // Patch 8B: Use buildEmbeddingInput for enriched input
    const texts = batch.map(doc => buildEmbeddingInput(doc));

    const response = await openai.embeddings.create({
      model,
      input: texts,
    });

    for (let j = 0; j < batch.length; j++) {
      const doc = batch[j];
      const embedding = response.data[j].embedding;
      const metadata = extractMetadata(doc);

      await vectorStore.upsert(doc.doc_id, embedding, metadata);
      cacheEntries.push({ doc_id: doc.doc_id, embedding, metadata });
    }

    const progress = Math.min(i + batchSize, documents.length);
    console.log(`   ${progress}/${documents.length} embeddings generated`);
  }

  // Save cache to disk with content_hash (Patch 8B)
  const cacheDir = dirname(cachePath);
  if (!existsSync(cacheDir)) {
    mkdirSync(cacheDir, { recursive: true });
  }

  const cache: EmbeddingCache = {
    model,
    dimensions: config.embeddingDimensions,
    content_hash: currentHash,
    created_at: new Date().toISOString(),
    entries: cacheEntries,
  };

  writeFileSync(cachePath, JSON.stringify(cache));
  console.log(`💾 Embeddings cached to: ${cachePath}`);
  console.log(`   content_hash: ${currentHash.substring(0, 16)}...`);
  console.log(`✅ ${cacheEntries.length} embeddings loaded into vector store`);
}

/** Result of loading an embedding cache */
export interface EmbeddingCacheLoadResult {
  loaded: boolean;
  entries: number;
  path: string;
  reason?: string;
  /** Whether cache has content_hash (Patch 8B) */
  hasContentHash?: boolean;
  /** Whether cache is a legacy format without content_hash */
  isLegacy?: boolean;
}

/**
 * Load a pre-built embedding cache into a vector store.
 * Does NOT call the OpenAI API. Returns false if cache doesn't exist or is invalid.
 */
export function loadEmbeddingCacheToVectorStore(
  cachePath: string,
  vectorStore: VectorStore
): boolean {
  const result = loadEmbeddingCacheWithStats(cachePath, vectorStore);
  return result.loaded;
}

/**
 * Load a pre-built embedding cache with detailed stats.
 * Does NOT call the OpenAI API.
 */
export function loadEmbeddingCacheWithStats(
  cachePath: string,
  vectorStore: VectorStore
): EmbeddingCacheLoadResult {
  if (!existsSync(cachePath)) {
    return { loaded: false, entries: 0, path: cachePath, reason: 'Cache file not found' };
  }

  try {
    const cached = JSON.parse(readFileSync(cachePath, 'utf-8')) as EmbeddingCache;
    if (!cached.entries || cached.entries.length === 0) {
      return { loaded: false, entries: 0, path: cachePath, reason: 'Cache is empty' };
    }

    // Patch 8B: Detect legacy caches
    const hasContentHash = !!cached.content_hash;
    const isLegacy = !hasContentHash;
    if (isLegacy) {
      console.warn(`⚠️  Embedding cache has no content_hash (legacy format); regenerate recommended: ${cachePath}`);
    } else {
      console.log(`📐 Cache content_hash: ${cached.content_hash!.substring(0, 16)}...`);
    }

    for (const entry of cached.entries) {
      // Synchronous upsert via fire-and-forget (InMemoryVectorStore is sync internally)
      void vectorStore.upsert(entry.doc_id, entry.embedding, entry.metadata);
    }

    console.log(`✅ Loaded ${cached.entries.length} embeddings from ${cachePath}`);
    return { loaded: true, entries: cached.entries.length, path: cachePath, hasContentHash, isLegacy };
  } catch (e) {
    console.warn(`⚠️  Failed to load embedding cache ${cachePath}:`, (e as Error).message);
    return { loaded: false, entries: 0, path: cachePath, reason: (e as Error).message };
  }
}
