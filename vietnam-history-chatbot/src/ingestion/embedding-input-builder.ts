/**
 * Embedding Input Builder — constructs standardized embedding input strings
 * from IndexableDocument fields for text-embedding-3-large.
 *
 * Patch 8B: Prepends title, aliases, keywords, people, organizations,
 * and locations to text_for_embedding for improved semantic recall.
 *
 * Event docs format:
 *   [TITLE] {title}
 *   [ALIASES] {aliases}     — if present
 *   [KEYWORDS] {keywords}   — if present
 *   [PEOPLE] {people}       — if present
 *   [ORGANIZATIONS] {orgs}  — if present
 *   [LOCATIONS] {locs}      — if present
 *   {text_for_embedding}
 *
 * Synthesis docs format:
 *   [TITLE] {title}
 *   [KEYWORDS] {keywords}   — if present
 *   {text_for_embedding}
 */

import type { IndexableDocument } from '../shared/types.js';

/**
 * Build the embedding input string for a document.
 *
 * Deterministic for the same input. Omits empty sections.
 * Never crashes on undefined/null fields.
 */
export function buildEmbeddingInput(doc: IndexableDocument): string {
  const parts: string[] = [];

  // Title is always first — ensures high semantic weight
  parts.push(`[TITLE] ${doc.title}`);

  if (doc.doc_source === 'event') {
    // Event docs: include aliases, keywords, people, organizations, locations
    if (doc.aliases && doc.aliases.length > 0) {
      parts.push(`[ALIASES] ${doc.aliases.join(', ')}`);
    }

    if (doc.retrieval_keywords_vi && doc.retrieval_keywords_vi.length > 0) {
      parts.push(`[KEYWORDS] ${doc.retrieval_keywords_vi.join(', ')}`);
    }

    if (doc.people_labels && doc.people_labels.length > 0) {
      parts.push(`[PEOPLE] ${doc.people_labels.join(', ')}`);
    }

    if (doc.organization_labels && doc.organization_labels.length > 0) {
      parts.push(`[ORGANIZATIONS] ${doc.organization_labels.join(', ')}`);
    }

    if (doc.place_labels && doc.place_labels.length > 0) {
      parts.push(`[LOCATIONS] ${doc.place_labels.join(', ')}`);
    }
  } else {
    // Synthesis docs: only keywords (lighter metadata)
    if (doc.retrieval_keywords_vi && doc.retrieval_keywords_vi.length > 0) {
      parts.push(`[KEYWORDS] ${doc.retrieval_keywords_vi.join(', ')}`);
    }
  }

  // text_for_embedding is the main body — always included
  if (doc.text_for_embedding) {
    parts.push(doc.text_for_embedding);
  }

  return parts.join('\n');
}

/**
 * Compute a deterministic content hash input for a sorted list of documents.
 * Used by embedding-generator for cache invalidation.
 *
 * The hash covers doc_id + buildEmbeddingInput(doc) for each document,
 * sorted by doc_id to ensure determinism regardless of input order.
 */
export function computeEmbeddingContentString(docs: IndexableDocument[]): string {
  const sorted = [...docs].sort((a, b) => a.doc_id.localeCompare(b.doc_id));
  return sorted.map(doc => doc.doc_id + '\0' + buildEmbeddingInput(doc)).join('\n');
}
