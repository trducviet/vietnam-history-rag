/**
 * Citation Enricher — attaches provenance data from sources.jsonl
 * to LLM-generated or fallback citations.
 *
 * This module never hallucinate sources. It only enriches citations
 * with data actually present in the loaded dataset.
 *
 * PATCH 4: Added defensive guards:
 *   - Empty citation fallback from context bundle
 *   - Missing title fill from dataset
 *   - Relevance normalization
 *   - source_ids preserved even when sources can't resolve
 */

import type {
  Citation,
  ContextBundle,
  LoadedDataset,
  IndexableDocument,
} from '../shared/types.js';

// ─── Relevance Normalization ─────────────────────────────────

/**
 * Normalize citation relevance to a consistent string value.
 * Handles various input types (number, string, undefined).
 */
export function normalizeCitationRelevance(value: unknown): string {
  if (typeof value === 'string') {
    const lower = value.toLowerCase().trim();
    if (lower === 'primary' || lower === '1' || lower === '1.0') return 'primary';
    if (lower === 'supporting' || lower === '0.8') return 'supporting';
    if (lower === 'background' || lower === '0.5') return 'background';
    if (lower.length > 0) return lower; // preserve custom relevance text
    return 'supporting';
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (value >= 0.9) return 'primary';
    if (value >= 0.6) return 'supporting';
    return 'background';
  }
  return 'supporting';
}

// ─── Citation Enrichment ─────────────────────────────────────

/**
 * Enrich citations with provenance from dataset sources.
 *
 * For each citation:
 * 1. Normalize relevance
 * 2. Find the matching document in the dataset
 * 3. Fill missing title from dataset
 * 4. Attach doc_source, doc_type, year
 * 5. Resolve source_ids to full SourceDocument objects
 */
export function enrichCitations(
  citations: Citation[],
  contextBundle: ContextBundle,
  dataset: LoadedDataset
): Citation[] {
  let enriched = citations.map(citation => enrichSingleCitation(citation, dataset));

  // Guard: if citations is empty, build from context bundle
  if (enriched.length === 0) {
    enriched = buildFallbackCitationsFromContext(contextBundle, dataset);
  }

  return enriched;
}

/** Enrich a single citation with provenance */
function enrichSingleCitation(
  citation: Citation,
  dataset: LoadedDataset
): Citation {
  // Normalize relevance
  const normalizedRelevance = normalizeCitationRelevance(citation.relevance);

  const doc = findDocument(citation.record_id, dataset);
  if (!doc) {
    return {
      ...citation,
      relevance: normalizedRelevance,
    };
  }

  // Fill missing title from dataset
  const title = citation.title || doc.title;

  // Resolve source_ids from the document
  const sourceIds = doc.source_ids ?? [];
  const sources = sourceIds
    .map(id => dataset.sources.get(id))
    .filter((s): s is NonNullable<typeof s> => s != null);

  return {
    ...citation,
    title,
    relevance: normalizedRelevance,
    doc_source: doc.doc_source,
    doc_type: doc.doc_type,
    year: doc.year,
    // Keep source_ids even if sources can't resolve (traceability)
    source_ids: sourceIds.length > 0 ? sourceIds : undefined,
    sources: sources.length > 0 ? sources : undefined,
  };
}

// ─── Fallback Citation Builder ───────────────────────────────

/**
 * Build fallback citations from context bundle when LLM doesn't provide them.
 *
 * Patch 7F-1: If citation_plan is present, use it (sorted by priority) so that
 * only role-approved docs are cited. Falls back to primary+supporting when absent.
 */
export function buildFallbackCitationsFromContext(
  bundle: ContextBundle,
  dataset: LoadedDataset
): Citation[] {
  let citations: Citation[];

  if (bundle.citation_plan && bundle.citation_plan.length > 0) {
    // Use citation_plan — respect role-aware filtering
    const sorted = [...bundle.citation_plan]
      .filter(item => item.citation_role !== 'excluded')
      .sort((a, b) => (a.priority ?? 99) - (b.priority ?? 99));

    citations = sorted.map(item => ({
      record_id: item.doc_id,
      // Map citation role to relevance string
      relevance: item.citation_role === 'primary' ? 'primary'
               : item.citation_role === 'contrast' ? 'contrast'
               : 'supporting',
      title: '', // will be filled by enrichment
    }));
  } else {
    // Legacy: primary + supporting
    citations = [
      ...bundle.primary_docs.map(doc => docToCitation(doc, 'primary')),
      ...bundle.supporting_docs.map(doc => docToCitation(doc, 'supporting')),
    ];
  }

  // Enrich without recursive fallback
  return citations.map(citation => enrichSingleCitation(citation, dataset));
}

/** Convert an IndexableDocument to a basic Citation */
function docToCitation(doc: IndexableDocument, relevance: string): Citation {
  return {
    record_id: doc.doc_id,
    title: doc.title,
    relevance,
  };
}

// ─── Helpers ─────────────────────────────────────────────────

function findDocument(docId: string, dataset: LoadedDataset): IndexableDocument | null {
  return dataset.events.get(docId) ?? dataset.synthesis.get(docId) ?? dataset.disambiguationRules.get(docId) ?? null;
}
