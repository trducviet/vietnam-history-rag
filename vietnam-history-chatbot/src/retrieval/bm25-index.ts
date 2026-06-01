/**
 * BM25 Index — lexical retrieval with normalization pipeline.
 *
 * Baseline tokenizer uses whitespace splitting + Unicode NFKC normalization.
 * The Tokenizer interface allows plugging in stronger Vietnamese NLP
 * (e.g., underthesea, vncorenlp) without changing retrieval logic.
 */

import type { IndexableDocument, DocMetadata, MetadataFilter, SearchResult } from '../shared/types.js';

// ─── Tokenizer Interface ─────────────────────────────────────

/** Pluggable tokenizer — replace with Vietnamese word segmenter later */
export interface Tokenizer {
  tokenize(text: string): string[];
}

/**
 * Baseline tokenizer: lowercase → NFKC normalize → strip punctuation → split whitespace.
 * Sufficient for initial retrieval quality. Upgrade by implementing a Vietnamese
 * word-segmentation tokenizer behind the same interface.
 */
export class BaselineTokenizer implements Tokenizer {
  tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .normalize('NFKC')
      .replace(/[^\p{L}\p{N}\s]/gu, ' ')  // strip punctuation, keep letters/numbers
      .split(/\s+/)
      .filter(token => token.length > 0);
  }
}

// ─── BM25 Index ──────────────────────────────────────────────

interface BM25Document {
  doc_id: string;
  tokens: string[];
  metadata: DocMetadata;
}

/**
 * BM25 (Okapi BM25) inverted index for lexical retrieval.
 * Parameters follow standard defaults: k1=1.2, b=0.75.
 */
export class BM25Index {
  private documents: BM25Document[] = [];
  private invertedIndex: Map<string, Set<number>> = new Map();
  private avgDocLength: number = 0;
  private readonly k1: number;
  private readonly b: number;
  private tokenizer: Tokenizer;

  constructor(
    tokenizer?: Tokenizer,
    options?: { k1?: number; b?: number }
  ) {
    this.tokenizer = tokenizer ?? new BaselineTokenizer();
    this.k1 = options?.k1 ?? 1.2;
    this.b = options?.b ?? 0.75;
  }

  /** Index all canonical documents */
  indexDocuments(documents: IndexableDocument[]): void {
    this.documents = [];
    this.invertedIndex = new Map();

    let totalTokens = 0;

    for (const doc of documents) {
      const tokens = this.tokenizer.tokenize(doc.text_for_embedding);
      const idx = this.documents.length;

      const metadata: DocMetadata = {
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

      this.documents.push({ doc_id: doc.doc_id, tokens, metadata });

      // Build inverted index
      const uniqueTokens = new Set(tokens);
      for (const token of uniqueTokens) {
        if (!this.invertedIndex.has(token)) {
          this.invertedIndex.set(token, new Set());
        }
        this.invertedIndex.get(token)!.add(idx);
      }

      totalTokens += tokens.length;
    }

    this.avgDocLength = this.documents.length > 0
      ? totalTokens / this.documents.length
      : 0;

    console.log(`📝 BM25 index built: ${this.documents.length} docs, ${this.invertedIndex.size} unique terms, avg length: ${Math.round(this.avgDocLength)}`);
  }

  /** Number of indexed documents */
  size(): number {
    return this.documents.length;
  }

  /** Number of unique terms in the index */
  vocabularySize(): number {
    return this.invertedIndex.size;
  }

  /** Search with BM25 scoring, optionally filtered by metadata */
  search(query: string, topK: number, filter?: MetadataFilter): SearchResult[] {
    const queryTokens = this.tokenizer.tokenize(query);
    const N = this.documents.length;
    const scores = new Map<number, number>();

    for (const qToken of queryTokens) {
      const postings = this.invertedIndex.get(qToken);
      if (!postings) continue;

      // IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
      const df = postings.size;
      const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);

      for (const docIdx of postings) {
        const doc = this.documents[docIdx];

        // Apply metadata filter
        if (filter && !this.matchesFilter(doc.metadata, filter)) continue;

        // Term frequency in document
        let tf = 0;
        for (const t of doc.tokens) {
          if (t === qToken) tf++;
        }

        // BM25 TF component
        const docLen = doc.tokens.length;
        const tfNorm = (tf * (this.k1 + 1)) /
          (tf + this.k1 * (1 - this.b + this.b * docLen / this.avgDocLength));

        const currentScore = scores.get(docIdx) || 0;
        scores.set(docIdx, currentScore + idf * tfNorm);
      }
    }

    // Sort by score descending
    const results: SearchResult[] = [];
    for (const [docIdx, score] of scores) {
      const doc = this.documents[docIdx];
      results.push({
        doc_id: doc.doc_id,
        score,
        metadata: doc.metadata,
      });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /** Check metadata filter match */
  private matchesFilter(meta: DocMetadata, filter: MetadataFilter): boolean {
    if (filter.doc_source && meta.doc_source !== filter.doc_source) return false;
    if (filter.doc_type && meta.doc_type !== filter.doc_type) return false;
    if (filter.period_label && meta.period_label !== filter.period_label) return false;
    if (filter.event_status && meta.event_status !== filter.event_status) return false;
    if (filter.verification_status && meta.verification_status !== filter.verification_status)
      return false;
    if (filter.canonical_only && !meta.canonical) return false;
    if (meta.year !== null) {
      if (filter.year_min !== undefined && meta.year < filter.year_min) return false;
      if (filter.year_max !== undefined && meta.year > filter.year_max) return false;
    } else {
      if (filter.year_min !== undefined || filter.year_max !== undefined) return false;
    }
    return true;
  }
}
