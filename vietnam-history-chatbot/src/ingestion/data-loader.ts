/**
 * Data Loader — reads JSONL files from the dataset pack, validates, and builds
 * in-memory lookup structures. Enforces canonical/duplicate distinction.
 *
 * Patch 7C-1: Enriches runtime documents from corpus/events.jsonl:
 *   - retrieval.keywords_vi → retrieval_keywords_vi
 *   - retrieval.aliases → aliases
 *   - retrieval.hard_negative_ids → hard_negative_ids
 *   - retrieval.related_record_ids → related_record_ids
 *   - benchmark fields → benchmark_role, benchmark_target_query_ids, etc.
 *   - classification.category_l1/l2/l3 → classification
 *   - Attaches semantic_features via extractSemanticFeatures
 */

import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { config } from '../shared/config.js';
import type {
  EventDocument,
  SynthesisDocument,
  DisambiguationRuleDocument,
  IndexableDocument,
  SourceDocument,
  EventEntityLink,
  SynthesisEventLink,
  RuntimeProvenanceLink,
  QABenchmarkEntry,
  RetrievalQuery,
  HardNegative,
  LoadedDataset,
} from '../shared/types.js';
import { extractSemanticFeatures } from '../indexing/semantic-feature-extractor.js';

// ─── JSONL Parser ────────────────────────────────────────────

function parseJsonl<T>(filePath: string): T[] {
  const content = readFileSync(filePath, 'utf-8');
  return content
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)
    .map((line, idx) => {
      try {
        return JSON.parse(line) as T;
      } catch (e) {
        throw new Error(`Failed to parse line ${idx + 1} in ${filePath}: ${(e as Error).message}`);
      }
    });
}

// ─── Data Loading ────────────────────────────────────────────

/** Resolve a path relative to the dataset pack root */
function dataFile(relativePath: string): string {
  return resolve(config.dataPath, relativePath);
}

// ─── Full Corpus Enrichment Types ────────────────────────────

/** Raw shape of a record in corpus/events.jsonl (enrichment source only) */
interface RawFullCorpusRecord {
  record_id?: string;
  retrieval?: {
    keywords_vi?: string[];
    aliases?: string[];
    hard_negative_ids?: string[];
    related_record_ids?: string[];
    rag_cluster?: string | null;
    indexing_hint?: string;
  };
  benchmark?: {
    benchmark_role?: string;
    target_query_ids?: string[];
    acceptable_query_ids?: string[];
    hard_negative_query_ids?: string[];
  };
  classification?: {
    category_l1?: string;
    category_l2?: string;
    category_l3?: string;
  };
  evidence?: {
    verification_status?: string;
    source_ids?: string[];
  };
}

/**
 * Attempt to load corpus/events.jsonl as an enrichment map.
 * Returns empty map and logs a warning if the file is not found.
 * Does NOT crash — runtime index is the source of truth.
 */
function loadFullCorpusEnrichmentMap(): Map<string, RawFullCorpusRecord> {
  const enrichMap = new Map<string, RawFullCorpusRecord>();
  const candidatePaths = [
    dataFile('corpus/events.jsonl'),
    dataFile('../corpus/events.jsonl'),
  ];
  const foundPath = candidatePaths.find(p => existsSync(p));
  if (!foundPath) {
    console.warn('⚠️  corpus/events.jsonl not found — enrichment will use runtime fields only');
    return enrichMap;
  }
  const raw = parseJsonl<RawFullCorpusRecord>(foundPath);
  let skipped = 0;
  for (const rec of raw) {
    if (rec.record_id) {
      enrichMap.set(rec.record_id, rec);
    } else {
      skipped++;
    }
  }
  console.log(`📚 Full corpus enrichment map: ${enrichMap.size} records loaded from ${foundPath}${skipped > 0 ? ` (${skipped} skipped, missing record_id)` : ''}`);
  return enrichMap;
}

/**
 * Enrich a single runtime event doc with fields from the full corpus record.
 * Mutates the doc in place. Safe to call if fullRec is undefined.
 */
function enrichEventDoc(
  doc: EventDocument,
  fullRec: RawFullCorpusRecord | undefined
): void {
  if (fullRec) {
    // Retrieval enrichment
    if (fullRec.retrieval?.keywords_vi?.length) {
      doc.retrieval_keywords_vi = fullRec.retrieval.keywords_vi;
    }
    if (fullRec.retrieval?.aliases?.length) {
      doc.aliases = fullRec.retrieval.aliases;
    }
    if (fullRec.retrieval?.hard_negative_ids?.length) {
      doc.hard_negative_ids = fullRec.retrieval.hard_negative_ids;
    }
    if (fullRec.retrieval?.related_record_ids?.length) {
      doc.related_record_ids = fullRec.retrieval.related_record_ids;
    }
    // rag_cluster override if present in full corpus
    if (fullRec.retrieval?.rag_cluster !== undefined) {
      doc.rag_cluster = fullRec.retrieval.rag_cluster;
    }

    // Benchmark enrichment
    if (fullRec.benchmark) {
      doc.benchmark_role = fullRec.benchmark.benchmark_role;
      doc.benchmark_target_query_ids = fullRec.benchmark.target_query_ids;
      doc.benchmark_acceptable_query_ids = fullRec.benchmark.acceptable_query_ids;
      doc.benchmark_hard_negative_query_ids = fullRec.benchmark.hard_negative_query_ids;
    }

    // Classification enrichment
    if (fullRec.classification) {
      doc.classification = {
        category_l1: fullRec.classification.category_l1,
        category_l2: fullRec.classification.category_l2,
        category_l3: fullRec.classification.category_l3,
      };
    }

    // Evidence fallback: if runtime doc has no source_ids, try full corpus
    if ((!doc.source_ids || doc.source_ids.length === 0) &&
        fullRec.evidence?.source_ids?.length) {
      doc.source_ids = fullRec.evidence.source_ids;
    }
    // verification_status fallback
    if (!doc.verification_status && fullRec.evidence?.verification_status) {
      doc.verification_status = fullRec.evidence.verification_status as EventDocument['verification_status'];
    }
  }

  // Always attach semantic_features (with or without full corpus)
  doc.semantic_features = extractSemanticFeatures({
    doc_id: doc.doc_id,
    title: doc.title,
    summary: doc.summary,
    text_for_embedding: doc.text_for_embedding,
    doc_type: doc.doc_type,
    doc_kind: doc.doc_kind,
    doc_source: doc.doc_source,
    event_status: doc.event_status,
    year: doc.year,
    end_year: doc.end_year,
    period_label: doc.period_label,
    verification_status: doc.verification_status,
    significance_level: doc.significance_level,
    source_ids: doc.source_ids,
    retrieval_keywords_vi: doc.retrieval_keywords_vi,
    aliases: doc.aliases,
    classification: doc.classification,
  });
}

/** Load all events and separate canonical from duplicates */
function loadEvents(
  enrichMap: Map<string, RawFullCorpusRecord>
): { all: EventDocument[]; canonical: EventDocument[]; map: Map<string, EventDocument> } {
  const all = parseJsonl<EventDocument>(dataFile('corpus/runtime/events_indexable.jsonl'));
  const map = new Map<string, EventDocument>();
  const canonical: EventDocument[] = [];

  let enriched = 0;
  let missingEnrich = 0;

  for (const evt of all) {
    // Enrich with full corpus metadata + semantic features
    const fullRec = enrichMap.get(evt.doc_id);
    enrichEventDoc(evt, fullRec);
    if (fullRec) enriched++;
    else missingEnrich++;

    map.set(evt.doc_id, evt);
    if (evt.canonical) {
      canonical.push(evt);
    }
  }

  console.log(`📄 Events loaded: ${all.length} total, ${canonical.length} canonical, ${all.length - canonical.length} duplicates`);
  console.log(`   Corpus enrichment: ${enriched} enriched, ${missingEnrich} using runtime-only fields`);
  return { all, canonical, map };
}

/** Load all synthesis documents and attach lightweight semantic_features */
function loadSynthesis(): { all: SynthesisDocument[]; map: Map<string, SynthesisDocument> } {
  const all = parseJsonl<SynthesisDocument>(dataFile('corpus/runtime/synthesis_indexable.jsonl'));
  const map = new Map<string, SynthesisDocument>();

  for (const syn of all) {
    // Attach semantic_features for synthesis docs (lightweight — no full corpus enrichment needed)
    syn.semantic_features = extractSemanticFeatures({
      doc_id: syn.doc_id,
      title: syn.title,
      summary: syn.summary,
      text_for_embedding: syn.text_for_embedding,
      doc_type: syn.doc_type,
      doc_kind: syn.doc_kind,
      doc_source: syn.doc_source,
      event_status: syn.event_status,
      year: syn.year,
      end_year: syn.end_year,
      period_label: syn.period_label,
      verification_status: syn.verification_status,
      significance_level: syn.significance_level,
      source_ids: syn.source_ids,
    });
    map.set(syn.doc_id, syn);
  }

  console.log(`📄 Synthesis loaded: ${all.length} documents`);
  return { all, map };
}

interface RawDisambiguationRule {
  rule_id?: string;
  title?: string;
  topic?: string;
  period_label?: string | null;
  period?: string | null;
  confusion_area?: string;
  wrong_or_ambiguous_interpretation?: string;
  correct_interpretation?: string;
  recommendation_for_RAG?: string;
  example_user_questions?: string[];
  risk_level?: string;
  source_ids?: string[];
  source_pack_id?: string;
  pack_id?: string;
  verification_status?: string;
  runtime_merge_status?: string;
  embedding_status?: string;
  needs_embedding_rebuild?: boolean;
}

function compactText(parts: Array<string | undefined | null>): string {
  return parts
    .map(part => String(part ?? '').trim())
    .filter(part => part.length > 0)
    .join('\n');
}

function normalizeRuleAsDocument(rule: RawDisambiguationRule): DisambiguationRuleDocument {
  const ruleId = String(rule.rule_id ?? '');
  const title = String(rule.title ?? rule.confusion_area ?? rule.topic ?? ruleId);
  const summary = compactText([
    rule.correct_interpretation,
    rule.recommendation_for_RAG,
    rule.wrong_or_ambiguous_interpretation
      ? `Dễ nhầm: ${rule.wrong_or_ambiguous_interpretation}`
      : undefined,
  ]) || title;
  const textForEmbedding = compactText([
    `[TIÊU ĐỀ] ${title}`,
    `[LOẠI] disambiguation_rule`,
    `[GIAI ĐOẠN] ${rule.period_label ?? rule.period ?? ''}`,
    `[VẤN ĐỀ DỄ NHẦM] ${rule.confusion_area ?? title}`,
    rule.wrong_or_ambiguous_interpretation
      ? `[CÁCH HIỂU SAI] ${rule.wrong_or_ambiguous_interpretation}`
      : undefined,
    rule.correct_interpretation
      ? `[CÁCH HIỂU ĐÚNG] ${rule.correct_interpretation}`
      : undefined,
    rule.recommendation_for_RAG
      ? `[QUY TẮC RAG] ${rule.recommendation_for_RAG}`
      : undefined,
    rule.example_user_questions?.length
      ? `[VÍ DỤ CÂU HỎI] ${rule.example_user_questions.join('; ')}`
      : undefined,
    rule.source_ids?.length
      ? `[NGUỒN] ${rule.source_ids.join(', ')}`
      : undefined,
  ]);

  const doc: DisambiguationRuleDocument = {
    ...rule,
    doc_id: ruleId,
    rule_id: ruleId,
    doc_source: 'disambiguation_rule',
    doc_kind: 'disambiguation_rule',
    title,
    summary,
    text_for_embedding: textForEmbedding || summary,
    year: null,
    end_year: null,
    period_id: null,
    period_label: rule.period_label ?? rule.period ?? null,
    doc_type: 'disambiguation_rule',
    event_status: 'rule',
    verification_status: rule.verification_status ?? 'reviewed_candidate',
    significance_level: rule.risk_level === 'high' ? 'major' : 'supporting',
    person_ids: [],
    people_labels: [],
    place_ids: [],
    place_labels: [],
    organization_ids: [],
    organization_labels: [],
    rag_cluster: rule.confusion_area ?? title,
    related_event_ids: [],
    source_ids: Array.isArray(rule.source_ids) ? rule.source_ids : [],
    canonical: true,
    confusion_area: rule.confusion_area ?? title,
    wrong_or_ambiguous_interpretation: rule.wrong_or_ambiguous_interpretation,
    correct_interpretation: rule.correct_interpretation,
    recommendation_for_RAG: rule.recommendation_for_RAG,
    example_user_questions: rule.example_user_questions ?? [],
    risk_level: rule.risk_level,
  };

  doc.semantic_features = extractSemanticFeatures({
    doc_id: doc.doc_id,
    title: doc.title,
    summary: doc.summary,
    text_for_embedding: doc.text_for_embedding,
    doc_type: doc.doc_type,
    doc_kind: doc.doc_kind,
    doc_source: doc.doc_source,
    event_status: doc.event_status,
    year: doc.year,
    end_year: doc.end_year,
    period_label: doc.period_label,
    verification_status: doc.verification_status,
    significance_level: doc.significance_level,
    source_ids: doc.source_ids,
  });

  return doc;
}

/** Load disambiguation rules as searchable runtime documents. */
function loadDisambiguationRules(): {
  all: DisambiguationRuleDocument[];
  map: Map<string, DisambiguationRuleDocument>;
} {
  const candidatePaths = [
    dataFile('corpus/runtime/disambiguation_rules_indexable.jsonl'),
    dataFile('corpus/disambiguation_rules.jsonl'),
  ];
  const foundPath = candidatePaths.find(p => existsSync(p));
  if (!foundPath) {
    console.warn('⚠️  disambiguation_rules_indexable.jsonl not found — rules lane disabled');
    return { all: [], map: new Map() };
  }

  const rawRules = parseJsonl<RawDisambiguationRule>(foundPath);
  const all = rawRules
    .filter(rule => rule.rule_id)
    .map(rule => normalizeRuleAsDocument(rule));
  const map = new Map<string, DisambiguationRuleDocument>();
  for (const rule of all) map.set(rule.doc_id, rule);

  console.log(`📄 Disambiguation rules loaded: ${all.length} documents from ${foundPath}`);
  return { all, map };
}

/** Load event-entity links and build reverse lookups */
function loadEventEntityLinks(): {
  links: EventEntityLink[];
  entityToEvents: Map<string, string[]>;
} {
  const links = parseJsonl<EventEntityLink>(dataFile('links/event_entity_links.jsonl'));
  const entityToEvents = new Map<string, string[]>();

  for (const link of links) {
    const existing = entityToEvents.get(link.entity_id) || [];
    existing.push(link.record_id);
    entityToEvents.set(link.entity_id, existing);
  }

  console.log(`🔗 Event-entity links loaded: ${links.length} links, ${entityToEvents.size} unique entities`);
  return { links, entityToEvents };
}

/** Load synthesis-event links and build bidirectional lookups */
function loadSynthesisEventLinks(): {
  links: SynthesisEventLink[];
  synthesisToEvents: Map<string, string[]>;
  eventToSynthesis: Map<string, string[]>;
} {
  const links = parseJsonl<SynthesisEventLink>(dataFile('links/synthesis_event_links.jsonl'));
  const synthesisToEvents = new Map<string, string[]>();
  const eventToSynthesis = new Map<string, string[]>();

  for (const link of links) {
    // Synthesis → Events
    const sEvents = synthesisToEvents.get(link.synthesis_id) || [];
    sEvents.push(link.event_id);
    synthesisToEvents.set(link.synthesis_id, sEvents);

    // Event → Synthesis (reverse)
    const eSynths = eventToSynthesis.get(link.event_id) || [];
    eSynths.push(link.synthesis_id);
    eventToSynthesis.set(link.event_id, eSynths);
  }

  console.log(`🔗 Synthesis-event links loaded: ${links.length} links, ${synthesisToEvents.size} synthesis, ${eventToSynthesis.size} events`);
  return { links, synthesisToEvents, eventToSynthesis };
}

/** Load runtime links as provenance map, not as search documents. */
function loadRuntimeProvenanceLinks(): {
  links: RuntimeProvenanceLink[];
  linksByFromDocId: Map<string, RuntimeProvenanceLink[]>;
  linksByToSourceId: Map<string, RuntimeProvenanceLink[]>;
} {
  const candidatePaths = [
    dataFile('corpus/runtime/links_indexable.jsonl'),
    dataFile('corpus/links.jsonl'),
  ];
  const foundPath = candidatePaths.find(p => existsSync(p));
  if (!foundPath) {
    console.warn('⚠️  links_indexable.jsonl not found — runtime provenance links disabled');
    return {
      links: [],
      linksByFromDocId: new Map(),
      linksByToSourceId: new Map(),
    };
  }

  const links = parseJsonl<RuntimeProvenanceLink>(foundPath);
  const linksByFromDocId = new Map<string, RuntimeProvenanceLink[]>();
  const linksByToSourceId = new Map<string, RuntimeProvenanceLink[]>();

  for (const link of links) {
    if (link.from_doc_id) {
      const existing = linksByFromDocId.get(link.from_doc_id) || [];
      existing.push(link);
      linksByFromDocId.set(link.from_doc_id, existing);
    }
    if (link.to_source_id) {
      const existing = linksByToSourceId.get(link.to_source_id) || [];
      existing.push(link);
      linksByToSourceId.set(link.to_source_id, existing);
    }
  }

  console.log(`🔗 Runtime provenance links loaded: ${links.length} links from ${foundPath}`);
  return { links, linksByFromDocId, linksByToSourceId };
}

/** Load evaluation files */
function loadEvaluation(): {
  qaBenchmark: QABenchmarkEntry[];
  retrievalQueries: RetrievalQuery[];
  hardNegatives: HardNegative[];
} {
  const qaBenchmark = parseJsonl<QABenchmarkEntry>(dataFile('evaluation/qa_benchmark.jsonl'));
  const retrievalQueries = parseJsonl<RetrievalQuery>(dataFile('evaluation/retrieval_queries.jsonl'));
  const hardNegatives = parseJsonl<HardNegative>(dataFile('evaluation/hard_negatives.jsonl'));

  console.log(`📊 Evaluation loaded: ${qaBenchmark.length} QA, ${retrievalQueries.length} retrieval queries, ${hardNegatives.length} hard negatives`);
  return { qaBenchmark, retrievalQueries, hardNegatives };
}

// ─── Source Loading ──────────────────────────────────────────

/** Load sources.jsonl for citation provenance. Returns empty Map if file not found. */
function loadSources(): Map<string, SourceDocument> {
  const sources = new Map<string, SourceDocument>();

  // Try multiple possible paths
  const candidatePaths = [
    dataFile('corpus/sources.jsonl'),
    dataFile('sources.jsonl'),
    dataFile('corpus/runtime/sources.jsonl'),
  ];

  const foundPath = candidatePaths.find(p => existsSync(p));
  if (!foundPath) {
    console.warn('⚠️  sources.jsonl not found at any expected path — citations will have no provenance');
    return sources;
  }

  const content = readFileSync(foundPath, 'utf-8');
  const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 0);

  for (let i = 0; i < lines.length; i++) {
    try {
      const raw = JSON.parse(lines[i]) as Record<string, unknown>;

      // Normalize source_id: accept 'source_id', 'id', or generate fallback
      const sourceId = String(raw.source_id ?? raw.id ?? '');
      if (!sourceId) {
        console.warn(`⚠️  sources.jsonl line ${i + 1}: missing source_id/id, skipping`);
        continue;
      }

      // Normalize title: accept 'title', 'source_title', 'book_title', 'name'
      const title = String(
        raw.title ?? raw.source_title ?? raw.book_title ?? raw.name ?? sourceId
      );

      // Normalize year if string
      let pubYear = raw.publication_year ?? raw.year;
      if (typeof pubYear === 'string') {
        const parsed = parseInt(pubYear, 10);
        pubYear = isNaN(parsed) ? undefined : parsed;
      }

      const source: SourceDocument = {
        ...raw,
        source_id: sourceId,
        title,
        publication_year: typeof pubYear === 'number' ? pubYear : undefined,
      };

      sources.set(sourceId, source);
    } catch (e) {
      console.warn(`⚠️  sources.jsonl line ${i + 1}: parse error — ${(e as Error).message}`);
    }
  }

  console.log(`📚 Sources loaded: ${sources.size} sources from ${foundPath}`);
  return sources;
}

// ─── Main Loader ─────────────────────────────────────────────

/**
 * Load the entire dataset into memory.
 *
 * CANONICAL ENFORCEMENT:
 * - Only canonical events are placed in `canonicalEvents` and `allCanonicalDocs`
 * - Duplicate events remain in the `events` map for redirect resolution,
 *   but are never used as primary retrieval documents
 */
export function loadDataset(): LoadedDataset {
  console.log(`\n📦 Loading dataset from: ${config.dataPath}\n`);

  // Load full corpus enrichment map first (Patch 7C-1)
  const enrichMap = loadFullCorpusEnrichmentMap();

  const { canonical: canonicalEvents, map: eventMap } = loadEvents(enrichMap);
  const { all: allSynthesis, map: synthesisMap } = loadSynthesis();
  const { all: canonicalDisambiguationRules, map: disambiguationRuleMap } = loadDisambiguationRules();
  const { links: eventEntityLinks, entityToEvents } = loadEventEntityLinks();
  const { links: synthesisEventLinks, synthesisToEvents, eventToSynthesis } = loadSynthesisEventLinks();
  const { links: runtimeLinks, linksByFromDocId, linksByToSourceId } = loadRuntimeProvenanceLinks();
  const { qaBenchmark, retrievalQueries, hardNegatives } = loadEvaluation();
  const sources = loadSources();

  // Canonical synthesis = all synthesis docs unless explicitly marked non-canonical.
  // Data-pack synthesis records may omit `canonical`; absence should not remove them from retrieval.
  const canonicalSynthesis = allSynthesis.filter(s => s.canonical !== false);

  // Combined canonical documents for unified indexing
  const allCanonicalDocs: IndexableDocument[] = [
    ...canonicalEvents,
    ...canonicalSynthesis,
    ...canonicalDisambiguationRules,
  ];

  console.log(`\n✅ Dataset loaded successfully:`);
  console.log(`   Canonical docs for indexing: ${allCanonicalDocs.length}`);
  console.log(`   (${canonicalEvents.length} events + ${canonicalSynthesis.length} synthesis + ${canonicalDisambiguationRules.length} rules)`);
  console.log(`   Sources: ${sources.size}\n`);

  return {
    events: eventMap,
    synthesis: synthesisMap,
    disambiguationRules: disambiguationRuleMap,
    canonicalEvents,
    canonicalSynthesis,
    canonicalDisambiguationRules,
    allCanonicalDocs,
    eventEntityLinks,
    synthesisEventLinks,
    qaBenchmark,
    retrievalQueries,
    hardNegatives,
    sources,
    runtimeLinks,
    linksByFromDocId,
    linksByToSourceId,
    entityToEvents,
    synthesisToEvents,
    eventToSynthesis,
  };
}
