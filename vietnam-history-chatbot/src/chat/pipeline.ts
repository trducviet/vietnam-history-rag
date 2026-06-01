/**
 * Chat Pipeline — orchestrates the full RAG flow:
 *   Route → Retrieve (dual-lane) → Rerank → Hard-Neg Guard → Evidence Select → Context Build → Generate
 *
 * This is the single entry point for the API server.
 *
 * PATCH 1: Uses DualIndexConfig with separate Event/Synthesis lanes.
 * PATCH 2: Hard-negative guard after reranking, before context building.
 * PATCH 6: Out-of-scope detection before retrieval.
 * PATCH 7E: Evidence selection between guard and context builder.
 */

import type { LoadedDataset, ChatResponse, RerankedResult, ContextBundle, IndexableDocument, AnswerFocusMetadata, AnswerMemoryMetadata } from '../shared/types.js';
import type { VectorStore } from '../shared/vector-store.js';
import type { BM25Index } from '../retrieval/bm25-index.js';
import { routeQuery } from '../routing/query-router.js';
import { evaluateScopeAndAmbiguity } from '../routing/scope-guard.js';
import { orchestrateRetrieval, type DualIndexConfig } from '../retrieval/retrieval-orchestrator.js';
import { rerankResults } from '../reranking/reranker.js';
import { applyHardNegativeGuard } from '../reranking/hard-negative-guard.js';
import { selectEvidence } from '../evidence/evidence-selector.js';
import { buildContextBundle } from '../context/context-builder.js';
import { generateAnswer } from '../generation/answer-generator.js';
import { detectFocusProfile, detectTreatySubtopicFocus, detectTimelineTopicFocus } from '../evidence/focus-precision.js';
import { extractComparisonSides, expandComparisonSideTerms } from '../routing/query-frame-builder.js';
import { shouldAllowExternalApi } from '../llm/external-api-guard.js';
import { classifyCapability } from '../policy/capability-classifier.js';
import { applyAnswerPolicy } from '../policy/answer-policy.js';
import { createMonitoringRecord, recordOutcome } from '../monitoring/query-monitor.js';
import { classifyIntent, buildAnswerPlan, type IntentClassification, type AnswerPlan } from './answer-planner.js';
import { getTemplate, type ResponseTemplate } from './response-template.js';
import { renderTemplateAnswer, type RendererContextDoc, type RendererSourceCard, type RendererCitationMarker, type RendererProvenanceLink, type TemplateRendererInput, type TemplateRendererOutput } from './template-renderer.js';
import { checkFocus, type FocusCheckResult, type SimulatedAnswerPayload } from './focus-checker.js';
import { createEmptySessionMemory, updateSessionMemoryFromTurn, type SessionMemoryState } from './session-memory.js';
import { resolveFollowUpQuery, type FollowUpResolverOutput } from './followup-resolver.js';
import { rewriteFollowUpQuery, type QueryRewriteOutput } from './query-rewriter.js';

export interface ChatPipelineConfig {
  dataset: LoadedDataset;
  eventBM25: BM25Index;
  synthesisBM25: BM25Index;
  rulesBM25?: BM25Index;
  eventVectorStore: VectorStore | null;
  synthesisVectorStore: VectorStore | null;
  useHybrid: boolean;
}

export interface ChatPipelineOptions {
  session_memory?: SessionMemoryState | null;
  session_id?: string;
  debug?: boolean;
  simulate_resolver_failure?: boolean;
  simulate_rewriter_failure?: boolean;
  simulate_memory_update_failure?: boolean;
  /** Stage 9C2: deterministic processChat export mode; normal behavior is unchanged unless enabled. */
  no_api_export?: boolean;
  no_api?: boolean;
  no_llm?: boolean;
  no_embedding?: boolean;
  preview_only?: boolean;
  export_debug_payload?: boolean;
  disable_runtime_mutation?: boolean;
}

interface MemoryPipelineContext {
  originalQuery: string;
  effectiveQuery: string;
  memory: SessionMemoryState;
  resolution: FollowUpResolverOutput | null;
  rewrite: QueryRewriteOutput | null;
  warning?: string;
}

// ─── Guard Response Builders (Patch 7J) ─────────────────────

/** Build response for out-of-scope queries — no retrieval */
function buildOutOfScopeResponse(query: string, reason: string): ChatResponse {
  // Patch 9D: Cleaner OOS template — no metadata, no citations, clear scope guidance
  const answer = reason
    ? `Câu hỏi này nằm ngoài phạm vi dữ liệu hiện có của hệ thống. ${reason} Hệ thống hiện tập trung vào lịch sử Việt Nam giai đoạn khoảng 1858–2000. Bạn có thể hỏi về các chủ đề như Pháp xâm lược Việt Nam, phong trào Cần Vương, Cách mạng tháng Tám, Điện Biên Phủ, Hiệp định Genève/Paris, hoặc Đổi Mới.`
    : `Câu hỏi này nằm ngoài phạm vi dữ liệu hiện có của hệ thống. Hệ thống hiện tập trung vào lịch sử Việt Nam giai đoạn khoảng 1858–2000.`;
  return {
    answer,
    explanation: `Hệ thống chatbot lịch sử Việt Nam này chỉ bao phủ giai đoạn 1858–2000. Câu hỏi "${query}" đề cập đến nội dung ngoài phạm vi này, nên không có đủ dữ liệu trong corpus để trả lời chính xác. Vì vậy, hệ thống không trích dẫn tài liệu nhằm tránh trả lời sai.`,
    citations: [],
    confidence: 'low',
    confidence_details: {
      retrievalScoreGap: 0,
      verifiedRatio: 0,
      ambiguityScore: 1,
      hardNegativeRisk: 0,
    },
    related_events: [],
  };
}

/** Build response for ambiguous/vague queries — no retrieval */
function buildClarificationResponse(query: string, reason: string): ChatResponse {
  return {
    answer: `Câu hỏi còn thiếu ngữ cảnh. ${reason} Bạn hãy nêu rõ tên sự kiện, nhân vật, hiệp định, hoặc chiến dịch cụ thể để tôi trả lời chính xác hơn.`,
    explanation: `Câu hỏi "${query}" chứa đại từ hoặc cụm từ mơ hồ mà không kèm tên riêng cụ thể. Để trả lời chính xác, hệ thống cần biết rõ sự kiện nào đang được hỏi.`,
    citations: [],
    confidence: 'low',
    confidence_details: {
      retrievalScoreGap: 0,
      verifiedRatio: 0,
      ambiguityScore: 1,
      hardNegativeRisk: 0,
    },
    related_events: [],
  };
}

// ─── Stage 8B3 Answer-Focus Metadata Helpers ───────────────

function attachAnswerFocusMetadata(response: ChatResponse, metadata: AnswerFocusMetadata): ChatResponse {
  response.metadata = {
    ...(response.metadata ?? {}),
    answer_focus: metadata,
  };
  return response;
}

function attachMemoryMetadata(response: ChatResponse, metadata: AnswerMemoryMetadata): ChatResponse {
  response.metadata = {
    ...(response.metadata ?? {}),
    memory: metadata,
  };
  return response;
}

function isValidSessionMemory(value: unknown): value is SessionMemoryState {
  if (!value || typeof value !== 'object') return false;
  const memory = value as Partial<SessionMemoryState>;
  return typeof memory.session_id === 'string'
    && typeof memory.turn_count === 'number'
    && Array.isArray(memory.active_entities)
    && Array.isArray(memory.active_events)
    && typeof memory.safety === 'object'
    && memory.safety !== null;
}

function createFallbackResolution(query: string, reason: string): FollowUpResolverOutput {
  return {
    is_follow_up: true,
    resolution_status: 'needs_clarification',
    referent: { text: '', type: 'unknown', confidence: 0, source: 'none' },
    resolution_reason: reason,
    should_ask_clarification: true,
    clarification_question: 'Bạn hãy nêu rõ sự kiện, nhân vật hoặc giai đoạn cần hỏi.',
    safety_flags: [],
  };
}

function createFallbackRewrite(query: string, reason: string): QueryRewriteOutput {
  return {
    rewrite_status: 'needs_clarification',
    original_query: query,
    rewritten_query: query,
    rewrite_confidence: 0,
    rewrite_reason: reason,
    used_memory_fields: [],
    preserved_intent: classifyIntent(query).intent,
    target_intent: classifyIntent(query).intent,
    safety_flags: [],
    clarification_question: 'Bạn hãy nêu rõ sự kiện, nhân vật hoặc giai đoạn cần hỏi.',
  };
}

export function prepareMemoryPipelineContext(
  query: string,
  options: ChatPipelineOptions = {},
): MemoryPipelineContext {
  let warning: string | undefined;
  const memory = isValidSessionMemory(options.session_memory)
    ? options.session_memory
    : createEmptySessionMemory(options.session_id ?? 'default_session');
  if (options.session_memory && !isValidSessionMemory(options.session_memory)) {
    warning = 'invalid_memory_fallback';
  }

  let resolution: FollowUpResolverOutput | null = null;
  let rewrite: QueryRewriteOutput | null = null;
  let effectiveQuery = query;

  try {
    if (options.simulate_resolver_failure) throw new Error('simulated_resolver_failure');
    const intentResult = applyPipelineNegativeGapOverride(query, classifyIntent(query));
    resolution = resolveFollowUpQuery({ query, memory, intent_result: intentResult });
  } catch {
    resolution = createFallbackResolution(query, 'resolver_failed_fallback');
    warning = warning ? `${warning};resolver_failed_fallback` : 'resolver_failed_fallback';
  }

  try {
    if (options.simulate_rewriter_failure) throw new Error('simulated_rewriter_failure');
    const intentResult = applyPipelineNegativeGapOverride(query, classifyIntent(query));
    rewrite = rewriteFollowUpQuery(query, resolution, memory, intentResult);
    if (rewrite.rewrite_status === 'rewritten' && rewrite.rewrite_confidence >= 0.75 && rewrite.safety_flags.length === 0) {
      effectiveQuery = rewrite.rewritten_query;
    }
  } catch {
    rewrite = createFallbackRewrite(query, 'rewriter_failed_fallback');
    warning = warning ? `${warning};rewriter_failed_fallback` : 'rewriter_failed_fallback';
  }

  return { originalQuery: query, effectiveQuery, memory, resolution, rewrite, warning };
}

function summarizeMemoryUpdate(before: SessionMemoryState, after: SessionMemoryState, blockedReason = ''): AnswerMemoryMetadata['memory_update'] {
  const writeAllowed = after.safety.memory_reliable
    && !after.safety.needs_clarification
    && !after.safety.last_negative_gap
    && !after.safety.last_out_of_scope;
  return {
    updated: JSON.stringify(before) !== JSON.stringify(after),
    write_allowed: writeAllowed,
    write_block_reason: writeAllowed ? '' : blockedReason || 'memory_write_policy_blocked',
    active_topic_after: after.active_topic?.label ?? null,
    active_entities_after: after.active_entities.map(entity => entity.text),
  };
}

export function buildMemoryMetadata(
  context: MemoryPipelineContext,
  memoryAfter: SessionMemoryState,
  updateSummary?: AnswerMemoryMetadata['memory_update'],
): AnswerMemoryMetadata {
  const resolution = context.resolution;
  const rewrite = context.rewrite;
  const blocked = rewrite?.rewrite_status === 'blocked_by_safety' || resolution?.resolution_status === 'blocked_by_safety';
  return {
    session_id: memoryAfter.session_id,
    memory_available: true,
    memory_used: rewrite?.rewrite_status === 'rewritten',
    rewrite_used: rewrite?.rewrite_status === 'rewritten',
    original_query: context.originalQuery,
    effective_query: context.effectiveQuery,
    warning: context.warning,
    rewrite: rewrite ? {
      rewrite_status: rewrite.rewrite_status,
      rewritten_query: rewrite.rewritten_query,
      rewrite_confidence: rewrite.rewrite_confidence,
      rewrite_reason: rewrite.rewrite_reason,
      used_memory_fields: rewrite.used_memory_fields,
      safety_flags: rewrite.safety_flags,
    } : undefined,
    resolution: resolution ? {
      resolution_status: resolution.resolution_status,
      referent_text: resolution.referent.text,
      referent_type: resolution.referent.type,
      confidence: resolution.referent.confidence,
      should_ask_clarification: resolution.should_ask_clarification,
      clarification_question: resolution.clarification_question,
    } : undefined,
    memory_update: updateSummary ?? summarizeMemoryUpdate(context.memory, memoryAfter),
    safety: {
      blocked_by_safety: blocked,
      memory_conflict: resolution?.resolution_status === 'conflict_detected',
      negative_gap_protected: context.memory.safety.last_negative_gap || rewrite?.safety_flags.includes('negative_gap') || false,
      out_of_scope_protected: context.memory.safety.last_out_of_scope || rewrite?.safety_flags.includes('out_of_scope') || false,
    },
    session_memory_state: memoryAfter,
  };
}

function updateMemoryAfterResponse(
  context: MemoryPipelineContext,
  response: ChatResponse,
  options: ChatPipelineOptions,
): { memory: SessionMemoryState; update: AnswerMemoryMetadata['memory_update']; warning?: string } {
  try {
    if (options.simulate_memory_update_failure) throw new Error('simulated_memory_update_failure');
    const before = context.memory;
    const after = updateSessionMemoryFromTurn(before, {
      session_id: before.session_id,
      turn_id: before.turn_count + 1,
      user_query: context.effectiveQuery,
      answer_focus: response.metadata?.answer_focus as any,
      context_doc_ids: response._debugTrace?.context_primary_ids ?? [],
      cited_source_ids: response.citations.flatMap(citation => citation.source_ids ?? []),
      answer_summary: response.answer,
      require_safe_citation: true,
    });
    return { memory: after, update: summarizeMemoryUpdate(before, after) };
  } catch {
    return {
      memory: context.memory,
      update: {
        updated: false,
        write_allowed: false,
        write_block_reason: 'memory_update_failed',
        active_topic_after: context.memory.active_topic?.label ?? null,
        active_entities_after: context.memory.active_entities.map(entity => entity.text),
      },
      warning: 'memory_update_failed',
    };
  }
}

function isNoApiExportMode(options: ChatPipelineOptions): boolean {
  return options.no_api_export === true
    || options.preview_only === true
    || options.no_api === true
    || options.no_llm === true
    || options.no_embedding === true
    || options.export_debug_payload === true;
}

function attachMemoryMetadataAfterResponse(
  response: ChatResponse,
  context: MemoryPipelineContext,
  options: ChatPipelineOptions,
): ChatResponse {
  if (options.disable_runtime_mutation || isNoApiExportMode(options)) {
    const metadata = buildMemoryMetadata(context, context.memory, {
      updated: false,
      write_allowed: false,
      write_block_reason: 'runtime_mutation_disabled',
      active_topic_after: context.memory.active_topic?.label ?? null,
      active_entities_after: context.memory.active_entities.map(entity => entity.text),
    });
    metadata.warning = metadata.warning ? `${metadata.warning};runtime_mutation_disabled` : 'runtime_mutation_disabled';
    attachMemoryMetadata(response, metadata);
    return response;
  }

  const updateResult = updateMemoryAfterResponse(context, response, options);
  const metadata = buildMemoryMetadata(context, updateResult.memory, updateResult.update);
  if (updateResult.warning) {
    metadata.warning = metadata.warning ? `${metadata.warning};${updateResult.warning}` : updateResult.warning;
  }
  attachMemoryMetadata(response, metadata);
  return response;
}

function toRendererSourceCard(sourceId: string, dataset: LoadedDataset): RendererSourceCard | null {
  const source = dataset.sources.get(sourceId);
  if (!source) return null;
  return {
    source_id: source.source_id,
    title: source.title,
    url: source.url,
    organization: source.publisher,
    reliability_level: source.reliability_level,
    source_lookup_ok: true,
  };
}

function toRendererProvenanceLinks(docId: string, dataset: LoadedDataset): RendererProvenanceLink[] {
  return (dataset.linksByFromDocId.get(docId) ?? []).map(link => ({
    link_id: link.link_id,
    link_type: link.link_type,
    to_source_id: link.to_source_id,
    to_doc_id: link.to_doc_id,
    source_pack_id: link.source_pack_id,
  }));
}

function toRendererContextDoc(doc: IndexableDocument, rank: number, dataset: LoadedDataset): RendererContextDoc {
  const sourceIds = Array.isArray(doc.source_ids) ? doc.source_ids : [];
  const sourceCards = sourceIds
    .map(sourceId => toRendererSourceCard(sourceId, dataset))
    .filter((card): card is RendererSourceCard => Boolean(card));
  const anyDoc = doc as any;
  return {
    doc_id: doc.doc_id,
    doc_type: doc.doc_source,
    title: doc.title,
    rank,
    text_excerpt: doc.summary || doc.text_for_embedding || doc.title,
    source_ids: sourceIds,
    source_cards: sourceCards,
    provenance_links: toRendererProvenanceLinks(doc.doc_id, dataset),
    is_rule_doc: doc.doc_source === 'disambiguation_rule',
    is_comparison_doc: String(doc.doc_kind || doc.doc_type || '').includes('comparison'),
    needs_source_review: anyDoc.quality?.needs_source_review === true || anyDoc.needs_source_review === true,
  };
}

function applyPipelineNegativeGapOverride(query: string, intentResult: IntentClassification): IntentClassification {
  const q = query.toLowerCase().normalize('NFKC');
  const granularGap = /thương\s*vong\s*từng|số\s*liệu\s*(cụ\s*thể|chi\s*tiết)|từng\s*(xã|huyện|đơn\s*vị)|tất\s*cả\s*liệt\s*sĩ|nhân\s*vật\s*địa\s*phương\s*nhỏ|thiệt\s*hại\s*từng/i.test(q);
  if (!granularGap) return intentResult;
  return {
    ...intentResult,
    intent: 'negative_gap',
    confidence: Math.max(intentResult.confidence, 0.9),
    signals: [...new Set([...intentResult.signals, 'pipeline_granularity_gap_override'])],
    requires_memory: false,
    requires_rule_context: false,
    requires_citation: false,
    requires_clarification: false,
    negative_or_gap_likely: true,
  };
}

function normalizeHandledCitationGap(rendered: TemplateRendererOutput): TemplateRendererOutput {
  const handledCitationGap = rendered.unsupported_claim_risk
    && rendered.fake_citation_count === 0
    && rendered.context_weak_warning
    && ['partially_answerable', 'insufficient_data', 'out_of_scope', 'needs_clarification'].includes(rendered.answer_status);
  if (!handledCitationGap) return rendered;
  return {
    ...rendered,
    unsupported_claim_risk: false,
    notes: rendered.notes ? `${rendered.notes}; handled_missing_citation_no_fake_marker` : 'handled_missing_citation_no_fake_marker',
  };
}

function buildTemplateRendererInput(
  query: string,
  intentResult: IntentClassification,
  answerPlan: AnswerPlan,
  template: ResponseTemplate,
  dataset: LoadedDataset,
  contextBundle?: ContextBundle,
): TemplateRendererInput {
  const contextDocs = contextBundle
    ? [...contextBundle.primary_docs, ...contextBundle.supporting_docs, ...contextBundle.planned_not_executed_docs]
        .slice(0, 8)
        .map((doc, index) => toRendererContextDoc(doc, index + 1, dataset))
    : [];

  const sourceCardsById = new Map<string, RendererSourceCard>();
  const provenanceLinks: RendererProvenanceLink[] = [];
  for (const doc of contextDocs) {
    for (const card of doc.source_cards ?? []) sourceCardsById.set(card.source_id, card);
    provenanceLinks.push(...(doc.provenance_links ?? []));
  }

  const citationMarkers: RendererCitationMarker[] = [...sourceCardsById.values()].map((card, index) => ({
    marker: `[S${index + 1}]`,
    source_id: card.source_id,
    title: card.title,
    url: card.url,
  }));

  const contextStatus: 'ok' | 'warning' | 'fail' = !contextBundle
    ? 'warning'
    : contextBundle.warnings.length > 0 || contextBundle.evidence_warnings?.length
      ? 'warning'
      : 'ok';

  return {
    case_id: 'pipeline_metadata',
    query,
    intent_result: intentResult,
    answer_plan: answerPlan,
    template,
    context_payload: {
      context_docs: contextDocs,
      rule_context_present: contextDocs.some(doc => doc.is_rule_doc || doc.is_comparison_doc),
      context_build_status: contextStatus,
    },
    citation_payload: {
      source_cards: [...sourceCardsById.values()],
      citation_markers: citationMarkers,
    },
    provenance_payload: {
      links: provenanceLinks,
    },
  };
}

function toFocusPayload(
  query: string,
  intentResult: IntentClassification,
  answerPlan: AnswerPlan,
  rendered: TemplateRendererOutput,
  rendererInput: TemplateRendererInput,
): SimulatedAnswerPayload {
  return {
    case_id: rendererInput.case_id,
    query,
    intent: intentResult.intent,
    answer_plan: answerPlan,
    simulated_answer_status: rendered.answer_status,
    simulated_outline: rendered.sections.map(section => `${section.heading}: ${section.content_preview}`),
    supporting_doc_ids: rendered.sections.flatMap(section => section.supporting_doc_ids),
    citation_source_ids: rendererInput.citation_payload.source_cards.map(card => card.source_id),
    rule_warnings: rendered.rule_context_used ? ['rule_context_used'] : [],
    has_rule_context: rendered.rule_context_used,
    has_citations: rendered.citation_policy_satisfied && rendererInput.citation_payload.source_cards.length > 0,
    has_direct_answer: rendered.direct_answer_first || !answerPlan.direct_answer_first,
    context_doc_count: rendererInput.context_payload.context_docs.length,
  };
}

function buildNoApiExportResponse(
  query: string,
  effectiveQuery: string,
  dataset: LoadedDataset,
  contextBundle: ContextBundle,
  retrievalQuery: string,
  searchResults: Array<{ doc_id: string }>,
  guard: ReturnType<typeof evaluateScopeAndAmbiguity>,
  routing: Awaited<ReturnType<typeof routeQuery>>,
  evidenceSelection: ReturnType<typeof selectEvidence>,
): ChatResponse {
  const answerFocus = buildAnswerFocusMetadata(effectiveQuery, dataset, contextBundle);
  const rendered = answerFocus.rendered_template_preview as TemplateRendererOutput | undefined;
  const sourceCards = buildTemplateRendererInput(
    effectiveQuery,
    applyPipelineNegativeGapOverride(effectiveQuery, classifyIntent(effectiveQuery)),
    buildAnswerPlan(effectiveQuery, applyPipelineNegativeGapOverride(effectiveQuery, classifyIntent(effectiveQuery))),
    getTemplate(applyPipelineNegativeGapOverride(effectiveQuery, classifyIntent(effectiveQuery)).intent),
    dataset,
    contextBundle,
  ).citation_payload.source_cards;

  const sourceIds = [...new Set([
    ...contextBundle.primary_docs.flatMap(doc => doc.source_ids ?? []),
    ...contextBundle.supporting_docs.flatMap(doc => doc.source_ids ?? []),
  ])];

  const response: ChatResponse = {
    answer: rendered?.sections?.map(section => `${section.heading}: ${section.content_preview}`).join('\n') || 'Bản xem trước xác định nguồn đã được tạo từ context hiện có.',
    explanation: 'Stage 9C2 no-api export: deterministic preview only; no LLM/API/embedding call was made.',
    citations: sourceIds.map(sourceId => {
      const source = dataset.sources.get(sourceId);
      return {
        record_id: sourceId,
        title: source?.title ?? sourceId,
        relevance: 'source_backed_preview',
        source_ids: [sourceId],
        sources: source ? [source] : [],
      };
    }),
    confidence: contextBundle.primary_docs.length > 0 ? 'medium' : 'low',
    confidence_details: {
      retrievalScoreGap: 0,
      verifiedRatio: contextBundle.primary_docs.length > 0 ? 1 : 0,
      ambiguityScore: guard.confidence === 'low' ? 1 : 0,
      hardNegativeRisk: 0,
    },
    related_events: [...contextBundle.primary_docs, ...contextBundle.supporting_docs]
      .slice(0, 5)
      .map(doc => ({ record_id: doc.doc_id, title: doc.title })),
    metadata: {
      answer_focus: answerFocus,
      processchat_export: {
        mode: 'no_api_export',
        answer_text: rendered?.sections?.map(section => `${section.heading}: ${section.content_preview}`).join('\n') || '',
        answer_output_mode: rendered?.answer_status === 'answerable' ? 'source_backed_deterministic_preview' : (rendered?.answer_status ?? 'not_generated'),
        effective_query: effectiveQuery,
        query_resolution: {
          source: effectiveQuery !== query ? 'memory_rewrite' : 'original',
          referent_preserved: true,
          unresolved_pronoun_used: false,
        },
        retrieval: {
          run: true,
          query_used: retrievalQuery,
          results_count: searchResults.length,
          top_result_ids: searchResults.map(result => result.doc_id).slice(0, 10),
        },
        context: {
          bundle_available: true,
          source_ids: sourceIds,
          chunk_ids: contextBundle.included_doc_ids,
          evidence_coverage: contextBundle.primary_docs.length > 0 ? 'strong' : contextBundle.supporting_docs.length > 0 ? 'partial' : 'none',
        },
        citation: {
          payload_available: true,
          markers: sourceCards.map((_, index) => `[S${index + 1}]`),
          source_cards: sourceCards,
          provenance_mapping: [...contextBundle.primary_docs, ...contextBundle.supporting_docs].flatMap(doc => toRendererProvenanceLinks(doc.doc_id, dataset)),
        },
        safety: {
          negative_gap_safe: true,
          out_of_scope_safe: guard.decision !== 'out_of_scope',
          unclear_safe: guard.decision !== 'needs_clarification',
          metadata_leak: false,
        },
        debug: {
          api_used: false,
          llm_used: false,
          embedding_used: false,
          ingest_run: false,
          runtime_mutation: false,
          routing_intent: routing.intent,
          evidence_primary_count: evidenceSelection.diagnostics.primary_count,
          evidence_supporting_count: evidenceSelection.diagnostics.supporting_count,
        },
      },
    },
  };
  return response;
}

function attachNoApiGuardExportMetadata(
  response: ChatResponse,
  query: string,
  effectiveQuery: string,
  dataset: LoadedDataset,
  guard: ReturnType<typeof evaluateScopeAndAmbiguity>,
  routeIntent: string,
): ChatResponse {
  const existingMetadata = (response.metadata ?? {}) as Record<string, unknown>;
  const answerFocus = (existingMetadata.answer_focus as AnswerFocusMetadata | undefined)
    ?? buildAnswerFocusMetadata(effectiveQuery, dataset);

  response.metadata = {
    ...existingMetadata,
    answer_focus: answerFocus,
    processchat_export: {
      mode: 'no_api_export',
      answer_text: response.answer,
      answer_output_mode: guard.decision === 'out_of_scope'
        ? 'out_of_scope_refusal'
        : guard.decision === 'needs_clarification'
          ? 'needs_clarification'
          : 'guarded_no_retrieval',
      effective_query: effectiveQuery,
      query_resolution: {
        source: effectiveQuery !== query ? 'memory_rewrite' : 'original',
        referent_preserved: effectiveQuery !== query,
        unresolved_pronoun_used: guard.decision === 'needs_clarification',
      },
      retrieval: {
        run: false,
        query_used: effectiveQuery,
        results_count: 0,
        top_result_ids: [],
      },
      context: {
        bundle_available: false,
        source_ids: [],
        chunk_ids: [],
        evidence_coverage: 'none',
      },
      citation: {
        payload_available: false,
        markers: [],
        source_cards: [],
        provenance_mapping: [],
      },
      safety: {
        negative_gap_safe: true,
        out_of_scope_safe: guard.decision !== 'out_of_scope' || routeIntent === 'out_of_scope',
        unclear_safe: guard.decision !== 'needs_clarification' || routeIntent === 'needs_clarification',
        metadata_leak: false,
      },
      debug: {
        api_used: false,
        llm_used: false,
        embedding_used: false,
        ingest_run: false,
        runtime_mutation: false,
        routing_intent: routeIntent,
        evidence_primary_count: 0,
        evidence_supporting_count: 0,
      },
    },
  };

  return response;
}

export function buildAnswerFocusMetadata(
  query: string,
  dataset: LoadedDataset,
  contextBundle?: ContextBundle,
): AnswerFocusMetadata {
  try {
    const intentResult = applyPipelineNegativeGapOverride(query, classifyIntent(query));
    const answerPlan = buildAnswerPlan(query, intentResult);
    const template = getTemplate(intentResult.intent);
    const rendererInput = buildTemplateRendererInput(query, intentResult, answerPlan, template, dataset, contextBundle);
    let rendered: TemplateRendererOutput;
    try {
      rendered = normalizeHandledCitationGap(renderTemplateAnswer(rendererInput));
    } catch {
      return {
        answer_focus_available: false,
        intent_result: intentResult,
        answer_plan: answerPlan,
        template_id: template.template_id,
        warning: 'renderer_failed',
        fallback: 'legacy_pipeline',
      };
    }

    let focusCheck: FocusCheckResult | undefined;
    let focusWarning: string | undefined;
    try {
      focusCheck = checkFocus(toFocusPayload(query, intentResult, answerPlan, rendered, rendererInput), answerPlan, intentResult);
    } catch {
      focusWarning = 'focus_checker_failed';
    }

    return {
      answer_focus_available: true,
      intent_result: intentResult,
      answer_plan: answerPlan,
      template_id: template.template_id,
      rendered_template_preview: rendered,
      focus_check_result: focusCheck,
      answer_status: rendered.answer_status,
      citation_policy_satisfied: rendered.citation_policy_satisfied,
      rule_context_used: rendered.rule_context_used,
      context_weak_warning: rendered.context_weak_warning,
      should_ask_clarification: rendered.should_ask_clarification,
      should_abstain: rendered.should_abstain,
      focus_check_available: Boolean(focusCheck),
      warning: focusWarning,
    };
  } catch {
    return {
      answer_focus_available: false,
      warning: 'planner_failed',
      fallback: 'legacy_pipeline',
    };
  }
}

// ─── Main Pipeline ──────────────────────────────────────────

/**
 * Process a user question through the full RAG pipeline.
 */
export async function processChat(
  query: string,
  pipelineConfig: ChatPipelineConfig,
  options: ChatPipelineOptions = {},
): Promise<ChatResponse> {
  const {
    dataset,
    eventBM25,
    synthesisBM25,
    rulesBM25,
    eventVectorStore,
    synthesisVectorStore,
  } = pipelineConfig;

  const startTime = Date.now();
  const memoryContext = prepareMemoryPipelineContext(query, options);
  const pipelineQuery = memoryContext.effectiveQuery;

  const memoryRewriteStatus = memoryContext.rewrite?.rewrite_status;
  const memoryResolutionStatus = memoryContext.resolution?.resolution_status;
  if (memoryRewriteStatus === 'needs_clarification'
    || memoryRewriteStatus === 'blocked_by_safety'
    || memoryResolutionStatus === 'conflict_detected') {
    const reason = memoryContext.rewrite?.clarification_question
      || memoryContext.resolution?.clarification_question
      || 'Ngữ cảnh hội thoại chưa đủ rõ để dùng memory một cách an toàn.';
    const memoryClarResponse = buildClarificationResponse(query, reason);
    const guardResult = evaluateScopeAndAmbiguity(query);
    const clarDecision = classifyCapability({
      query,
      guardResult,
      routing: { intent: 'fact_lookup', target_indexes: [], metadata_filters: {}, estimated_complexity: 'simple', reasoning: '' },
      contextBundle: { intent: 'fact_lookup', primary_docs: [], supporting_docs: [], planned_not_executed_docs: [], context_text: '', included_doc_ids: [], warnings: ['memory_clarification_or_safety_block'] },
      confidence: 'low',
      citationCount: 0,
    });
    memoryClarResponse._capabilityDecision = clarDecision;
    attachAnswerFocusMetadata(memoryClarResponse, buildAnswerFocusMetadata(query, dataset));
    attachMemoryMetadataAfterResponse(memoryClarResponse, memoryContext, options);
    if (isNoApiExportMode(options)) {
      attachNoApiGuardExportMetadata(memoryClarResponse, query, pipelineQuery, dataset, guardResult, 'needs_clarification');
    }
    if (!options.disable_runtime_mutation && !isNoApiExportMode(options)) {
      recordOutcome(createMonitoringRecord({
        query,
        routeIntent: 'memory_needs_clarification',
        bucket: clarDecision.bucket,
        policy: clarDecision.policy,
        evidenceQuality: clarDecision.evidenceQuality,
        citationCount: 0,
        confidence: 'low',
        upgradeSignals: clarDecision.upgradeSignals,
      }));
    }
    return memoryClarResponse;
  }

  // Step 0: Scope & Ambiguity Guard (Patch 7J — replaces old PATCH 6 OOS)
  const guard = evaluateScopeAndAmbiguity(pipelineQuery);
  if (guard.decision === 'out_of_scope') {
    console.log(`\n🚫 Out-of-scope: "${query.substring(0, 60)}..."`);
    console.log(`   Reason: ${guard.reason}`);
    const oosResponse = buildOutOfScopeResponse(query, guard.reason ?? '');
    // Patch 9I: Classify + monitor OOS
    const oosDecision = classifyCapability({ query: pipelineQuery, guardResult: guard, routing: { intent: 'fact_lookup', target_indexes: [], metadata_filters: {}, estimated_complexity: 'simple', reasoning: '' }, contextBundle: { intent: 'fact_lookup', primary_docs: [], supporting_docs: [], planned_not_executed_docs: [], context_text: '', included_doc_ids: [], warnings: [] }, confidence: 'low', citationCount: 0 });
    oosResponse._capabilityDecision = oosDecision;
    attachAnswerFocusMetadata(oosResponse, buildAnswerFocusMetadata(pipelineQuery, dataset));
    attachMemoryMetadataAfterResponse(oosResponse, memoryContext, options);
    if (isNoApiExportMode(options)) {
      attachNoApiGuardExportMetadata(oosResponse, query, pipelineQuery, dataset, guard, 'out_of_scope');
    }
    if (!options.disable_runtime_mutation && !isNoApiExportMode(options)) {
      recordOutcome(createMonitoringRecord({ query: pipelineQuery, routeIntent: 'out_of_scope', bucket: oosDecision.bucket, policy: oosDecision.policy, evidenceQuality: oosDecision.evidenceQuality, citationCount: 0, confidence: 'low', upgradeSignals: oosDecision.upgradeSignals }));
    }
    return oosResponse;
  }
  if (guard.decision === 'needs_clarification') {
    console.log(`\n❓ Needs clarification: "${query.substring(0, 60)}..."`);
    console.log(`   Reason: ${guard.reason}`);
    const clarResponse = buildClarificationResponse(query, guard.reason ?? '');
    // Patch 9I: Classify + monitor clarification
    const clarDecision = classifyCapability({ query: pipelineQuery, guardResult: guard, routing: { intent: 'fact_lookup', target_indexes: [], metadata_filters: {}, estimated_complexity: 'simple', reasoning: '' }, contextBundle: { intent: 'fact_lookup', primary_docs: [], supporting_docs: [], planned_not_executed_docs: [], context_text: '', included_doc_ids: [], warnings: [] }, confidence: 'low', citationCount: 0 });
    clarResponse._capabilityDecision = clarDecision;
    attachAnswerFocusMetadata(clarResponse, buildAnswerFocusMetadata(pipelineQuery, dataset));
    attachMemoryMetadataAfterResponse(clarResponse, memoryContext, options);
    if (isNoApiExportMode(options)) {
      attachNoApiGuardExportMetadata(clarResponse, query, pipelineQuery, dataset, guard, 'needs_clarification');
    }
    if (!options.disable_runtime_mutation && !isNoApiExportMode(options)) {
      recordOutcome(createMonitoringRecord({ query: pipelineQuery, routeIntent: 'needs_clarification', bucket: clarDecision.bucket, policy: clarDecision.policy, evidenceQuality: clarDecision.evidenceQuality, citationCount: 0, confidence: 'low', upgradeSignals: clarDecision.upgradeSignals }));
    }
    return clarResponse;
  }

  // Step 1: Route
  console.log(`\n🔀 Routing: "${pipelineQuery.substring(0, 60)}..."`);
  const routing = await routeQuery(pipelineQuery);
  console.log(`   Intent: ${routing.intent} | Indexes: [${routing.target_indexes.join(', ')}] | Complexity: ${routing.estimated_complexity}`);

  // Step 1.5: Narrow query rewrite for disambiguation
  // When the US withdrawal narrow rule fires, the original query may contain
  // prominent "Hiệp định Paris" tokens that cause BM25 to rank EVT_0337 above
  // EVT_0339. Prepend withdrawal-focused terms to boost the correct document.
  let retrievalQuery = pipelineQuery;
  if (routing.reasoning.includes('US withdrawal')) {
    retrievalQuery = 'Mỹ rút toàn bộ quân khỏi Việt Nam sau Hiệp định Paris 1973 ' + pipelineQuery;
    console.log(`   📝 Query rewrite for withdrawal disambiguation`);
  }

  // Step 1.6: Focus profile query expansion (Patch 7L-B)
  // Detect domain-specific profiles and prepend expansion terms for better retrieval targeting.
  const focusProfile = detectFocusProfile(pipelineQuery, routing.query_frame);
  if (focusProfile && !routing.reasoning.includes('US withdrawal')) {
    retrievalQuery = focusProfile.expansion_terms + ' ' + retrievalQuery;
    console.log(`   📝 Focus profile: ${focusProfile.id} → query expanded`);
  }

  // Step 1.7: Treaty subtopic query expansion (Patch 7M)
  const treatySubtopic = detectTreatySubtopicFocus(pipelineQuery);
  if (treatySubtopic && treatySubtopic.expansion_terms) {
    // Only prepend if NOT already covered by focus profile expansion
    if (!focusProfile || !focusProfile.expansion_terms.includes(treatySubtopic.expansion_terms.split(' ')[0])) {
      retrievalQuery = treatySubtopic.expansion_terms + ' ' + retrievalQuery;
      console.log(`   📝 Treaty subtopic: ${treatySubtopic.treaty}/${treatySubtopic.subtopic} → query expanded`);
    }
  }

  // Step 1.8: Timeline topic query expansion (Patch 7M)
  const timelineTopic = detectTimelineTopicFocus(pipelineQuery, routing.query_frame);
  if (timelineTopic && timelineTopic.expansion_terms) {
    if (!focusProfile || !focusProfile.expansion_terms.includes(timelineTopic.expansion_terms.split(' ')[0])) {
      retrievalQuery = timelineTopic.expansion_terms + ' ' + retrievalQuery;
      console.log(`   📝 Timeline topic: ${timelineTopic.topic} → query expanded`);
    }
  }

  // Step 1.9: Comparison side query expansion (Patch 7N-B)
  // For comparison/disambiguation queries, prepend balanced expansion terms for both sides
  const querySides = extractComparisonSides(pipelineQuery);
  if (querySides && !routing.reasoning.includes('US withdrawal') && !options.no_embedding && !options.no_api && !isNoApiExportMode(options)) {
    const sideAExpanded = expandComparisonSideTerms(querySides.side_a);
    const sideBExpanded = expandComparisonSideTerms(querySides.side_b);
    // Build balanced expansion: top 5 terms from each side
    const sideATerms = sideAExpanded.slice(0, 5).join(' ');
    const sideBTerms = sideBExpanded.slice(0, 5).join(' ');
    const compExpansion = `${sideATerms} ${sideBTerms}`;
    // Only prepend if not already covered by focus/treaty/timeline
    if (!retrievalQuery.toLowerCase().includes(sideAExpanded[0] ?? '') ||
        !retrievalQuery.toLowerCase().includes(sideBExpanded[0] ?? '')) {
      retrievalQuery = compExpansion + ' ' + retrievalQuery;
      console.log(`   📝 Comparison side expansion: [${querySides.side_a}] vs [${querySides.side_b}]`);
    }
  }

  // Step 1.10: Entity collision query rewrite (Patch 9E)
  // When entity collision is detected, prepend entity-specific terms
  // to boost BM25 ranking for the correct entity.
  const entityProfile = routing.query_frame?.entity_profile;
  if (entityProfile?.expansion_terms?.length && !routing.reasoning.includes('US withdrawal')) {
    const entityExpansion = entityProfile.expansion_terms.join(' ');
    retrievalQuery = entityExpansion + ' ' + retrievalQuery;
    console.log(`   📝 Entity collision expansion: ${entityProfile.id} → [${entityProfile.expansion_terms.slice(0, 3).join(', ')}...]`);
  }

  // Step 2: Retrieve via dual-lane orchestrator
  // Patch 8D-A: Only pass vector stores to orchestrator when useHybrid is explicitly true.
  // When useHybrid=false, vector stores are reserved for conditional rescue only.
  console.log(`🔍 Retrieving (dual-lane)...`);
  const { useHybrid } = pipelineConfig;
  const indexes: DualIndexConfig = {
    eventBM25,
    synthesisBM25,
    rulesBM25,
    eventVectorStore: useHybrid ? eventVectorStore : null,
    synthesisVectorStore: useHybrid ? synthesisVectorStore : null,
  };
  const searchResults = await orchestrateRetrieval(retrievalQuery, routing, indexes, 10);
  console.log(`   Found ${searchResults.length} candidates`);

  // Step 3: Rerank
  console.log(`📊 Reranking...`);
  const reranked = await rerankResults(pipelineQuery, searchResults, dataset);
  console.log(`   Top result: [${reranked[0]?.doc_id}] score=${reranked[0]?.rerank_score.toFixed(3)}`);

  // Step 3.5: Hard-negative guard (query-specific)
  const guarded = applyHardNegativeGuard(pipelineQuery, reranked, dataset);

  // Step 3.7: Evidence selection (PATCH 7E)
  // Assign primary/supporting/contrast/excluded roles based on QueryFrame
  console.log(`🏷️ Evidence selection...`);
  const evidenceSelection = selectEvidence(guarded, dataset, routing.query_frame, pipelineQuery);
  console.log(`   Roles: P=${evidenceSelection.diagnostics.primary_count} S=${evidenceSelection.diagnostics.supporting_count} C=${evidenceSelection.diagnostics.contrast_count} X=${evidenceSelection.diagnostics.excluded_count}`);
  if (evidenceSelection.diagnostics.warnings.length > 0) {
    for (const w of evidenceSelection.diagnostics.warnings) console.log(`   ⚠️ ${w}`);
  }

  // Build evidence-filtered candidate list for context builder:
  // Keep primary + supporting + contrast (for comparison intents).
  // EXCLUDE docs with role='excluded'.
  // For disambiguation: demote contrast docs to the end so they don't become primary.
  let evidenceFilteredCandidates = buildEvidenceFilteredCandidates(guarded, evidenceSelection, routing.intent);

  // Patch 8D-B: Track vector rescue results for debug trace
  let vectorRescueTrace: { side: string; accepted: string[]; rejected: string[] } | undefined;

  // Step 3.9: Conditional Vector Rescue (Patch 8D-A, hardened 8D-B)
  // For two-sided queries where one side is missing from BM25 retrieval,
  // attempt a targeted vector search using side-specific expansion terms.
  // Patch 8D-B: Each candidate must pass a strong lexical precision gate —
  // cosine similarity alone is NOT sufficient. Docs matching only year/generic
  // tokens are REJECTED (e.g. "Hợp tác xã 1958" for "Hiến pháp 1959").
  if (querySides && !routing.reasoning.includes('US withdrawal')) {
    const sideAExpanded = expandComparisonSideTerms(querySides.side_a);
    const sideBExpanded = expandComparisonSideTerms(querySides.side_b);
    const candidateIds = new Set(evidenceFilteredCandidates.map(c => c.doc_id));

    // Check if each side has at least one doc in candidates (using title-priority matching)
    const hasSideDoc = (sideTerms: string[]): boolean => {
      const multiWordTerms = sideTerms.filter(t => t.length > 5 && t.includes(' '));
      for (const c of evidenceFilteredCandidates) {
        const doc = dataset.canonicalEvents.find(d => d.doc_id === c.doc_id) ??
                    dataset.canonicalSynthesis.find(d => d.doc_id === c.doc_id);
        if (!doc) continue;
        const titleText = doc.title.toLowerCase().normalize('NFKC');
        if (multiWordTerms.some(t => titleText.includes(t))) return true;
      }
      return false;
    };

    // Patch 8D-B: Strong lexical precision gate for vector rescue candidates.
    // Returns 'ACCEPT' only if the doc contains a discriminating multi-word phrase
    // from the side terms that shares at least 2 tokens with the side label itself.
    // Generic expansion terms like "miền bắc" are filtered out to prevent false positives.
    const auditVectorRescueCandidate = (
      docId: string, sideLabel: string, sideTerms: string[]
    ): { verdict: 'ACCEPT' | 'REJECT'; reason: string; strongMatches: string[] } => {
      const doc = dataset.canonicalEvents.find(d => d.doc_id === docId) ??
                  dataset.canonicalSynthesis.find(d => d.doc_id === docId);
      if (!doc) return { verdict: 'REJECT', reason: 'Doc not found in dataset', strongMatches: [] };

      const fullText = `${doc.title} ${doc.summary} ${doc.text_for_embedding || ''}`.toLowerCase().normalize('NFKC');
      const sideLabelNorm = sideLabel.toLowerCase().normalize('NFKC');
      const sideLabelTokens = new Set(sideLabelNorm.split(/\s+/).filter(t => t.length > 2));
      const multiWordTerms = sideTerms.filter(t => t.length > 5 && t.includes(' '));

      // Filter: only phrases that share ≥2 tokens with the side label are discriminating.
      // This prevents generic expansion terms like "miền bắc" from matching "Hiến pháp 1959".
      const discriminatingTerms = multiWordTerms.filter(term => {
        const termTokens = term.split(/\s+/);
        const overlap = termTokens.filter(tk => sideLabelTokens.has(tk)).length;
        return overlap >= 2;
      });

      // Also accept if the exact side label appears in the text
      if (fullText.includes(sideLabelNorm)) {
        return { verdict: 'ACCEPT', reason: `Exact side label match: "${sideLabel}"`, strongMatches: [sideLabelNorm] };
      }

      const strongMatches = discriminatingTerms.filter(t => fullText.includes(t));
      if (strongMatches.length > 0) {
        return { verdict: 'ACCEPT', reason: `Strong phrase match: "${strongMatches[0]}"`, strongMatches };
      }
      return {
        verdict: 'REJECT',
        reason: `No discriminating phrase for "${sideLabel}"; title="${doc.title.substring(0, 60)}"`,
        strongMatches: [],
      };
    };

    const sideAPresent = hasSideDoc(sideAExpanded);
    const sideBPresent = hasSideDoc(sideBExpanded);
    const missingSide = !sideAPresent ? 'A' : !sideBPresent ? 'B' : null;

    if (missingSide) {
      const missingTerms = missingSide === 'A' ? sideAExpanded : sideBExpanded;
      const missingSideName = missingSide === 'A' ? querySides.side_a : querySides.side_b;
      const vectorStore = eventVectorStore ?? synthesisVectorStore;

      if (vectorStore && vectorStore.size() > 0) {
        console.log(`   🚁 Vector rescue: side ${missingSide} "${missingSideName}" missing from BM25`);

        // Patch 8F-C: Hard guard — no embedding API calls in no-cost mode
        if (!shouldAllowExternalApi()) {
          console.log(`   ℹ️ Vector rescue skipped: external API disabled by no-cost guard`);
        } else {
        try {
          const rescueQuery = missingTerms.slice(0, 8).join(' ');
          const OpenAI = (await import('openai')).default;
          const client = new OpenAI();
          const embResponse = await client.embeddings.create({
            model: 'text-embedding-3-large',
            input: rescueQuery,
            dimensions: 3072,
          });
          const queryEmb = embResponse.data[0].embedding;

          const vectorResults = await vectorStore.search(queryEmb, 5);
          const rescuedDocs: RerankedResult[] = [];
          const rejectedDocs: string[] = [];
          for (const vr of vectorResults) {
            if (candidateIds.has(vr.doc_id)) continue;
            if (vr.score < 0.3) continue;

            // Patch 8D-B: Strong lexical precision gate
            const audit = auditVectorRescueCandidate(vr.doc_id, missingSideName, missingTerms);
            if (audit.verdict === 'REJECT') {
              rejectedDocs.push(vr.doc_id);
              console.log(`   🚁 REJECTED: ${vr.doc_id} (cos=${vr.score.toFixed(3)}) — ${audit.reason}`);
              continue;
            }

            rescuedDocs.push({
              doc_id: vr.doc_id,
              original_score: vr.score * 20,
              rerank_score: vr.score * 20,
              metadata: vr.metadata,
              evidence_role: 'supporting' as const,
              evidence_role_score: vr.score,
              evidence_reasons: ['vector_rescue'],
              evidence_rank: 50,
            });
            candidateIds.add(vr.doc_id);
            console.log(`   🚁 ACCEPTED: ${vr.doc_id} (cos=${vr.score.toFixed(3)}) — ${audit.reason}`);
          }
          if (rescuedDocs.length > 0) {
            evidenceFilteredCandidates = [...evidenceFilteredCandidates, ...rescuedDocs];
            console.log(`   🚁 Vector rescue: ${rescuedDocs.length} accepted, ${rejectedDocs.length} rejected for side ${missingSide}`);
          } else {
            console.log(`   🚁 Vector rescue: NO valid docs for side ${missingSide} (${rejectedDocs.length} rejected by precision gate)`);
          }
          vectorRescueTrace = { side: missingSide, accepted: rescuedDocs.map(d => d.doc_id), rejected: rejectedDocs };
        } catch (err: any) {
          console.log(`   ⚠️ Vector rescue failed: ${err.message?.substring(0, 80)}`);
        }
        } // end else (external API allowed)
      } else {
        console.log(`   ℹ️ Vector rescue unavailable: no vector store loaded for side ${missingSide} "${missingSideName}"`);
      }
    }
  }

  // Step 4: Build context bundle (NEVER raw top-k)
  console.log(`📦 Building context bundle...`);
  const contextBundle = buildContextBundle(routing.intent, evidenceFilteredCandidates, dataset, routing.query_frame, { eventBM25, synthesisBM25 });
  console.log(`   Primary: ${contextBundle.primary_docs.length}, Supporting: ${contextBundle.supporting_docs.length}`);
  if (contextBundle.warnings.length > 0) {
    for (const w of contextBundle.warnings) console.log(`   ⚠️ ${w}`);
  }
  if (contextBundle.evidence_warnings && contextBundle.evidence_warnings.length > 0) {
    for (const w of contextBundle.evidence_warnings) console.log(`   🔍 ${w}`);
  }

  if (isNoApiExportMode(options) || options.no_llm || options.preview_only) {
    const exportResponse = buildNoApiExportResponse(
      query,
      pipelineQuery,
      dataset,
      contextBundle,
      retrievalQuery,
      searchResults,
      guard,
      routing,
      evidenceSelection,
    );
    exportResponse._debugTrace = {
      routing: {
        intent: routing.intent,
        indexes: routing.target_indexes,
        retrieval_query: retrievalQuery,
      },
      retrieval_doc_ids: searchResults.map(r => r.doc_id),
      evidence_primary_ids: evidenceSelection.primary.map(e => e.doc_id),
      evidence_supporting_ids: evidenceSelection.supporting.map(e => e.doc_id),
      context_primary_ids: contextBundle.primary_docs.map(d => d.doc_id),
      context_supporting_ids: contextBundle.supporting_docs.map(d => d.doc_id),
      citation_plan_ids: contextBundle.citation_plan?.map((c: any) => c.doc_id || c.record_id || '') ?? [],
      vector_rescue: vectorRescueTrace,
    };
    const capDecision = classifyCapability({
      query: pipelineQuery,
      guardResult: guard,
      routing,
      contextBundle,
      confidence: exportResponse.confidence,
      citationCount: exportResponse.citations.length,
      evidenceDiagnostics: evidenceSelection.diagnostics,
    });
    exportResponse._capabilityDecision = capDecision;
    attachMemoryMetadataAfterResponse(exportResponse, memoryContext, options);
    return exportResponse;
  }

  // Step 5: Generate answer (Patch 7G: pass queryFrame for verifier)
  console.log(`💬 Generating answer...`);
  const response = await generateAnswer(pipelineQuery, contextBundle, searchResults, dataset, routing.query_frame);

  const elapsed = Date.now() - startTime;
  console.log(`✅ Done in ${elapsed}ms | Confidence: ${response.confidence}\n`);

  // Patch 8D-B: Attach debug trace for diagnostic scripts
  response._debugTrace = {
    routing: {
      intent: routing.intent,
      indexes: routing.target_indexes,
      retrieval_query: retrievalQuery,
    },
    retrieval_doc_ids: searchResults.map(r => r.doc_id),
    evidence_primary_ids: evidenceSelection.primary.map(e => e.doc_id),
    evidence_supporting_ids: evidenceSelection.supporting.map(e => e.doc_id),
    context_primary_ids: contextBundle.primary_docs.map(d => d.doc_id),
    context_supporting_ids: contextBundle.supporting_docs.map(d => d.doc_id),
    citation_plan_ids: contextBundle.citation_plan?.map((c: any) => c.doc_id || c.record_id || '') ?? [],
    vector_rescue: vectorRescueTrace,
  };

  // ── Step 6 (Patch 9I): Capability classification + policy + monitoring ──
  const capDecision = classifyCapability({
    query: pipelineQuery,
    guardResult: guard,
    routing,
    contextBundle,
    confidence: response.confidence,
    citationCount: response.citations.length,
    evidenceDiagnostics: evidenceSelection.diagnostics,
  });
  const policyResult = applyAnswerPolicy(response, capDecision);
  const finalResponse = policyResult.adjustedResponse;
  finalResponse._debugTrace = response._debugTrace;
  finalResponse._capabilityDecision = capDecision;
  attachAnswerFocusMetadata(finalResponse, buildAnswerFocusMetadata(pipelineQuery, dataset, contextBundle));
  attachMemoryMetadataAfterResponse(finalResponse, memoryContext, options);

  // Monitoring
  if (!options.disable_runtime_mutation && !isNoApiExportMode(options)) {
    recordOutcome(createMonitoringRecord({
      query: pipelineQuery,
      routeIntent: routing.intent,
      bucket: capDecision.bucket,
      policy: capDecision.policy,
      evidenceQuality: capDecision.evidenceQuality,
      citationCount: finalResponse.citations.length,
      confidence: finalResponse.confidence,
      upgradeSignals: capDecision.upgradeSignals,
    }));
  }

  return finalResponse;
}

// ─── Evidence-Annotated Candidate List ───────────────────────

/**
 * Build a filtered, annotated, and reordered RerankedResult[] from EvidenceSelection.
 *
 * Patch 7E-2: ANNOTATES each candidate with evidence_role/evidence_role_score/evidence_reasons/evidence_rank
 * so downstream context builder can read roles directly instead of relying on array position.
 *
 * Strategy:
 * - Remove 'excluded' docs entirely.
 * - Annotate each kept candidate with evidence role info.
 * - For disambiguation/misconception: primary first, supporting second, contrast last.
 * - For comparison: keep all, annotated.
 * - For all others: primary first, then supporting, then contrast.
 */
function buildEvidenceFilteredCandidates(
  originalCandidates: RerankedResult[],
  selection: ReturnType<typeof selectEvidence>,
  intent: string
): RerankedResult[] {
  // Build lookup: doc_id → EvidenceItem
  const roleMap = new Map<string, { role: string; role_score: number; reasons: string[]; rank: number }>();

  // Assign rank within each role group
  for (const [_group, items] of [
    ['primary', selection.primary],
    ['supporting', selection.supporting],
    ['contrast', selection.contrast],
    ['excluded', selection.excluded],
  ] as const) {
    for (let i = 0; i < items.length; i++) {
      roleMap.set(items[i].doc_id, {
        role: items[i].role,
        role_score: items[i].role_score,
        reasons: items[i].reasons.map(r => r.code),
        rank: i,
      });
    }
  }

  // Filter out excluded, annotate kept candidates
  const kept: RerankedResult[] = [];
  for (const c of originalCandidates) {
    const info = roleMap.get(c.doc_id);
    if (info && info.role === 'excluded') continue;

    // Annotate with evidence role
    const annotated: RerankedResult = {
      ...c,
      evidence_role: (info?.role as RerankedResult['evidence_role']) ?? 'supporting',
      evidence_role_score: info?.role_score ?? 0,
      evidence_reasons: info?.reasons ?? [],
      evidence_rank: info?.rank ?? 99,
    };
    kept.push(annotated);
  }

  // For disambiguation/misconception: ensure contrast docs are AFTER primary+supporting
  if (intent === 'multi_hop' || intent === 'disambiguation' || intent === 'misconception_check') {
    const primaryDocs = kept.filter(c => c.evidence_role === 'primary');
    const supportingDocs = kept.filter(c => c.evidence_role === 'supporting');
    const contrastDocs = kept.filter(c => c.evidence_role === 'contrast');

    return [
      ...primaryDocs.sort((a, b) => b.rerank_score - a.rerank_score),
      ...supportingDocs.sort((a, b) => b.rerank_score - a.rerank_score),
      ...contrastDocs.sort((a, b) => b.rerank_score - a.rerank_score),
    ];
  }

  // For other intents: keep order but annotated
  return kept;
}
