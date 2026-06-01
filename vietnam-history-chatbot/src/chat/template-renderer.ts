/**
 * Template Renderer — Stage 8B2 (No-API, deterministic)
 *
 * Renders an extractive answer preview/payload from:
 * query, intent result, answer plan, response template, context payload,
 * citation payload, and provenance links.
 *
 * This module NEVER calls an LLM/API and NEVER fabricates citations.
 */

import type { AnswerIntent, AnswerPlan, AnswerStatus, IntentClassification } from './answer-planner.js';
import type { ResponseTemplate } from './response-template.js';
import { getTemplate } from './response-template.js';

export interface RendererContextDoc {
  doc_id: string;
  doc_type: 'event' | 'synthesis' | 'disambiguation_rule' | string;
  title: string;
  rank?: number;
  text_excerpt?: string;
  source_ids?: string[];
  source_cards?: RendererSourceCard[];
  provenance_links?: RendererProvenanceLink[];
  is_rule_doc?: boolean;
  is_comparison_doc?: boolean;
  needs_source_review?: boolean;
}

export interface RendererSourceCard {
  source_id: string;
  title?: string;
  url?: string;
  organization?: string;
  reliability_level?: string;
  source_lookup_ok?: boolean;
}

export interface RendererCitationMarker {
  marker: string;
  source_id: string;
  doc_id?: string;
  title?: string;
  url?: string;
}

export interface RendererProvenanceLink {
  link_id?: string;
  link_type?: string;
  to_source_id?: string | null;
  to_doc_id?: string | null;
  source_pack_id?: string;
}

export interface TemplateRendererInput {
  case_id: string;
  query: string;
  intent_result: IntentClassification;
  answer_plan: AnswerPlan;
  template?: ResponseTemplate;
  context_payload: {
    context_docs: RendererContextDoc[];
    rule_context_present?: boolean;
    context_build_status?: 'ok' | 'warning' | 'fail';
  };
  citation_payload: {
    source_cards: RendererSourceCard[];
    citation_markers?: RendererCitationMarker[];
  };
  provenance_payload?: {
    links: RendererProvenanceLink[];
  };
}

export interface RenderedSection {
  section_id: string;
  heading: string;
  content_preview: string;
  supporting_doc_ids: string[];
  citation_markers: string[];
  evidence_status: 'source_backed' | 'provenance_backed' | 'source_and_provenance_backed' | 'flagged_source_gap' | 'weak_no_citation';
}

export interface TemplateRendererOutput {
  case_id: string;
  query: string;
  intent: AnswerIntent;
  template_id: string;
  answer_status: AnswerStatus;
  rendered_answer_preview: string;
  sections: RenderedSection[];
  word_count_estimate: number;
  bullet_count: number;
  direct_answer_first: boolean;
  citation_policy_satisfied: boolean;
  rule_context_used: boolean;
  provenance_used: boolean;
  context_weak_warning: boolean;
  docs_without_citation_handled: boolean;
  should_ask_clarification: boolean;
  should_abstain: boolean;
  fake_citation_count: number;
  unsupported_claim_risk: boolean;
  over_expansion_risk: 'low' | 'medium' | 'high';
  notes: string;
}

interface GroundedDoc {
  doc: RendererContextDoc;
  marker: string | null;
  sourceBacked: boolean;
  provenanceBacked: boolean;
  flaggedGap: boolean;
}

function words(text: string): string[] {
  return text.trim().split(/\s+/).filter(Boolean);
}

function estimateWords(text: string): number {
  return words(text).length;
}

function clampWords(text: string, maxWords: number): string {
  const tokens = words(text);
  if (tokens.length <= maxWords) return text.trim();
  return `${tokens.slice(0, maxWords).join(' ')}...`;
}

function compact(text: string, fallback = ''): string {
  const cleaned = String(text || fallback || '').replace(/\s+/g, ' ').trim();
  return cleaned;
}

function firstSentence(text: string, maxWords = 28): string {
  const cleaned = compact(text);
  const sentence = cleaned.split(/(?<=[.!?。])\s+/)[0] || cleaned;
  return clampWords(sentence, maxWords);
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function outputAnswerStatus(input: TemplateRendererInput, hasCitation: boolean, contextWeak: boolean): AnswerStatus {
  const intent = input.intent_result.intent;
  if (intent === 'out_of_scope') return 'out_of_scope';
  if (intent === 'negative_gap') return 'insufficient_data';
  if (intent === 'follow_up' || intent === 'unclear' || input.intent_result.requires_clarification) {
    return 'needs_clarification';
  }
  if (input.template?.citation_required && !hasCitation) {
    return input.template.allow_partial_answer || contextWeak ? 'partially_answerable' : 'insufficient_data';
  }
  if (contextWeak) return 'partially_answerable';
  return input.answer_plan.answer_status === 'answerable' ? 'answerable' : input.answer_plan.answer_status;
}

function realMarkers(input: TemplateRendererInput): RendererCitationMarker[] {
  const sourceIds = new Set((input.citation_payload.source_cards || []).map(card => card.source_id).filter(Boolean));
  return (input.citation_payload.citation_markers || [])
    .filter(marker => marker.source_id && sourceIds.has(marker.source_id));
}

function markerForSource(sourceId: string, index: number): string {
  return `[S${index + 1}]`;
}

function buildGroundedDocs(input: TemplateRendererInput): GroundedDoc[] {
  const explicitMarkers = realMarkers(input);
  const markerBySource = new Map<string, string>();
  explicitMarkers.forEach(marker => markerBySource.set(marker.source_id, marker.marker));
  for (const [index, card] of (input.citation_payload.source_cards || []).entries()) {
    if (card.source_id && !markerBySource.has(card.source_id)) {
      markerBySource.set(card.source_id, markerForSource(card.source_id, index));
    }
  }

  return (input.context_payload.context_docs || []).map(doc => {
    const sourceIds = doc.source_ids || [];
    const firstSource = sourceIds.find(sourceId => markerBySource.has(sourceId));
    const sourceBacked = !!firstSource && (doc.source_cards || []).some(card => card.source_id === firstSource);
    const provenanceBacked = (doc.provenance_links || []).length > 0;
    const flaggedGap = !sourceBacked && doc.needs_source_review === true;
    return {
      doc,
      marker: firstSource ? markerBySource.get(firstSource)! : null,
      sourceBacked,
      provenanceBacked,
      flaggedGap,
    };
  });
}

function evidenceStatus(item: GroundedDoc): RenderedSection['evidence_status'] {
  if (item.sourceBacked && item.provenanceBacked) return 'source_and_provenance_backed';
  if (item.sourceBacked) return 'source_backed';
  if (item.provenanceBacked) return 'provenance_backed';
  if (item.flaggedGap) return 'flagged_source_gap';
  return 'weak_no_citation';
}

function sectionFromDoc(sectionId: string, heading: string, item: GroundedDoc, content: string): RenderedSection {
  return {
    section_id: sectionId,
    heading,
    content_preview: content,
    supporting_doc_ids: [item.doc.doc_id],
    citation_markers: item.marker ? [item.marker] : [],
    evidence_status: evidenceStatus(item),
  };
}

function directLine(query: string, item: GroundedDoc | undefined, status: AnswerStatus): string {
  if (status === 'needs_clarification') {
    return 'Bạn đang hỏi tiếp về sự kiện/khái niệm nào? Hãy nêu rõ đối tượng để mình không đoán sai.';
  }
  if (status === 'insufficient_data' || status === 'out_of_scope') {
    return status === 'out_of_scope'
      ? 'Câu hỏi này nằm ngoài phạm vi dữ liệu hiện có, nên mình không trả lời như một kết luận chắc chắn.'
      : 'Dữ liệu hiện có chưa đủ chi tiết để trả lời chắc chắn yêu cầu này.';
  }
  if (!item) return 'Chưa có đủ ngữ cảnh đã dẫn nguồn để trả lời chắc chắn.';
  const citation = item.marker ? ` ${item.marker}` : '';
  return `${firstSentence(item.doc.text_excerpt || item.doc.title, 30)}${citation}`;
}

function renderShortDirect(input: TemplateRendererInput, grounded: GroundedDoc[], status: AnswerStatus): RenderedSection[] {
  const first = grounded[0];
  if (!first) {
    return [{
      section_id: 'direct_answer',
      heading: 'Trả lời ngắn',
      content_preview: directLine(input.query, undefined, status),
      supporting_doc_ids: [],
      citation_markers: [],
      evidence_status: 'weak_no_citation',
    }];
  }
  return [sectionFromDoc('direct_answer', 'Trả lời ngắn', first, directLine(input.query, first, status))];
}

function renderBulletExplanation(input: TemplateRendererInput, grounded: GroundedDoc[], status: AnswerStatus): RenderedSection[] {
  const sections: RenderedSection[] = [];
  const first = grounded[0];
  if (first) sections.push(sectionFromDoc('direct_answer', 'Ý chính', first, directLine(input.query, first, status)));
  for (const [index, item] of grounded.slice(1, input.template?.max_bullets ?? 4).entries()) {
    const marker = item.marker ? ` ${item.marker}` : '';
    sections.push(sectionFromDoc(`evidence_${index + 1}`, `Dẫn chứng ${index + 1}`, item, `- ${firstSentence(item.doc.text_excerpt || item.doc.title, 24)}${marker}`));
  }
  return sections.length ? sections : renderShortDirect(input, grounded, status);
}

function renderComparison(input: TemplateRendererInput, grounded: GroundedDoc[], status: AnswerStatus): RenderedSection[] {
  const sections: RenderedSection[] = [];
  const [sideA, sideB] = comparisonLabels(input.query);
  const first = grounded[0];
  const second = grounded.find(item => item !== first) ?? grounded[1];
  if (!first) return renderShortDirect(input, grounded, status);

  sections.push(sectionFromDoc(
    'comparison_side_a',
    `A - ${sideA}`,
    first,
    `- ${firstSentence(first.doc.text_excerpt || first.doc.title, 32)}${first.marker ? ` ${first.marker}` : ''}`,
  ));

  if (second) {
    sections.push(sectionFromDoc(
      'comparison_side_b',
      `B - ${sideB}`,
      second,
      `- ${firstSentence(second.doc.text_excerpt || second.doc.title, 32)}${second.marker ? ` ${second.marker}` : ''}`,
    ));
  } else {
    sections.push({
      section_id: 'comparison_side_b_gap',
      heading: `B - ${sideB}`,
      content_preview: `Nguồn hiện có chưa đủ để nêu chắc vế ${sideB}.`,
      supporting_doc_ids: [],
      citation_markers: [],
      evidence_status: 'weak_no_citation',
    });
  }

  const diffMarker = second?.marker ?? first.marker ?? '';
  sections.push(sectionFromDoc(
    'main_difference',
    'Khác nhau chính',
    second ?? first,
    `- Hai vế thuộc bối cảnh hoặc hệ quả lịch sử khác nhau; chỉ kết luận trong phạm vi nguồn đã truy xuất.${diffMarker ? ` ${diffMarker}` : ''}`,
  ));
  return sections;
}

function comparisonLabels(query: string): [string, string] {
  if (/gen[eè]ve|geneve/i.test(query) && /paris/i.test(query)) {
    return ['Hiệp định Genève 1954', 'Hiệp định Paris 1973'];
  }
  const parts = query
    .split(/\s+(?:và|với|so với)\s+/i)
    .map(part => compact(part.replace(/[?.!]+$/g, ''), 'vế').slice(0, 80))
    .filter(Boolean);
  if (parts.length >= 2) return [parts[0], parts[1]];
  return ['Vế A', 'Vế B'];
}

function renderTimeline(input: TemplateRendererInput, grounded: GroundedDoc[]): RenderedSection[] {
  const docs = grounded.slice(0, Math.min(input.template?.max_bullets ?? 8, 8));
  if (docs.length === 0) return renderShortDirect(input, grounded, 'partially_answerable');
  return docs.map((item, index) => {
    const marker = item.marker ? ` ${item.marker}` : '';
    return sectionFromDoc(`timeline_${index + 1}`, `Mốc ${index + 1}`, item, `- ${item.doc.title}: ${firstSentence(item.doc.text_excerpt || item.doc.title, 22)}${marker}`);
  });
}

function renderClaimEvidence(input: TemplateRendererInput, grounded: GroundedDoc[], status: AnswerStatus): RenderedSection[] {
  const sections: RenderedSection[] = [];
  const first = grounded[0];
  if (first) sections.push(sectionFromDoc('thesis', 'Luận điểm', first, `${directLine(input.query, first, status)}`));
  for (const [index, item] of grounded.slice(0, 4).entries()) {
    const marker = item.marker ? ` ${item.marker}` : '';
    sections.push(sectionFromDoc(`claim_${index + 1}`, `Luận cứ ${index + 1}`, item, `- Luận cứ: ${firstSentence(item.doc.text_excerpt || item.doc.title, 26)}${marker}`));
  }
  return sections;
}

function renderCitationSource(input: TemplateRendererInput, grounded: GroundedDoc[], status: AnswerStatus): RenderedSection[] {
  if (grounded.length === 0) return renderShortDirect(input, grounded, status);
  const sections: RenderedSection[] = [];
  for (const [index, item] of grounded.slice(0, Math.min(input.template?.max_bullets ?? 3, 3)).entries()) {
    const marker = item.marker ? ` ${item.marker}` : '';
    const sourceLabel = item.doc.source_cards?.[0]?.title || item.doc.title;
    sections.push(sectionFromDoc(
      `source_${index + 1}`,
      `Nguồn ${index + 1}`,
      item,
      `- ${sourceLabel} ghi nhận ${firstSentence(item.doc.text_excerpt || item.doc.title, 26)}${marker}. Vì vậy, nguồn này hỗ trợ nhận định được hỏi${marker}.`,
    ));
  }
  const first = grounded[0];
  sections.push(sectionFromDoc(
    'source_conclusion',
    'Kết luận ngắn',
    first,
    `Dựa trên các nguồn trên, chỉ nên trả lời trong phạm vi dữ liệu được truy xuất${first.marker ? ` ${first.marker}` : ''}.`,
  ));
  return sections;
}

function renderClarification(input: TemplateRendererInput): RenderedSection[] {
  return [{
    section_id: 'clarification_request',
    heading: 'Cần làm rõ',
    content_preview: 'Bạn đang muốn hỏi về sự kiện, nhân vật hoặc giai đoạn nào? Hãy nêu rõ đối tượng để mình không đoán sai.',
    supporting_doc_ids: [],
    citation_markers: [],
    evidence_status: 'weak_no_citation',
  }];
}

function renderInsufficient(input: TemplateRendererInput, grounded: GroundedDoc[], status: AnswerStatus): RenderedSection[] {
  const sections: RenderedSection[] = [{
    section_id: status === 'out_of_scope' ? 'scope_notice' : 'data_limitation_warning',
    heading: status === 'out_of_scope' ? 'Ngoài phạm vi' : 'Dữ liệu chưa đủ',
    content_preview: directLine(input.query, grounded[0], status),
    supporting_doc_ids: [],
    citation_markers: [],
    evidence_status: 'weak_no_citation',
  }];
  const safe = grounded.find(item => item.sourceBacked || item.provenanceBacked);
  if (safe) {
    sections.push(sectionFromDoc('safe_related_info', 'Ngữ cảnh liên quan an toàn', safe, `Thông tin liên quan có thể tham khảo: ${firstSentence(safe.doc.text_excerpt || safe.doc.title, 24)}${safe.marker ? ` ${safe.marker}` : ''}`));
  }
  return sections;
}

function renderSections(input: TemplateRendererInput, grounded: GroundedDoc[], status: AnswerStatus): RenderedSection[] {
  const shape = input.template?.output_shape ?? input.answer_plan.output_shape;
  if (status === 'needs_clarification' || shape === 'clarification_question') return renderClarification(input);
  if (status === 'insufficient_data' || status === 'out_of_scope' || shape === 'insufficient_data') return renderInsufficient(input, grounded, status);
  if (shape === 'comparison_table') return renderComparison(input, grounded, status);
  if (shape === 'timeline') return renderTimeline(input, grounded);
  if (input.intent_result.intent === 'citation_source') return renderCitationSource(input, grounded, status);
  if (shape === 'claim_evidence') return renderClaimEvidence(input, grounded, status);
  if (shape === 'bullet_explanation') return renderBulletExplanation(input, grounded, status);
  return renderShortDirect(input, grounded, status);
}

function joinSections(sections: RenderedSection[], template: ResponseTemplate): string {
  const body = sections.map(section => {
    if (section.content_preview.startsWith('|') || section.content_preview.startsWith('-')) {
      return section.content_preview;
    }
    return `${section.heading}: ${section.content_preview}`;
  }).join('\n');
  return clampWords(body, template.max_words);
}

function countBullets(text: string): number {
  return text.split(/\r?\n/).filter(line => /^\s*[-*•]|\|/.test(line)).length;
}

export function renderTemplateAnswer(input: TemplateRendererInput): TemplateRendererOutput {
  const template = input.template ?? getTemplate(input.intent_result.intent);
  const grounded = buildGroundedDocs({ ...input, template });
  const hasCitation = grounded.some(item => item.sourceBacked);
  const hasProvenance = grounded.some(item => item.provenanceBacked) || (input.provenance_payload?.links?.length ?? 0) > 0;
  const contextWeak = input.context_payload.context_build_status === 'warning'
    || input.context_payload.context_build_status === 'fail'
    || grounded.length === 0
    || grounded.some(item => !item.sourceBacked && !item.provenanceBacked && !item.flaggedGap);
  const status = outputAnswerStatus({ ...input, template }, hasCitation, contextWeak);
  const sections = renderSections({ ...input, template }, grounded, status);
  const rendered = joinSections(sections, template);
  const wordCount = estimateWords(rendered);
  const bulletCount = countBullets(rendered);
  const ruleContextUsed = grounded.some(item => item.doc.is_rule_doc || item.doc.is_comparison_doc);
  const abstentionTemplate = template.output_shape === 'insufficient_data' || template.must_abstain_if_insufficient;
  const docsWithoutCitationHandled = abstentionTemplate || grounded.length === 0 || grounded
    .filter(item => !item.sourceBacked && !item.provenanceBacked)
    .every(item => item.flaggedGap || item.doc.needs_source_review === true);
  const citationPolicySatisfied = !template.citation_required
    || hasCitation
    || (input.answer_plan.citation_policy.allow_flagged_source_gap && docsWithoutCitationHandled);
  const shouldAskClarification = status === 'needs_clarification' || template.should_ask_clarification;
  const shouldAbstain = status === 'insufficient_data' || status === 'out_of_scope' || template.must_abstain_if_insufficient;
  const unsupportedClaimRisk = template.citation_required && !citationPolicySatisfied;
  const overExpansionHigh = wordCount > template.max_words || bulletCount > template.max_bullets;
  const overExpansionRisk: TemplateRendererOutput['over_expansion_risk'] = overExpansionHigh ? 'high' : wordCount > template.max_words * 0.9 || bulletCount > Math.max(0, template.max_bullets - 1) ? 'medium' : 'low';
  const fakeCitationCount = realMarkers(input).length === (input.citation_payload.citation_markers || []).length ? 0
    : (input.citation_payload.citation_markers || []).length - realMarkers(input).length;
  const directAnswerFirst = status === 'answerable' || status === 'partially_answerable'
    ? (template.output_shape === 'timeline' ? true : template.direct_answer_first)
    : false;

  const notes: string[] = [];
  if (contextWeak) notes.push('context_weak_warning');
  if (template.rule_context_required && !ruleContextUsed) notes.push('rule_context_required_but_missing');
  if (!citationPolicySatisfied) notes.push('citation_policy_not_satisfied');
  if (docsWithoutCitationHandled) notes.push('docs_without_citation_handled');

  return {
    case_id: input.case_id,
    query: input.query,
    intent: input.intent_result.intent,
    template_id: template.template_id,
    answer_status: status,
    rendered_answer_preview: rendered,
    sections,
    word_count_estimate: wordCount,
    bullet_count: bulletCount,
    direct_answer_first: directAnswerFirst,
    citation_policy_satisfied: citationPolicySatisfied,
    rule_context_used: ruleContextUsed,
    provenance_used: hasProvenance,
    context_weak_warning: contextWeak,
    docs_without_citation_handled: docsWithoutCitationHandled,
    should_ask_clarification: shouldAskClarification,
    should_abstain: shouldAbstain,
    fake_citation_count: fakeCitationCount,
    unsupported_claim_risk: unsupportedClaimRisk,
    over_expansion_risk: overExpansionRisk,
    notes: notes.join('; '),
  };
}
