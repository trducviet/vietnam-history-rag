/**
 * Follow-up Resolver — Stage 8C2 (No-API, deterministic)
 *
 * Resolves short conversational references against SessionMemoryState. It is
 * intentionally conservative: unresolved, unsafe, stale, or conflicting memory
 * produces clarification/block decisions rather than guessed referents.
 */

import type { AnswerIntent, IntentClassification } from './answer-planner.js';
import { classifyIntent } from './answer-planner.js';
import type { ActiveEntity, ActiveEvent, SessionMemoryState } from './session-memory.js';
import { hasExplicitHistoricalAnchor, hasFollowUpReference, hasPronominalFollowUpReference } from './memory-policy.js';

export type FollowUpResolutionStatus =
  | 'resolved'
  | 'needs_clarification'
  | 'not_follow_up'
  | 'blocked_by_safety'
  | 'conflict_detected';

export type FollowUpReferentType = 'event' | 'entity' | 'topic' | 'period' | 'comparison' | 'unknown';
export type FollowUpReferentSource = 'active_topic' | 'active_event' | 'active_entity' | 'active_period' | 'last_answer_summary' | 'none';

export interface FollowUpReferent {
  text: string;
  type: FollowUpReferentType;
  confidence: number;
  source: FollowUpReferentSource;
}

export interface FollowUpResolverInput {
  query: string;
  memory: SessionMemoryState;
  intent_result?: IntentClassification;
}

export interface FollowUpResolverOutput {
  is_follow_up: boolean;
  resolution_status: FollowUpResolutionStatus;
  referent: FollowUpReferent;
  resolution_reason: string;
  should_ask_clarification: boolean;
  clarification_question: string;
  safety_flags: string[];
}

function normalize(text: string): string {
  return text.toLowerCase().normalize('NFKC').trim();
}

function emptyReferent(): FollowUpReferent {
  return { text: '', type: 'unknown', confidence: 0, source: 'none' };
}

function highConfidenceEvents(memory: SessionMemoryState): ActiveEvent[] {
  return memory.active_events.filter(event => event.confidence >= 0.75);
}

function highConfidenceEntities(memory: SessionMemoryState): ActiveEntity[] {
  return memory.active_entities.filter(entity => entity.confidence >= 0.75);
}

function highConfidenceOrganizations(memory: SessionMemoryState): ActiveEntity[] {
  return highConfidenceEntities(memory).filter(entity => entity.type === 'organization');
}

function highConfidencePersons(memory: SessionMemoryState): ActiveEntity[] {
  return highConfidenceEntities(memory).filter(entity => entity.type === 'person');
}

function highConfidenceDocuments(memory: SessionMemoryState): ActiveEntity[] {
  return highConfidenceEntities(memory).filter(entity => entity.type === 'document' || entity.type === 'agreement');
}

function hasSummaryFollowUp(query: string): boolean {
  return /tóm\s*lại|tóm\s*tắt|nhắc\s*lại|nói\s*ngắn\s*gọn|nói\s*kỹ\s*hơn|giải\s*thích\s*thêm|điểm\s*chính/i.test(query);
}

function hasChronologyFollowUp(query: string): boolean {
  return /sau\s*đó|tiếp\s*theo|còn\s+sau\s+đó|còn\s+giai\s*đoạn\s+sau/i.test(query);
}

function hasInstructionalFollowUp(query: string): boolean {
  return /trả\s*lời\s*trước|kết\s*luận\s*chính|lập\s*bảng|timeline|đừng\s*lan\s*man|luận\s*điểm|bằng\s*chứng|dẫn\s*chứng|căn\s*cứ/i.test(query)
    && !hasExplicitHistoricalAnchor(query);
}

function hasCitationFollowUp(query: string): boolean {
  return /nguồn\s*(nào|gì)|dựa\s*vào\s*(đâu|nguồn|tài\s*liệu)|căn\s*cứ\s*(nào|gì)|tài\s*liệu\s*(nào|gì)|ở\s*đâu\s*nói\s*vậy|nguồn\s*của\s*phần\s*đó|nguồn\s*cho\s*nhận\s*định\s*đó|bằng\s*chứng\s*(nào|gì)|dẫn\s*chứng\s*(nào|gì)|theo\s*nguồn\s*nào|có\s*nguồn\s*nào/i.test(query);
}

function hasComparisonFollowUp(query: string): boolean {
  return /điểm\s*khác\s*nhau|toàn\s*bộ\s*khác\s*biệt|so\s*sánh\s*thêm|cái\s*nào|giống\s*nhau|khác\s*nhau\s*chính|quan\s*trọng\s*hơn|ý\s*nghĩa\s*lớn\s*hơn/i.test(query);
}

function hasEntityRoleFollowUp(query: string): boolean {
  return /ông\s*(ấy|ta)|bà\s*ấy|người\s*đó|tổ\s*chức\s*(đó|này)|lực\s*lượng\s*(đó|này)|phong\s*trào\s*(đó|này)|hiệp\s*định\s*đó|văn\s*kiện\s*đó|sự\s*kiện\s*đó|vai\s*trò\s*đó|điều\s*đó/i.test(query)
    && /vai\s*trò|ảnh\s*hưởng|ý\s*nghĩa|đóng\s*vai\s*trò|tác\s*động|trong\s*sự\s*kiện/i.test(query);
}

function hasCorrectionOrShift(query: string): boolean {
  return /không\s*phải|ý\s*tôi\s*là|chuyển\s*sang|chủ\s*đề\s*khác|quên\s*(cái|phần|chủ\s*đề)\s*trước|bỏ\s*qua\s*(cái|phần)\s*trước/i.test(query);
}

function hasExplicitNewTopic(query: string): boolean {
  const q = normalize(query);
  return hasExplicitHistoricalAnchor(q) && !/(nó|sự\s*kiện\s*đó|cái\s*đó|điều\s*này|điều\s*đó|tổ\s*chức\s*đó|mối\s*quan\s*hệ\s*đó|hai\s*(sự\s*kiện|văn\s*kiện|mốc)\s*đó)/i.test(q);
}

function comparisonTopic(memory: SessionMemoryState): FollowUpReferent | null {
  if (!memory.active_topic || memory.active_topic.confidence < 0.75) return null;
  if (memory.conversation_focus.scope === 'comparison' && memory.active_comparison_targets.length >= 2) {
    return {
      text: memory.active_comparison_targets.slice(0, 2).join(' và '),
      type: 'comparison',
      confidence: memory.active_topic.confidence,
      source: 'active_topic',
    };
  }
  return null;
}

function isAmbiguousCompoundEventLabel(label: string): boolean {
  return /\b\d+\s*\/\s*\d+(?:\s*\/\s*\d+)?\b| vs | và |,/.test(label);
}

function clarificationFor(reason: string, memory: SessionMemoryState): string {
  if (reason.includes('multiple') || reason.includes('ambiguous')) {
    const candidates = [
      ...highConfidenceEvents(memory).map(item => item.event_label),
      ...highConfidenceEntities(memory).map(item => item.text),
    ].slice(0, 4);
    return candidates.length
      ? `Bạn muốn hỏi về ${candidates.join(' hay ')}?`
      : 'Bạn muốn hỏi về sự kiện, nhân vật hoặc khái niệm nào?';
  }
  if (reason.includes('safety')) return 'Ngữ cảnh trước chưa đủ an toàn để suy ra tham chiếu. Bạn hãy nêu rõ đối tượng cần hỏi.';
  return 'Bạn muốn hỏi về sự kiện, nhân vật hoặc khái niệm nào?';
}

function resolved(referent: FollowUpReferent, reason: string): FollowUpResolverOutput {
  return {
    is_follow_up: true,
    resolution_status: 'resolved',
    referent,
    resolution_reason: reason,
    should_ask_clarification: false,
    clarification_question: '',
    safety_flags: [],
  };
}

function clarify(reason: string, memory: SessionMemoryState, isFollowUp = true): FollowUpResolverOutput {
  return {
    is_follow_up: isFollowUp,
    resolution_status: 'needs_clarification',
    referent: emptyReferent(),
    resolution_reason: reason,
    should_ask_clarification: true,
    clarification_question: clarificationFor(reason, memory),
    safety_flags: [],
  };
}

function blocked(reason: string, safetyFlags: string[], isFollowUp = true): FollowUpResolverOutput {
  return {
    is_follow_up: isFollowUp,
    resolution_status: 'blocked_by_safety',
    referent: emptyReferent(),
    resolution_reason: reason,
    should_ask_clarification: true,
    clarification_question: 'Không đủ cơ sở an toàn để dùng ngữ cảnh trước. Bạn hãy nêu rõ câu hỏi độc lập.',
    safety_flags: safetyFlags,
  };
}

function conflict(reason: string): FollowUpResolverOutput {
  return {
    is_follow_up: true,
    resolution_status: 'conflict_detected',
    referent: emptyReferent(),
    resolution_reason: reason,
    should_ask_clarification: true,
    clarification_question: 'Bạn đang chuyển hoặc chỉnh lại chủ đề. Hãy nêu câu hỏi đầy đủ theo chủ đề mới.',
    safety_flags: ['memory_conflict'],
  };
}

function notFollowUp(reason: string): FollowUpResolverOutput {
  return {
    is_follow_up: false,
    resolution_status: 'not_follow_up',
    referent: emptyReferent(),
    resolution_reason: reason,
    should_ask_clarification: false,
    clarification_question: '',
    safety_flags: [],
  };
}

export function resolveFollowUpQuery(input: FollowUpResolverInput): FollowUpResolverOutput {
  const query = input.query;
  const q = normalize(query);
  const memory = input.memory;
  const intent = input.intent_result ?? classifyIntent(query);
  const hasAnchor = hasExplicitHistoricalAnchor(q);
  const hasPronominal = hasPronominalFollowUpReference(q);

  if (intent.intent === 'negative_gap' || intent.intent === 'out_of_scope' || intent.intent === 'hallucination_trap') {
    return blocked(`query_intent_${intent.intent}_must_not_use_memory`, [intent.intent], hasFollowUpReference(q));
  }

  if (/^nói\s+(chung|rõ\s*hơn)\b/i.test(q) && hasAnchor && !hasPronominal) {
    return notFollowUp('word_boundary_or_local_anchor_nói_query_is_standalone');
  }

  if ((memory.safety.last_out_of_scope || memory.last_answer_status === 'out_of_scope') && hasAnchor && !hasPronominal) {
    return notFollowUp('explicit_historical_anchor_after_oos_current_query_wins');
  }

  if (hasCorrectionOrShift(q) && hasAnchor) {
    return notFollowUp('explicit_topic_switch_current_query_wins');
  }

  if (memory.safety.last_negative_gap || memory.last_answer_status === 'insufficient_data') {
    return blocked('last_turn_negative_gap_or_insufficient_data_safety_block', ['last_negative_gap'], hasFollowUpReference(q));
  }
  if (memory.safety.last_out_of_scope || memory.last_answer_status === 'out_of_scope') {
    return blocked('last_turn_out_of_scope_safety_block', ['last_out_of_scope'], hasFollowUpReference(q));
  }
  if (memory.safety.memory_conflict) {
    return conflict('memory_conflict_flag_present');
  }

  if (hasCorrectionOrShift(q)) return conflict('explicit_topic_shift_or_correction');

  const isFollowUp = hasFollowUpReference(q)
    || intent.intent === 'follow_up'
    || hasSummaryFollowUp(q)
    || hasChronologyFollowUp(q)
    || hasInstructionalFollowUp(q)
    || hasCitationFollowUp(q)
    || hasComparisonFollowUp(q)
    || hasEntityRoleFollowUp(q)
    || /tổ\s*chức\s*(đó|này)|lực\s*lượng\s*(đó|này)|phong\s*trào\s*(đó|này)|mối\s*quan\s*hệ\s*đó|hai\s*(sự\s*kiện|văn\s*kiện|mốc)\s*đó/i.test(q);

  // Patch 9C3R: If follow-up detected only from summary/instructional/generic keywords
  // BUT query contains an explicit historical anchor (event name, year, treaty, etc.),
  // AND no actual pronominal/demonstrative reference is present,
  // treat as standalone new-topic query — not a follow-up needing memory.
  // This prevents over-blocking answerable queries like:
  //   "Tóm tắt mốc chính của Cách mạng tháng Tám 1945."
  //   "Dựa vào đâu để nói Mậu Thân 1968 có ý nghĩa chính trị?"
  // Key: hasPronominalFollowUpReference checks ONLY pronominal/demonstrative refs
  // (sự kiện đó, nó, cái đó), NOT summary keywords (tóm tắt, tiếp theo, tóm lại).
  if (isFollowUp
    && hasExplicitHistoricalAnchor(q)
    && intent.intent !== 'follow_up'
    && !hasPronominal
    && !hasChronologyFollowUp(q)
    && !/tổ\s*chức\s*đó|mối\s*quan\s*hệ\s*đó|hai\s*(sự\s*kiện|văn\s*kiện|mốc)\s*đó/i.test(q)) {
    return notFollowUp('summary_or_instructional_keyword_with_explicit_historical_anchor_is_standalone');
  }

  if (!isFollowUp) {
    if (hasExplicitNewTopic(q)) {
      return notFollowUp('explicit_new_topic_query_does_not_need_memory');
    }
    return notFollowUp('query_has_no_follow_up_reference');
  }

  if (hasChronologyFollowUp(q)) return clarify('chronology_follow_up_requires_explicit_period_or_event', memory);

  if (/hai\s*(sự\s*kiện|văn\s*kiện|mốc)\s*đó/i.test(q)) {
    const candidates = highConfidenceEvents(memory);
    if (candidates.length === 2) {
      return resolved({
        text: candidates.map(item => item.event_label).join(' và '),
        type: 'comparison',
        confidence: Math.min(candidates[0].confidence, candidates[1].confidence),
        source: 'active_event',
      }, 'exactly_two_high_confidence_events');
    }
    if (memory.active_comparison_targets.length >= 2) {
      return resolved({
        text: memory.active_comparison_targets.slice(0, 2).join(' và '),
        type: 'comparison',
        confidence: memory.active_topic?.confidence ?? 0.8,
        source: 'active_topic',
      }, 'two_item_reference_uses_active_comparison_targets');
    }
    return clarify('two_item_reference_without_exactly_two_candidates', memory);
  }

  if (/giai\s*đoạn\s*đó/i.test(q)) {
    if (memory.active_period.label !== 'unknown' && memory.active_period.label !== 'cross_period' && memory.active_period.confidence >= 0.75) {
      return resolved({
        text: memory.active_period.label,
        type: 'period',
        confidence: memory.active_period.confidence,
        source: 'active_period',
      }, 'active_period_high_confidence');
    }
    return clarify('active_period_missing_low_confidence_or_cross_period', memory);
  }

  if (/tổ\s*chức\s*đó/i.test(q)) {
    const orgs = highConfidenceOrganizations(memory);
    if (orgs.length === 1) {
      return resolved({ text: orgs[0].text, type: 'entity', confidence: orgs[0].confidence, source: 'active_entity' }, 'single_high_confidence_organization');
    }
    return clarify('organization_reference_ambiguous_or_missing', memory);
  }

  if (/tổ\s*chức\s*này|lực\s*lượng\s*(đó|này)/i.test(q)) {
    const orgs = highConfidenceOrganizations(memory);
    if (orgs.length >= 1) {
      return resolved({ text: orgs[0].text, type: 'entity', confidence: orgs[0].confidence, source: 'active_entity' }, 'organization_or_force_reference_uses_primary_organization');
    }
    return clarify('organization_or_force_reference_missing', memory);
  }

  if (/ông\s*(ấy|ta)|bà\s*ấy|người\s*đó/i.test(q)) {
    const persons = highConfidencePersons(memory);
    if (persons.length >= 1) {
      return resolved({ text: persons[0].text, type: 'entity', confidence: persons[0].confidence, source: 'active_entity' }, 'person_reference_uses_primary_person');
    }
    if (memory.active_topic && memory.active_topic.confidence >= 0.75) {
      return resolved({ text: memory.active_topic.label, type: 'topic', confidence: memory.active_topic.confidence, source: 'active_topic' }, 'person_reference_falls_back_to_active_topic');
    }
    return clarify('person_reference_missing', memory);
  }

  if (/mối\s*quan\s*hệ\s*đó/i.test(q)) {
    const topic = memory.active_topic;
    if (topic && topic.confidence >= 0.75) {
      return resolved({ text: topic.label, type: topic.label.includes(' vs ') ? 'comparison' : 'topic', confidence: topic.confidence, source: 'active_topic' }, 'active_topic_relationship_reference');
    }
    return clarify('relationship_reference_without_active_topic', memory);
  }

  if (hasSummaryFollowUp(q)) {
    const comp = comparisonTopic(memory);
    if (comp) return resolved(comp, 'summary_followup_uses_active_comparison_topic');
    if (memory.active_topic && memory.active_topic.confidence >= 0.75 && ['answerable', 'partially_answerable'].includes(String(memory.last_answer_status))) {
      return resolved({ text: memory.active_topic.label, type: 'topic', confidence: memory.active_topic.confidence, source: 'active_topic' }, 'summary_followup_uses_active_topic');
    }
    if (memory.last_answer_summary && memory.conversation_focus.confidence >= 0.75) {
      return resolved({ text: memory.last_answer_summary, type: 'topic', confidence: memory.conversation_focus.confidence, source: 'last_answer_summary' }, 'summary_followup_uses_last_answer_summary');
    }
    return clarify('summary_followup_without_safe_answerable_memory', memory);
  }

  if (hasCitationFollowUp(q)) {
    if (memory.active_topic && memory.active_topic.confidence >= 0.65) {
      return resolved({ text: memory.last_citation_focus ?? memory.active_topic.label, type: 'topic', confidence: memory.active_topic.confidence, source: 'active_topic' }, 'citation_followup_uses_active_topic_or_citation_focus');
    }
    const docs = highConfidenceDocuments(memory);
    if (docs.length >= 1) return resolved({ text: docs[0].text, type: 'entity', confidence: docs[0].confidence, source: 'active_entity' }, 'citation_followup_uses_primary_document');
    const events = highConfidenceEvents(memory);
    if (events.length === 1) return resolved({ text: events[0].event_label, type: 'event', confidence: events[0].confidence, source: 'active_event' }, 'citation_followup_uses_single_event');
    return clarify('citation_followup_without_safe_topic', memory);
  }

  if (hasComparisonFollowUp(q)) {
    const comp = comparisonTopic(memory);
    if (comp) return resolved(comp, 'comparison_ellipsis_uses_active_comparison_topic');
    if (memory.active_comparison_targets.length >= 2) {
      return resolved({
        text: memory.active_comparison_targets.slice(0, 2).join(' và '),
        type: 'comparison',
        confidence: memory.active_topic?.confidence ?? 0.8,
        source: 'active_topic',
      }, 'comparison_ellipsis_uses_comparison_targets');
    }
    if (memory.active_topic && memory.active_topic.confidence >= 0.75) {
      return resolved({ text: memory.active_topic.label, type: 'topic', confidence: memory.active_topic.confidence, source: 'active_topic' }, 'comparison_followup_uses_active_topic');
    }
    return clarify('comparison_followup_without_targets', memory);
  }

  if (/nhận\s*định\s*đó/i.test(q)) {
    if (memory.active_topic && memory.active_topic.confidence >= 0.65) {
      return resolved({ text: memory.active_topic.label, type: 'topic', confidence: memory.active_topic.confidence, source: 'active_topic' }, 'claim_reference_uses_active_topic');
    }
    const events = highConfidenceEvents(memory);
    if (events.length === 1) return resolved({ text: events[0].event_label, type: 'event', confidence: events[0].confidence, source: 'active_event' }, 'claim_reference_uses_single_active_event');
    return clarify('claim_reference_without_safe_topic', memory);
  }

  if (hasInstructionalFollowUp(q)) {
    const comp = comparisonTopic(memory);
    if (comp) return resolved(comp, 'instructional_followup_uses_active_comparison_topic');
    if (memory.active_topic && memory.active_topic.confidence >= 0.65 && ['answerable', 'partially_answerable'].includes(String(memory.last_answer_status))) {
      return resolved({ text: memory.active_topic.label, type: 'topic', confidence: memory.active_topic.confidence, source: 'active_topic' }, 'instructional_followup_uses_active_topic');
    }
    const events = highConfidenceEvents(memory);
    if (events.length === 1) return resolved({ text: events[0].event_label, type: 'event', confidence: events[0].confidence, source: 'active_event' }, 'instructional_followup_uses_single_active_event');
    return clarify('instructional_followup_without_safe_answerable_memory', memory);
  }

  if (/sự\s*kiện\s*đó/i.test(q)) {
    const events = highConfidenceEvents(memory);
    if (events.length === 1) {
      if (isAmbiguousCompoundEventLabel(events[0].event_label)) {
        return clarify('ambiguous_event_reference_compound_event_label', memory);
      }
      return resolved({ text: events[0].event_label, type: 'event', confidence: events[0].confidence, source: 'active_event' }, 'single_high_confidence_active_event');
    }
    if (memory.active_topic && memory.active_topic.confidence >= 0.75) {
      return resolved({ text: memory.active_topic.label, type: 'topic', confidence: memory.active_topic.confidence, source: 'active_topic' }, 'event_reference_falls_back_to_active_topic');
    }
    const docs = highConfidenceDocuments(memory);
    if (docs.length === 1) return resolved({ text: docs[0].text, type: 'entity', confidence: docs[0].confidence, source: 'active_entity' }, 'event_reference_falls_back_to_document');
    return clarify('ambiguous_event_reference_multiple_or_missing_candidates', memory);
  }

  if (/(nó|cái\s*đó|điều\s*này|điều\s*đó)/i.test(q)) {
    if (/vai\s*trò/i.test(q)) {
      const orgs = highConfidenceOrganizations(memory);
      if (orgs.length === 1) {
        return resolved({ text: orgs[0].text, type: 'entity', confidence: orgs[0].confidence, source: 'active_entity' }, 'role_followup_prefers_single_active_organization');
      }
    }
    const comp = comparisonTopic(memory);
    if (comp && /khác|so\s*sánh|điểm\s*khác|toàn\s*bộ/i.test(q)) return resolved(comp, 'comparison_followup_uses_active_topic');
    if (comp) return clarify('ambiguous_comparison_singular_reference', memory);
    const events = highConfidenceEvents(memory);
    if (events.length === 1) {
      if (isAmbiguousCompoundEventLabel(events[0].event_label) && /sự\s*kiện\s*đó/i.test(q)) {
        return clarify('ambiguous_event_reference_compound_event_label', memory);
      }
      return resolved({ text: events[0].event_label, type: 'event', confidence: events[0].confidence, source: 'active_event' }, 'single_high_confidence_active_event');
    }
    if (events.length > 1 && /sự\s*kiện\s*đó/i.test(q)) return clarify('ambiguous_event_reference_multiple_candidates', memory);
    if (memory.active_topic && memory.active_topic.confidence >= 0.75) {
      return resolved({ text: memory.active_topic.label, type: comp ? 'comparison' : 'topic', confidence: memory.active_topic.confidence, source: 'active_topic' }, 'active_topic_high_confidence');
    }
    const entities = highConfidenceEntities(memory);
    if (entities.length === 1) return resolved({ text: entities[0].text, type: 'entity', confidence: entities[0].confidence, source: 'active_entity' }, 'single_high_confidence_active_entity');
    return clarify('generic_reference_without_single_high_confidence_candidate', memory);
  }

  return clarify('follow_up_reference_not_resolved_by_policy', memory);
}
