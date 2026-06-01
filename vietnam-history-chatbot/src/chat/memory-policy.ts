/**
 * Session Memory Policy — Stage 8C1 (No-API, deterministic)
 *
 * Defines conservative write/read/safety/reset rules for conversational
 * memory. This module never rewrites a query and never calls external APIs.
 */

import type { AnswerIntent, AnswerStatus } from './answer-planner.js';
import type {
  FollowUpResolution,
  MemorySignals,
  MemorySafetyState,
  MemoryTurnInput,
  SessionMemoryState,
} from './session-memory.js';

export interface MemoryPolicyDecision {
  allow: boolean;
  reason: string;
  safety_state: MemorySafetyState;
  confidence_threshold: number;
  warnings: string[];
}

export interface MemoryResetDecision {
  should_reset: boolean;
  should_decay: boolean;
  reason: string;
  target?: 'all' | 'topic' | 'period' | 'entities';
}

const NEGATIVE_OR_UNSAFE_INTENTS: AnswerIntent[] = ['negative_gap', 'out_of_scope', 'unclear', 'hallucination_trap'];
const UNSAFE_ANSWER_STATUSES: AnswerStatus[] = ['insufficient_data', 'needs_clarification', 'out_of_scope'];

export const MEMORY_POLICY = {
  write: {
    min_confidence: 0.72,
    allowed_answer_statuses: ['answerable', 'partially_answerable'] as AnswerStatus[],
    blocked_intents: ['negative_gap', 'out_of_scope', 'unclear'] as AnswerIntent[],
    unresolved_follow_up_blocked: true,
  },
  read: {
    min_topic_confidence: 0.75,
    min_entity_confidence: 0.75,
    max_turn_age: 5,
    clarify_below_confidence: 0.6,
  },
  decay: {
    unrelated_turn_decay: 0.1,
    stale_after_turns: 5,
    clarify_below_confidence: 0.6,
  },
};

function normalize(text: string): string {
  return text.toLowerCase().normalize('NFKC').trim();
}

export function hasFollowUpReference(query: string): boolean {
  const q = normalize(query);
  // Patch 9C3R: "nó" must be standalone (not inside "nói"/"nóng"/"nón").
  // JS \b treats Vietnamese "ó" as non-word char, so \bnó\b still matches "nói".
  // Use whitespace/punctuation/start/end boundaries instead.
  const hasStandaloneNo = /(?:^|\s)nó(?:\s|[,;.?!]|$)/i.test(q);
  return hasStandaloneNo
    || /sự\s*kiện\s*đó|cái\s*đó|điều\s*này|điều\s*đó|điều\s*ấy|nhận\s*định\s*đó|kết\s*luận\s*đó|chi\s*tiết\s*này|giai\s*đoạn\s*đó|hai\s*(sự\s*kiện|văn\s*kiện|mốc)\s*đó|tổ\s*chức\s*(đó|này)|lực\s*lượng\s*(đó|này)|phong\s*trào\s*(đó|này)|hiệp\s*định\s*đó|văn\s*kiện\s*đó|mối\s*quan\s*hệ\s*đó|vai\s*trò\s*đó|người\s*đó|ông\s*(ấy|ta)|bà\s*ấy|sau\s*đó|tiếp\s*theo|tóm\s*lại|tóm\s*tắt|nhắc\s*lại|vậy\s*thì|như\s*vậy|nói\s*kỹ\s*hơn|giải\s*thích\s*thêm|còn\s+sau\s+đó|còn\s+giai\s*đoạn\s+sau|điểm\s*khác\s*nhau|toàn\s*bộ\s*khác\s*biệt|so\s*sánh\s*thêm|cái\s*nào|nguồn\s*(nào|gì)|dựa\s*vào\s*(đâu|nguồn|tài\s*liệu)|căn\s*cứ\s*(nào|gì)|bằng\s*chứng\s*(nào|gì)|dẫn\s*chứng\s*(nào|gì)|theo\s*nguồn\s*nào|có\s*nguồn\s*nào/i.test(q);
}

/** Patch 9C3R: Check for pronominal/demonstrative follow-up references only
 * (not summary/instruction keywords like "tóm tắt" or "tiếp theo").
 * Used by followup-resolver to distinguish genuine follow-up pronouns from
 * standalone queries that happen to contain summary keywords. */
export function hasPronominalFollowUpReference(query: string): boolean {
  const q = normalize(query);
  const hasStandaloneNo = /(?:^|\s)nó(?:\s|[,;.?!]|$)/i.test(q);
  return hasStandaloneNo
    || /sự\s*kiện\s*đó|cái\s*đó|điều\s*này|điều\s*đó|điều\s*ấy|nhận\s*định\s*đó|kết\s*luận\s*đó|chi\s*tiết\s*này|giai\s*đoạn\s*đó|hai\s*(sự\s*kiện|văn\s*kiện|mốc)\s*đó|tổ\s*chức\s*(đó|này)|lực\s*lượng\s*(đó|này)|phong\s*trào\s*(đó|này)|hiệp\s*định\s*đó|văn\s*kiện\s*đó|mối\s*quan\s*hệ\s*đó|vai\s*trò\s*đó|người\s*đó|ông\s*(ấy|ta)|bà\s*ấy|sau\s*đó|vậy\s*thì|như\s*vậy|còn\s+sau\s+đó|còn\s+giai\s*đoạn\s+sau|cái\s*nào/i.test(q);
}

export function hasExplicitHistoricalAnchor(query: string): boolean {
  const q = normalize(query);
  // Patch 9C3R: Expanded with corpus-scope terms to prevent false follow-up classification
  return /\b(18\d{2}|19\d{2}|20\d{2})\b|điện\s*biên\s*phủ|gen[eè]ve|paris|việt\s*minh|cách\s*mạng\s*tháng\s*tám|mậu\s*thân|đồng\s*khởi|tổng\s*tiến\s*công|xuân\s*1975|hiệp\s*định|hội\s*nghị|chiến\s*dịch|cương\s*lĩnh|luận\s*cương|hậu\s*phương|tiền\s*tuyến|xô\s*viết|kháng\s*chiến|tuyên\s*ngôn|phong\s*trào|bảo\s*đại|cần\s*vương|trường\s*sơn|tây\s*nguyên|nguyễn\s*ái\s*quốc|hồ\s*chí\s*minh|trần\s*phú|ngô\s*đình\s*diệm|võ\s*nguyên\s*giáp|đảng\s*cộng\s*sản|mặt\s*trận\s*dân\s*tộc\s*giải\s*phóng|miền\s*bắc|miền\s*nam|giải\s*phóng|đổi\s*mới|toàn\s*quốc/i.test(q);
}

export function hasExplicitTopicShift(query: string): boolean {
  const q = normalize(query);
  return /chủ\s*đề\s*khác|chuyển\s*sang|không\s*phải|ý\s*tôi\s*là|quên\s*(cái|phần|chủ\s*đề)\s*trước|bỏ\s*qua\s*(cái|phần)\s*trước/i.test(q);
}

export function detectMemoryReset(query: string): MemoryResetDecision {
  const q = normalize(query);
  if (/quên\s*(cái|phần|chủ\s*đề)\s*trước|reset|xóa\s*ngữ\s*cảnh|bỏ\s*qua\s*(cái|phần)\s*trước/i.test(q)) {
    return { should_reset: true, should_decay: false, reason: 'user_requested_memory_reset', target: 'all' };
  }
  if (/chuyển\s*sang\s*giai\s*đoạn|chủ\s*đề\s*khác/i.test(q)) {
    return { should_reset: true, should_decay: false, reason: 'explicit_topic_shift', target: 'topic' };
  }
  if (/không\s*phải|ý\s*tôi\s*là/i.test(q)) {
    return { should_reset: false, should_decay: true, reason: 'correction_or_topic_refinement', target: 'topic' };
  }
  return { should_reset: false, should_decay: false, reason: 'no_reset_signal' };
}

export function evaluateMemoryWritePolicy(
  memory: SessionMemoryState,
  signals: MemorySignals,
  input: MemoryTurnInput,
): MemoryPolicyDecision {
  const warnings: string[] = [];
  const intent = signals.intent;
  const answerStatus = signals.answer_status;

  if (NEGATIVE_OR_UNSAFE_INTENTS.includes(intent)) {
    return {
      allow: false,
      reason: `blocked_intent:${intent}`,
      safety_state: intent === 'negative_gap' ? 'memory_unavailable' : intent === 'out_of_scope' ? 'memory_unavailable' : 'memory_should_ask_clarification',
      confidence_threshold: MEMORY_POLICY.write.min_confidence,
      warnings,
    };
  }

  if (intent === 'follow_up' && signals.unresolved_follow_up) {
    return {
      allow: false,
      reason: 'blocked_unresolved_follow_up',
      safety_state: 'memory_should_ask_clarification',
      confidence_threshold: MEMORY_POLICY.write.min_confidence,
      warnings,
    };
  }

  if (UNSAFE_ANSWER_STATUSES.includes(answerStatus)) {
    return {
      allow: false,
      reason: `blocked_answer_status:${answerStatus}`,
      safety_state: 'memory_unavailable',
      confidence_threshold: MEMORY_POLICY.write.min_confidence,
      warnings,
    };
  }

  if (!signals.topic && signals.entities.length === 0 && signals.events.length === 0) {
    return {
      allow: false,
      reason: 'no_storable_topic_or_entity',
      safety_state: 'memory_low_confidence',
      confidence_threshold: MEMORY_POLICY.write.min_confidence,
      warnings,
    };
  }

  if (signals.confidence < MEMORY_POLICY.write.min_confidence) {
    // Patch 11A-FIX: Bypass confidence threshold when high-confidence entities
    // or topic are clearly detected. The composite confidence can be artificially
    // low due to citation_weak penalty even though entities like 'Nguyễn Ái Quốc',
    // 'Việt Minh' are detected at 0.9 confidence from explicit query anchors.
    const hasHighConfidenceEntity = signals.entities.some(e => e.confidence >= 0.8);
    const hasDetectedTopic = signals.topic !== null && signals.topic.confidence >= 0.7;
    if (hasHighConfidenceEntity || hasDetectedTopic) {
      warnings.push('confidence_below_threshold_but_high_confidence_entities_bypass');
      // Allow — fall through
    } else {
      return {
        allow: false,
        reason: 'memory_signal_confidence_below_threshold',
        safety_state: 'memory_low_confidence',
        confidence_threshold: MEMORY_POLICY.write.min_confidence,
        warnings,
      };
    }
  }

  if (signals.citation_weak && input.require_safe_citation !== false) {
    // Patch 11A-FIX: Allow focus-tracking memory writes when entities are clearly
    // detected from explicit historical anchors (high confidence ≥ 0.8), even if
    // citations are weak. This is safe because we store conversational focus
    // (what topic is being discussed), not factual claims.
    // Without this, follow-up queries fail: memory has no active_topic/entities
    // to resolve references against, causing needs_clarification → early return.
    const hasHighConfidenceEntity = signals.entities.some(e => e.confidence >= 0.8);
    const hasDetectedTopic = signals.topic !== null && signals.topic.confidence >= 0.7;
    if (hasHighConfidenceEntity || hasDetectedTopic) {
      warnings.push('citation_weak_but_focus_tracking_allowed_due_to_high_confidence_entity');
      // Allow — fall through to success below
    } else {
      warnings.push('citation_context_weak_memory_not_written');
      return {
        allow: false,
        reason: 'citation_or_context_too_weak',
        safety_state: 'memory_low_confidence',
        confidence_threshold: MEMORY_POLICY.write.min_confidence,
        warnings,
      };
    }
  }

  if (memory.safety.memory_conflict) {
    return {
      allow: false,
      reason: 'existing_memory_conflict',
      safety_state: 'memory_conflict',
      confidence_threshold: MEMORY_POLICY.write.min_confidence,
      warnings,
    };
  }

  return {
    allow: true,
    reason: 'safe_storable_turn',
    safety_state: 'memory_safe_to_use',
    confidence_threshold: MEMORY_POLICY.write.min_confidence,
    warnings,
  };
}

export function evaluateMemoryReadPolicy(
  query: string,
  memory: SessionMemoryState,
  resolution?: FollowUpResolution,
): MemoryPolicyDecision {
  const warnings: string[] = [];
  const q = normalize(query);

  if (!hasFollowUpReference(q)) {
    return {
      allow: false,
      reason: 'query_has_no_follow_up_reference',
      safety_state: 'memory_unavailable',
      confidence_threshold: MEMORY_POLICY.read.min_topic_confidence,
      warnings,
    };
  }

  if (hasExplicitHistoricalAnchor(q) && !/^(nó|sự\s*kiện\s*đó|cái\s*đó|giai\s*đoạn\s*đó|điều\s*này|điều\s*đó)/i.test(q)) {
    return {
      allow: false,
      reason: 'query_contains_explicit_new_anchor',
      safety_state: 'memory_conflict',
      confidence_threshold: MEMORY_POLICY.read.min_topic_confidence,
      warnings,
    };
  }

  if (!memory.safety.memory_reliable || memory.safety.last_negative_gap || memory.safety.last_out_of_scope) {
    return {
      allow: false,
      reason: 'memory_last_turn_not_safe',
      safety_state: 'memory_should_ask_clarification',
      confidence_threshold: MEMORY_POLICY.read.min_topic_confidence,
      warnings,
    };
  }

  if (memory.safety.memory_conflict) {
    return {
      allow: false,
      reason: 'memory_conflict_present',
      safety_state: 'memory_conflict',
      confidence_threshold: MEMORY_POLICY.read.min_topic_confidence,
      warnings,
    };
  }

  const bestConfidence = Math.max(
    memory.active_topic?.confidence ?? 0,
    memory.active_events[0]?.confidence ?? 0,
    memory.active_entities[0]?.confidence ?? 0,
    memory.active_period?.confidence ?? 0,
  );

  if (bestConfidence < MEMORY_POLICY.read.min_topic_confidence) {
    return {
      allow: false,
      reason: 'memory_confidence_too_low',
      safety_state: bestConfidence < MEMORY_POLICY.decay.clarify_below_confidence ? 'memory_should_ask_clarification' : 'memory_low_confidence',
      confidence_threshold: MEMORY_POLICY.read.min_topic_confidence,
      warnings,
    };
  }

  if (resolution && !resolution.resolved) {
    return {
      allow: false,
      reason: resolution.reason,
      safety_state: 'memory_should_ask_clarification',
      confidence_threshold: MEMORY_POLICY.read.min_topic_confidence,
      warnings,
    };
  }

  return {
    allow: true,
    reason: 'follow_up_reference_resolved_from_memory',
    safety_state: 'memory_safe_to_use',
    confidence_threshold: MEMORY_POLICY.read.min_topic_confidence,
    warnings,
  };
}

export function evaluateMemorySafetyPolicy(memory: SessionMemoryState): MemorySafetyState {
  if (memory.safety.memory_conflict) return 'memory_conflict';
  if (memory.safety.needs_clarification) return 'memory_should_ask_clarification';
  if (memory.safety.last_negative_gap || memory.safety.last_out_of_scope || (memory.last_user_intent === 'hallucination_trap')) return 'memory_unavailable';
  if (!memory.safety.memory_reliable) return 'memory_low_confidence';
  return 'memory_safe_to_use';
}

export function shouldDecayMemory(memory: SessionMemoryState, currentTurn: number): boolean {
  const lastUpdated = Math.max(
    memory.active_topic?.last_updated_turn ?? 0,
    memory.active_period?.last_updated_turn ?? 0,
    ...memory.active_entities.map(entity => entity.last_updated_turn),
    ...memory.active_events.map(event => event.last_updated_turn),
  );
  return currentTurn - lastUpdated > 0;
}
