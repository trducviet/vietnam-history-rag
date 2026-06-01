/**
 * Query Rewriter — Stage 8C2 (No-API, deterministic)
 *
 * Converts resolved follow-up questions into standalone retrieval queries.
 * It never fabricates referents and never rewrites safety-blocked requests
 * into answerable historical questions.
 */

import type { AnswerIntent, IntentClassification } from './answer-planner.js';
import { classifyIntent } from './answer-planner.js';
import type { SessionMemoryState } from './session-memory.js';
import type { FollowUpResolverOutput } from './followup-resolver.js';

export type RewriteStatus = 'rewritten' | 'not_needed' | 'needs_clarification' | 'blocked_by_safety' | 'failed';

export interface QueryRewriteOutput {
  rewrite_status: RewriteStatus;
  original_query: string;
  rewritten_query: string;
  rewrite_confidence: number;
  rewrite_reason: string;
  used_memory_fields: string[];
  preserved_intent: AnswerIntent;
  target_intent: AnswerIntent;
  safety_flags: string[];
  clarification_question: string;
}

function normalize(text: string): string {
  return text.toLowerCase().normalize('NFKC').trim();
}

function cleanQuestion(text: string): string {
  const trimmed = text.replace(/\s+/g, ' ').trim();
  return trimmed.endsWith('?') ? trimmed : `${trimmed}?`;
}

function splitComparison(label: string): string[] {
  return label.split(/\s+vs\s+| và /i).map(item => item.trim()).filter(Boolean);
}

function explicitCounterpart(query: string): string {
  const direct = query.match(/(?:với|so\s*với)\s+(.+?)(?:\?|$)/i);
  if (direct?.[1]) return direct[1].trim().replace(/\.$/, '');
  const different = query.match(/khác\s+(.+?)\s+(?:thế\s*nào|như\s*thế\s*nào|ở\s*chỗ\s*nào|\?|$)/i);
  return different?.[1]?.trim().replace(/\.$/, '') ?? '';
}

function canonicalizeCounterpart(text: string): string {
  const q = normalize(text);
  if (/điện\s*biên\s*phủ\s*trên\s*không/i.test(q)) return 'Điện Biên Phủ trên không 1972';
  if (/gen[eè]ve/i.test(q)) return 'Genève 1954';
  if (/paris/i.test(q)) return 'Paris 1973';
  if (/xuân\s*1975|tổng\s*tiến\s*công/i.test(q)) return 'Tổng tiến công mùa Xuân 1975';
  return text.trim();
}

function intentForRewrite(query: string, fallback: AnswerIntent): AnswerIntent {
  const q = normalize(query);
  if (/khác|so\s*sánh|phân\s*biệt/i.test(q)) return 'comparison';
  if (/điểm\s*khác\s*nhau|toàn\s*bộ\s*khác\s*biệt|cái\s*nào|giống\s*nhau|quan\s*trọng\s*hơn|ý\s*nghĩa\s*lớn\s*hơn/i.test(q)) return 'comparison';
  if (/nguồn|dựa\s*vào|tài\s*liệu|căn\s*cứ|bằng\s*chứng|dẫn\s*chứng|theo\s*nguồn/i.test(q)) return 'citation_source';
  if (/vai\s*trò|tổ\s*chức\s*đó/i.test(q)) return 'entity_role';
  if (/tóm\s*lại|tóm\s*tắt|nhắc\s*lại|nói\s*ngắn\s*gọn/i.test(q)) return 'why_meaning';
  if (/ý\s*nghĩa|tác\s*động|bước\s*ngoặt/i.test(q)) return 'why_meaning';
  if (/timeline|mốc|diễn\s*biến|quá\s*trình/i.test(q)) return 'timeline';
  return fallback === 'follow_up' || fallback === 'unclear' ? 'fact' : fallback;
}

function notNeeded(query: string, intent: AnswerIntent, reason = 'query_is_not_follow_up'): QueryRewriteOutput {
  return {
    rewrite_status: 'not_needed',
    original_query: query,
    rewritten_query: query,
    rewrite_confidence: 1,
    rewrite_reason: reason,
    used_memory_fields: [],
    preserved_intent: intent,
    target_intent: intent,
    safety_flags: [],
    clarification_question: '',
  };
}

function clarification(query: string, intent: AnswerIntent, resolution: FollowUpResolverOutput): QueryRewriteOutput {
  return {
    rewrite_status: 'needs_clarification',
    original_query: query,
    rewritten_query: query,
    rewrite_confidence: 0,
    rewrite_reason: resolution.resolution_reason,
    used_memory_fields: [],
    preserved_intent: intent,
    target_intent: intent,
    safety_flags: resolution.safety_flags,
    clarification_question: resolution.clarification_question,
  };
}

function blocked(query: string, intent: AnswerIntent, resolution: FollowUpResolverOutput): QueryRewriteOutput {
  return {
    rewrite_status: 'blocked_by_safety',
    original_query: query,
    rewritten_query: query,
    rewrite_confidence: 0,
    rewrite_reason: resolution.resolution_reason,
    used_memory_fields: [],
    preserved_intent: intent,
    target_intent: intent,
    safety_flags: resolution.safety_flags.length ? resolution.safety_flags : ['blocked_by_safety'],
    clarification_question: resolution.clarification_question,
  };
}

function rewriteWithReferent(query: string, referent: string, memory: SessionMemoryState, targetIntent: AnswerIntent): string {
  const q = normalize(query);
  const counterpart = canonicalizeCounterpart(explicitCounterpart(query));
  const comparisonSides = splitComparison(referent);
  const primaryDocument = memory.active_primary_document?.text ?? memory.active_documents[0]?.text ?? '';

  if (/trong\s+sự\s*kiện\s*đó/i.test(q)) {
    return cleanQuestion(query.replace(/sự\s*kiện\s*đó/i, referent));
  }

  if (/tổ\s*chức\s*(đó|này)|lực\s*lượng\s*(đó|này)/i.test(q) && memory.active_events.length > 0) {
    return cleanQuestion(`${referent} có vai trò gì trong ${memory.active_events[0].event_label}`);
  }

  if (/mối\s*quan\s*hệ\s*đó/i.test(q)) {
    return cleanQuestion(`Mối quan hệ giữa ${referent} thể hiện thế nào`);
  }

  if (/nguồn|dựa\s*vào|tài\s*liệu|căn\s*cứ|chứng\s*minh|bằng\s*chứng|dẫn\s*chứng|theo\s*nguồn|nhận\s*định\s*đó|kết\s*luận\s*đó|chi\s*tiết\s*này/i.test(q)) {
    return cleanQuestion(`Nguồn nào cho thấy ${referent}`);
  }

  if (/tóm\s*lại|tóm\s*tắt|điểm\s*khác\s*nhau\s*chính|nhắc\s*lại|nói\s*ngắn\s*gọn/i.test(q)) {
    if (comparisonSides.length >= 2) {
      return cleanQuestion(`Tóm lại điểm khác nhau chính giữa ${comparisonSides[0]} và ${comparisonSides[1]} là gì`);
    }
    return cleanQuestion(`Tóm lại ý chính về ${referent} là gì`);
  }

  if (/trả\s*lời\s*trước|kết\s*luận\s*chính/i.test(q)) return cleanQuestion(`Kết luận chính về ${referent} là gì`);
  if (/lập\s*bảng/i.test(q)) return cleanQuestion(`Lập bảng hai ý chính về ${referent}`);
  if (/timeline/i.test(q)) return cleanQuestion(`Timeline các mốc liên quan đến ${referent}`);
  if (/đừng\s*lan\s*man|luận\s*điểm|bằng\s*chứng/i.test(q)) return cleanQuestion(`Nêu 3 luận điểm có bằng chứng về ${referent}`);

  if (/khác|so\s*sánh|điểm\s*khác\s*nhau|toàn\s*bộ\s*khác\s*biệt|giống\s*nhau/i.test(q)) {
    if (counterpart) return cleanQuestion(`${referent} khác ${counterpart} như thế nào`);
    if (comparisonSides.length >= 2) return cleanQuestion(`${comparisonSides[0]} khác ${comparisonSides[1]} như thế nào`);
    return cleanQuestion(`${referent} khác nhau như thế nào`);
  }

  if (/cái\s*nào|quan\s*trọng\s*hơn|ý\s*nghĩa\s*lớn\s*hơn/i.test(q)) {
    if (comparisonSides.length >= 2) return cleanQuestion(`Giữa ${comparisonSides[0]} và ${comparisonSides[1]}, cái nào có ý nghĩa lớn hơn`);
    return cleanQuestion(`${referent}: ${query}`);
  }

  if (/có\s*phải\s*toàn\s*bộ/i.test(q)) {
    return cleanQuestion(`${referent} có phải toàn bộ Tổng tiến công mùa Xuân 1975 không`);
  }

  if (/vai\s*trò/i.test(q)) {
    if (/ông\s*(ấy|ta)|bà\s*ấy|người\s*đó/i.test(q) && primaryDocument) {
      return cleanQuestion(`${referent} có vai trò gì đối với ${primaryDocument}`);
    }
    return cleanQuestion(`${referent} có vai trò gì`);
  }
  if (/ý\s*nghĩa|tác\s*động|bước\s*ngoặt/i.test(q)) return cleanQuestion(`${referent} có ý nghĩa gì`);
  if (/quy\s*định|nội\s*dung/i.test(q)) return cleanQuestion(`${referent} quy định gì`);
  if (/gồm|mốc|diễn\s*biến|quá\s*trình/i.test(q)) return cleanQuestion(`${referent} gồm những mốc nào`);

  return cleanQuestion(`${referent}: ${query.replace(/^(?:nó|sự\s*kiện\s*đó|cái\s*đó|điều\s*này|điều\s*đó)(?:\s+|[,;.?!]|$)/i, '')}`);
}

export function rewriteFollowUpQuery(
  query: string,
  resolution: FollowUpResolverOutput,
  memory: SessionMemoryState,
  intentResult?: IntentClassification,
): QueryRewriteOutput {
  const intent = intentResult?.intent ?? classifyIntent(query).intent;

  if (intent === 'negative_gap' || intent === 'out_of_scope' || intent === 'hallucination_trap') {
    return blocked(query, intent, { ...resolution, resolution_reason: `query_intent_${intent}_must_not_be_rewritten`, safety_flags: [...resolution.safety_flags, intent] });
  }

  if (resolution.resolution_status === 'not_follow_up') return notNeeded(query, intent, resolution.resolution_reason);
  if (resolution.resolution_status === 'needs_clarification') return clarification(query, intent, resolution);
  if (resolution.resolution_status === 'blocked_by_safety' || resolution.resolution_status === 'conflict_detected') return blocked(query, intent, resolution);
  if (resolution.resolution_status !== 'resolved' || !resolution.referent.text) {
    return {
      rewrite_status: 'failed',
      original_query: query,
      rewritten_query: query,
      rewrite_confidence: 0,
      rewrite_reason: 'resolution_missing_referent',
      used_memory_fields: [],
      preserved_intent: intent,
      target_intent: intent,
      safety_flags: ['resolution_failed'],
      clarification_question: 'Bạn hãy nêu rõ đối tượng cần hỏi.',
    };
  }

  const targetIntent = intentForRewrite(query, intent);
  const rewritten = rewriteWithReferent(query, resolution.referent.text, memory, targetIntent);
  return {
    rewrite_status: 'rewritten',
    original_query: query,
    rewritten_query: rewritten,
    rewrite_confidence: Math.min(0.98, resolution.referent.confidence),
    rewrite_reason: resolution.resolution_reason,
    used_memory_fields: [resolution.referent.source],
    preserved_intent: intent,
    target_intent: targetIntent,
    safety_flags: [],
    clarification_question: '',
  };
}
