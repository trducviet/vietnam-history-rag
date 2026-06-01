/**
 * Focus Checker — Stage 8A (No-API, deterministic)
 *
 * Validates that a simulated or real answer payload
 * adheres to the answer plan's focus constraints.
 *
 * This module NEVER calls an LLM or API.
 */

import type { AnswerPlan, AnswerIntent, IntentClassification } from './answer-planner.js';

// ─── Focus Check Result ─────────────────────────────────────

export interface FocusCheckResult {
  focus_ok: boolean;
  direct_answer_first_ok: boolean;
  required_sections_ok: boolean;
  citation_policy_ok: boolean;
  rule_context_ok: boolean;
  negative_gap_safe_ok: boolean;
  over_expansion_risk: 'low' | 'medium' | 'high';
  missing_focus_elements: string[];
  warnings: string[];
  blockers: string[];
}

// ─── Simulated Answer Payload ───────────────────────────────

export interface SimulatedAnswerPayload {
  case_id: string;
  query: string;
  intent: AnswerIntent;
  answer_plan: AnswerPlan;
  simulated_answer_status: string;
  simulated_outline: string[];
  supporting_doc_ids: string[];
  citation_source_ids: string[];
  rule_warnings: string[];
  has_rule_context: boolean;
  has_citations: boolean;
  has_direct_answer: boolean;
  context_doc_count: number;
}

// ─── Focus Check Logic ──────────────────────────────────────

/**
 * Check focus compliance of a simulated answer payload against its plan.
 */
export function checkFocus(
  payload: SimulatedAnswerPayload,
  plan: AnswerPlan,
  classification: IntentClassification,
): FocusCheckResult {
  const warnings: string[] = [];
  const blockers: string[] = [];
  const missing: string[] = [];

  // 1. Direct answer first
  const directFirstOk = !plan.direct_answer_first || payload.has_direct_answer;
  if (!directFirstOk) missing.push('direct_answer_first');

  // 2. Required sections
  const outlineText = payload.simulated_outline.join(' ').toLowerCase();
  let sectionsMissing = 0;
  for (const sec of plan.required_sections) {
    // Simplified check: outline should mention the section concept
    const sectionKeywords = [...(SECTION_KEYWORDS[sec] ?? []), sec];
    const found = sectionKeywords.some(kw => outlineText.includes(kw));
    if (!found) {
      sectionsMissing++;
      missing.push(`section:${sec}`);
    }
  }
  const requiredSectionsOk = sectionsMissing === 0;

  // 3. Citation policy
  let citationOk = true;
  if (plan.citation_policy.citations_required && !payload.has_citations) {
    citationOk = false;
    missing.push('citations_required');
  }
  if (plan.citation_policy.min_source_cards > 0 && payload.citation_source_ids.length < plan.citation_policy.min_source_cards) {
    // Warning, not blocker — source gaps are data-level
    warnings.push(`Citation sources ${payload.citation_source_ids.length} < min ${plan.citation_policy.min_source_cards}`);
  }

  // 4. Rule context
  let ruleContextOk = true;
  if (plan.context_policy.rule_context_required && !payload.has_rule_context) {
    ruleContextOk = false;
    missing.push('rule_context_required');
  }

  // 5. Negative gap safety
  let negGapOk = true;
  if (classification.negative_or_gap_likely) {
    if (payload.simulated_answer_status === 'answerable' && plan.negative_gap_policy.should_abstain) {
      negGapOk = false;
      blockers.push('negative_gap_answered_confidently');
    }
  }

  // 6. Over-expansion risk
  let overExpansion: 'low' | 'medium' | 'high' = 'low';
  if (payload.context_doc_count > 6) {
    overExpansion = 'medium';
    warnings.push(`Context has ${payload.context_doc_count} docs — potential over-expansion`);
  }
  if (payload.context_doc_count > 10) {
    overExpansion = 'high';
    blockers.push(`Context has ${payload.context_doc_count} docs — high over-expansion risk`);
  }

  // 7. Clarification required but not asked
  if (classification.requires_clarification && payload.simulated_answer_status === 'answerable') {
    warnings.push('Requires clarification but marked answerable');
  }

  const focusOk = directFirstOk && requiredSectionsOk && citationOk && ruleContextOk && negGapOk && overExpansion !== 'high';

  return {
    focus_ok: focusOk,
    direct_answer_first_ok: directFirstOk,
    required_sections_ok: requiredSectionsOk,
    citation_policy_ok: citationOk,
    rule_context_ok: ruleContextOk,
    negative_gap_safe_ok: negGapOk,
    over_expansion_risk: overExpansion,
    missing_focus_elements: missing,
    warnings,
    blockers,
  };
}

// ─── Section keyword mapping ────────────────────────────────

const SECTION_KEYWORDS: Record<string, string[]> = {
  direct_answer: ['trả lời', 'answer', 'kết luận', 'direct'],
  brief_context: ['bối cảnh', 'context', 'background'],
  reasons_or_meaning: ['nguyên nhân', 'ý nghĩa', 'reason', 'meaning', 'lý do', 'tác động'],
  evidence: ['chứng cứ', 'evidence', 'dẫn chứng', 'tài liệu', 'source'],
  impact: ['tác động', 'impact', 'ảnh hưởng', 'hậu quả'],
  shared_context: ['chung', 'shared', 'bối cảnh chung', 'common'],
  differences: ['khác', 'difference', 'phân biệt', 'distinction'],
  similarities: ['giống', 'similar', 'tương đồng'],
  conclusion: ['kết luận', 'conclusion', 'tổng kết', 'summary'],
  chronological_events: ['mốc', 'timeline', 'diễn biến', 'thời gian', 'năm'],
  brief_intro: ['giới thiệu', 'intro'],
  significance: ['ý nghĩa', 'significance', 'quan trọng'],
  role_description: ['vai trò', 'role', 'nhiệm vụ', 'đóng góp'],
  limitations: ['hạn chế', 'limitation', 'thiếu'],
  direct_correction: ['chính xác', 'correction', 'thực tế', 'phân biệt'],
  distinction: ['phân biệt', 'distinction', 'khác biệt'],
  common_confusion: ['nhầm lẫn', 'confusion', 'sai lầm'],
  thesis: ['luận điểm', 'thesis', 'chứng minh'],
  claims_with_evidence: ['dẫn chứng', 'claim', 'evidence', 'minh chứng'],
  counterargument: ['phản biện', 'counter'],
  clarification_request: ['hỏi lại', 'clarification', 'rõ hơn', 'nêu rõ'],
  data_limitation_warning: ['thiếu dữ liệu', 'limitation', 'không đủ', 'ngoài phạm vi'],
  safe_related_info: ['liên quan', 'related', 'tham khảo'],
  scope_notice: ['phạm vi', 'scope', 'ngoài phạm vi'],
  suggested_topics: ['gợi ý', 'suggest', 'có thể hỏi'],
  speculative_answer: ['suy đoán', 'speculative'],
  fabricated_answer: ['bịa', 'fabricat'],
  fabricated_data: ['bịa', 'fabricat'],
  confident_answer: ['chắc chắn', 'confident'],
  refusal_statement: ['từ chối', 'refusal', 'không thể bịa', 'không fabricate', 'không suy đoán'],
  safe_alternative: ['thay thế', 'alternative', 'gợi ý', 'có thể hỏi'],
  source_identification: ['nguồn', 'source', 'tài liệu', 'căn cứ'],
  evidence_summary: ['chứng cứ', 'evidence', 'dẫn chứng', 'bằng chứng'],
  source_limitations: ['hạn chế nguồn', 'source limitation', 'thiếu nguồn'],
  fabricated_source: ['bịa nguồn', 'fabricated source'],
  speculative_claim: ['suy đoán', 'speculative'],
};
