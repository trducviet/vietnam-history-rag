/**
 * Answer Planner — Stage 10A (No-API, deterministic)
 *
 * Classifies query intent and builds a structured answer plan
 * BEFORE answer generation. The plan constrains:
 * - output length and shape
 * - required/forbidden sections
 * - citation policy
 * - focus scope
 * - abstention/clarification triggers
 *
 * This module NEVER calls an LLM or API.
 */

// ─── Answer Intent (Stage 8A fine-grained) ──────────────────

export type AnswerIntent =
  | 'fact'
  | 'why_meaning'
  | 'comparison'
  | 'timeline'
  | 'entity_role'
  | 'disambiguation'
  | 'teacher_style_analysis'
  | 'follow_up'
  | 'negative_gap'
  | 'out_of_scope'
  | 'unclear'
  | 'citation_source'
  | 'hallucination_trap';

// ─── Intent Classification ──────────────────────────────────

export interface IntentClassification {
  intent: AnswerIntent;
  confidence: number;
  signals: string[];
  requires_memory: boolean;
  requires_rule_context: boolean;
  requires_citation: boolean;
  requires_clarification: boolean;
  negative_or_gap_likely: boolean;
}

/** Pattern-based signal with associated intent and weight */
interface SignalRule {
  patterns: RegExp[];
  intent: AnswerIntent;
  weight: number;
  signal: string;
}

const SIGNAL_RULES: SignalRule[] = [
  // ── follow_up (check FIRST — short pronominal references) ──
  { patterns: [/^(nó|sự kiện đó|giai đoạn đó|cái đó|điều đó|người đó)\s/i, /^(vậy thì|thế thì|so với cái đó|còn sau đó|rồi sao)/i, /^(còn|thế)\s.{0,15}(thì sao|thế nào)\s*\??$/i],
    intent: 'follow_up', weight: 10, signal: 'pronominal_reference' },

  // ── out_of_scope ──
  { patterns: [/sau\s*(năm\s*)?20[1-9]\d/i, /\bnăm\s*20[1-9]\d\b/i, /\b20[1-9]\d\b/i, /sau\s*2000/i, /thế\s*kỷ\s*21/i, /covid|internet|điện\s*thoại\s*thông\s*minh/i, /trận\s*bóng|chính\s*sách\s*giáo\s*dục|tình\s*hình\s*kinh\s*tế/i, /ngoài\s*phạm\s*vi/i],
    intent: 'out_of_scope', weight: 10, signal: 'temporal_oos' },

  // ── unclear ──
  { patterns: [/^.{0,8}$/], intent: 'unclear', weight: 8, signal: 'too_short' },

  // ── comparison ──
  { patterns: [/so\s*sánh/i, /khác\s*(nhau|gì)/i, /giống\s*và\s*khác/i, /theo\s*tiêu\s*chí/i, /khác\s+.*\s+thế\s*nào/i, /khác\s+.*\s+như\s*thế\s*nào/i, /khác\s+.*\s+ở\s*chỗ\s*nào/i],
    intent: 'comparison', weight: 9, signal: 'comparison_keyword' },

  // ── disambiguation ──
  { patterns: [/có\s*phải\s*(là)?/i, /phân\s*biệt/i, /không\s*đồng\s*nghĩa/i, /nhầm\s*lẫn/i, /không\s*phải\s*là/i, /có\s*phải\s*toàn\s*bộ/i],
    intent: 'disambiguation', weight: 9, signal: 'disambiguation_keyword' },

  // ── teacher_style_analysis ──
  { patterns: [/phân\s*tích/i, /chứng\s*minh\s*(rằng|là)?/i, /nhận\s*xét/i, /làm\s*rõ/i, /vì\s*sao\s*có\s*thể\s*nói/i, /mối\s*quan\s*hệ/i, /đánh\s*giá/i, /bình\s*luận/i, /trình\s*bày\s*.*\s*(mối|quan)/i],
    intent: 'teacher_style_analysis', weight: 8, signal: 'analysis_keyword' },

  // ── timeline ──
  { patterns: [/timeline/i, /diễn\s*biến/i, /quá\s*trình/i, /các\s*mốc/i, /trình\s*tự/i, /giai\s*đoạn/i, /theo\s*thời\s*gian/i, /gồm\s*những\s*mốc/i, /gồm\s*những\s*giai/i],
    intent: 'timeline', weight: 8, signal: 'timeline_keyword' },

  // ── entity_role ──
  { patterns: [/vai\s*trò/i, /đóng\s*góp/i, /ảnh\s*hưởng\s*của/i, /nhiệm\s*vụ\s*của/i, /công\s*lao/i, /quan\s*hệ\s*thế\s*nào/i],
    intent: 'entity_role', weight: 8, signal: 'entity_role_keyword' },

  // ── why_meaning ──
  { patterns: [/vì\s*sao/i, /tại\s*sao/i, /nguyên\s*nhân/i, /ý\s*nghĩa/i, /tác\s*động/i, /kết\s*quả/i, /hậu\s*quả/i, /mục\s*đích/i, /lý\s*do/i, /ảnh\s*hưởng\s*(đến|tới)/i, /bước\s*ngoặt/i],
    intent: 'why_meaning', weight: 7, signal: 'why_meaning_keyword' },

  // ── negative_gap ──
  { patterns: [/thương\s*vong\s*từng/i, /số\s*liệu\s*(cụ\s*thể|chi\s*tiết)?\s*(từng|mỗi)/i, /(từng|mỗi)\s*(xã|huyện|làng|thôn|đơn\s*vị)/i, /danh\s*sách\s*(đầy\s*đủ|tất\s*cả)|toàn\s*bộ\s*số\s*liệu/i, /tài\s*liệu\s*mật|chưa\s*công\s*bố|hồ\s*sơ\s*(mật|cá\s*nhân)/i, /nhân\s*vật\s*địa\s*phương\s*nhỏ/i, /tất\s*cả\s*liệt\s*sĩ|mọi\s*chiến\s*sĩ|cán\s*bộ\s*cấp\s*xã/i],
    intent: 'negative_gap', weight: 10, signal: 'granularity_oos' },

  // ── citation_source ──
  { patterns: [/dựa vào (đâu|nguồn|tài liệu)/i, /căn cứ (nào|gì)/i, /nguồn (nào|gì).*cho (thấy|biết)/i, /bằng chứng (nào|gì)/i, /tài liệu (nào|gì).*chứng minh/i, /có nguồn/i, /theo nguồn/i, /dẫn chứng (nào|gì)/i],
    intent: 'citation_source', weight: 9, signal: 'citation_source_query' },

  // ── hallucination_trap ──
  { patterns: [/cứ (suy luận|đoán|bịa)/i, /tự (suy luận|đoán|bịa)/i, /không có nguồn.*cũng được/i, /hãy bịa/i, /đừng nói là bịa/i, /dù (tài liệu|nguồn) không (ghi|có|nêu)/i, /nếu không có nguồn thì (tự|cứ)/i],
    intent: 'hallucination_trap', weight: 12, signal: 'fabrication_request' },

  // ── fact (default fallback) ──
  { patterns: [/là\s*gì/i, /diễn\s*ra\s*khi\s*nào/i, /ở\s*đâu/i, /^ai\s/i, /sự\s*kiện\s*nào/i, /khi\s*nào/i, /năm\s*nào/i, /bao\s*giờ/i, /nội\s*dung/i, /^(hãy\s*)?(kể|nêu|cho\s*biết)/i],
    intent: 'fact', weight: 5, signal: 'fact_keyword' },
];

function buildClassification(intent: AnswerIntent, confidence: number, signals: string[]): IntentClassification {
  return {
    intent,
    confidence,
    signals: signals.length > 0 ? signals : ['default_fact'],
    requires_memory: intent === 'follow_up',
    requires_rule_context: intent === 'comparison' || intent === 'disambiguation',
    requires_citation: intent !== 'unclear' && intent !== 'out_of_scope' && intent !== 'follow_up' && intent !== 'hallucination_trap',
    requires_clarification: intent === 'unclear' || intent === 'follow_up',
    negative_or_gap_likely: intent === 'negative_gap' || intent === 'out_of_scope' || intent === 'hallucination_trap',
  };
}

function safetyFirstIntentOverride(q: string): IntentClassification | null {
  const unclearExact = [
    /^nói\s*kỹ\s*hơn\s*(đi)?\.?\??$/,
    /^so\s*sánh\s*cái\s*đó\.?\??$/,
    /^nguyên\s*nhân\s*là\s*gì\.?\??$/,
    /^tiếp\s*theo\s*thì\s*sao\.?\??$/,
    /^giải\s*thích\s*thêm\.?\??$/,
    /^cái\s*nào\s*quan\s*trọng\.?\??$/,
  ];
  if (unclearExact.some(pattern => pattern.test(q))) {
    return buildClassification('unclear', 1.0, ['unresolved_short_request']);
  }

  if (/\bnăm\s*20[1-9]\d\b|\b20[1-9]\d\b|sau\s*(năm\s*)?2000|thế\s*kỷ\s*21|covid|internet|điện\s*thoại\s*thông\s*minh|trận\s*bóng|chính\s*sách\s*giáo\s*dục|tình\s*hình\s*kinh\s*tế/i.test(q)) {
    return buildClassification('out_of_scope', 1.0, ['temporal_or_domain_oos']);
  }

  if (/thương\s*vong\s*từng|số\s*liệu\s*(cụ\s*thể|chi\s*tiết)?\s*(từng|mỗi)|(từng|mỗi)\s*(xã|huyện|làng|thôn|đơn\s*vị)|danh\s*sách\s*(đầy\s*đủ|tất\s*cả)|toàn\s*bộ\s*số\s*liệu|tài\s*liệu\s*mật|chưa\s*công\s*bố|hồ\s*sơ\s*(mật|cá\s*nhân)|nhân\s*vật\s*địa\s*phương\s*nhỏ|tất\s*cả\s*liệt\s*sĩ|mọi\s*chiến\s*sĩ|cán\s*bộ\s*cấp\s*xã/i.test(q)) {
    return buildClassification('negative_gap', 1.0, ['granularity_or_unavailable_source_request']);
  }

  if (/cứ\s*(suy\s*luận|đoán|bịa)|tự\s*(suy\s*luận|đoán|bịa)|không\s*có\s*nguồn.*cũng\s*được|hãy\s*bịa|đừng\s*nói\s*là\s*bịa|dù\s*(tài\s*liệu|nguồn)\s*không\s*(ghi|có|nêu)/i.test(q)) {
    return buildClassification('hallucination_trap', 1.0, ['explicit_fabrication_request']);
  }

  if (/^(nó|sự\s*kiện\s*đó|giai\s*đoạn\s*đó|cái\s*đó|điều\s*đó|người\s*đó)\b/i.test(q)
      || /^(vậy\s*thì|thế\s*thì|còn\s*sau\s*đó|rồi\s*sao)\b/i.test(q)) {
    return buildClassification('follow_up', 1.0, ['unresolved_pronominal_reference']);
  }

  return null;
}

/** Classify query intent using deterministic rules. */
export function classifyIntent(query: string): IntentClassification {
  const q = query.toLowerCase().normalize('NFKC').trim();
  const override = safetyFirstIntentOverride(q);
  if (override) return override;

  const scores = new Map<AnswerIntent, { score: number; signals: string[] }>();

  for (const rule of SIGNAL_RULES) {
    for (const pat of rule.patterns) {
      if (pat.test(q)) {
        const entry = scores.get(rule.intent) ?? { score: 0, signals: [] };
        entry.score += rule.weight;
        if (!entry.signals.includes(rule.signal)) entry.signals.push(rule.signal);
        scores.set(rule.intent, entry);
        break; // one pattern per rule is enough
      }
    }
  }

  // Find best
  let bestIntent: AnswerIntent = 'fact';
  let bestScore = 0;
  let bestSignals: string[] = [];
  const priority: AnswerIntent[] = ['hallucination_trap', 'unclear', 'follow_up', 'out_of_scope', 'negative_gap', 'disambiguation', 'comparison', 'citation_source', 'teacher_style_analysis', 'timeline', 'entity_role', 'why_meaning', 'fact'];
  const priorityRank = (intent: AnswerIntent): number => priority.indexOf(intent) === -1 ? 99 : priority.indexOf(intent);
  for (const [intent, entry] of scores) {
    if (entry.score > bestScore || (entry.score === bestScore && priorityRank(intent) < priorityRank(bestIntent))) {
      bestIntent = intent;
      bestScore = entry.score;
      bestSignals = entry.signals;
    }
  }

  const confidence = bestScore >= 9 ? 1.0 : bestScore >= 7 ? 0.85 : bestScore >= 5 ? 0.7 : 0.5;

  return buildClassification(bestIntent, confidence, bestSignals);
}

// ─── Answer Plan ────────────────────────────────────────────

export type AnswerStatus =
  | 'answerable'
  | 'partially_answerable'
  | 'needs_clarification'
  | 'insufficient_data'
  | 'out_of_scope';

export type OutputShape =
  | 'short_direct'
  | 'bullet_explanation'
  | 'comparison_table'
  | 'timeline'
  | 'claim_evidence'
  | 'clarification_question'
  | 'insufficient_data';

export interface CitationPolicy {
  citations_required: boolean;
  min_source_cards: number;
  allow_flagged_source_gap: boolean;
  must_cite_each_main_claim: boolean;
}

export interface ContextPolicy {
  prefer_doc_types: string[];
  rule_context_required: boolean;
  provenance_required: boolean;
  allow_weak_context: boolean;
}

export interface FocusPolicy {
  answer_only_asked_scope: boolean;
  avoid_extra_background: boolean;
  allowed_periods: string[];
  forbidden_expansion: string[];
  must_include_keywords: string[];
  avoid_keywords: string[];
}

export interface NegativeGapPolicy {
  should_abstain: boolean;
  should_ask_clarifying_question: boolean;
  safe_partial_allowed: boolean;
}

export interface AnswerPlan {
  query: string;
  intent: AnswerIntent;
  answer_status: AnswerStatus;
  direct_answer_first: boolean;
  required_sections: string[];
  optional_sections: string[];
  forbidden_sections: string[];
  max_bullets: number;
  max_words: number;
  citation_policy: CitationPolicy;
  context_policy: ContextPolicy;
  focus_policy: FocusPolicy;
  negative_gap_policy: NegativeGapPolicy;
  output_shape: OutputShape;
}

/** Default citation policy */
const DEFAULT_CITATION: CitationPolicy = {
  citations_required: true, min_source_cards: 1,
  allow_flagged_source_gap: true, must_cite_each_main_claim: true,
};

/** Intent-specific plan templates */
const PLAN_TEMPLATES: Record<AnswerIntent, Omit<AnswerPlan, 'query' | 'intent' | 'answer_status'>> = {
  fact: {
    direct_answer_first: true,
    required_sections: ['direct_answer'],
    optional_sections: ['brief_context'],
    forbidden_sections: ['extended_analysis', 'other_period_events'],
    max_bullets: 3, max_words: 120,
    citation_policy: { ...DEFAULT_CITATION, min_source_cards: 1 },
    context_policy: { prefer_doc_types: ['event'], rule_context_required: false, provenance_required: true, allow_weak_context: false },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: ['unrelated_periods'], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: false, should_ask_clarifying_question: false, safe_partial_allowed: false },
    output_shape: 'short_direct',
  },
  why_meaning: {
    direct_answer_first: true,
    required_sections: ['direct_answer', 'reasons_or_meaning', 'evidence'],
    optional_sections: ['impact'],
    forbidden_sections: ['unrelated_timeline', 'other_period_events'],
    max_bullets: 5, max_words: 250,
    citation_policy: { ...DEFAULT_CITATION, min_source_cards: 2 },
    context_policy: { prefer_doc_types: ['synthesis', 'event'], rule_context_required: false, provenance_required: true, allow_weak_context: false },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: false, allowed_periods: [], forbidden_expansion: [], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: false, should_ask_clarifying_question: false, safe_partial_allowed: true },
    output_shape: 'bullet_explanation',
  },
  comparison: {
    direct_answer_first: true,
    required_sections: ['shared_context', 'differences', 'conclusion'],
    optional_sections: ['similarities'],
    forbidden_sections: ['unrelated_events', 'excessive_narrative'],
    max_bullets: 6, max_words: 300,
    citation_policy: { ...DEFAULT_CITATION, min_source_cards: 2, must_cite_each_main_claim: true },
    context_policy: { prefer_doc_types: ['disambiguation_rule', 'comparison_note', 'synthesis'], rule_context_required: true, provenance_required: true, allow_weak_context: false },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: ['single_side_only'], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: false, should_ask_clarifying_question: false, safe_partial_allowed: false },
    output_shape: 'comparison_table',
  },
  timeline: {
    direct_answer_first: false,
    required_sections: ['chronological_events'],
    optional_sections: ['brief_intro', 'significance'],
    forbidden_sections: ['detailed_analysis', 'unrelated_events'],
    max_bullets: 10, max_words: 300,
    citation_policy: { ...DEFAULT_CITATION, min_source_cards: 2 },
    context_policy: { prefer_doc_types: ['event', 'synthesis'], rule_context_required: false, provenance_required: true, allow_weak_context: false },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: ['unasked_periods'], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: false, should_ask_clarifying_question: false, safe_partial_allowed: true },
    output_shape: 'timeline',
  },
  entity_role: {
    direct_answer_first: true,
    required_sections: ['role_description', 'evidence'],
    optional_sections: ['limitations', 'context'],
    forbidden_sections: ['excessive_biography', 'unrelated_events'],
    max_bullets: 5, max_words: 200,
    citation_policy: { ...DEFAULT_CITATION, min_source_cards: 1 },
    context_policy: { prefer_doc_types: ['synthesis', 'event'], rule_context_required: false, provenance_required: true, allow_weak_context: false },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: ['full_biography'], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: false, should_ask_clarifying_question: false, safe_partial_allowed: true },
    output_shape: 'bullet_explanation',
  },
  disambiguation: {
    direct_answer_first: true,
    required_sections: ['direct_correction', 'distinction', 'evidence'],
    optional_sections: ['common_confusion'],
    forbidden_sections: ['extended_narrative', 'unrelated_events'],
    max_bullets: 4, max_words: 220,
    citation_policy: { ...DEFAULT_CITATION, min_source_cards: 1, must_cite_each_main_claim: true },
    context_policy: { prefer_doc_types: ['disambiguation_rule', 'event'], rule_context_required: true, provenance_required: true, allow_weak_context: false },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: ['tangential_events'], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: false, should_ask_clarifying_question: false, safe_partial_allowed: false },
    output_shape: 'short_direct',
  },
  teacher_style_analysis: {
    direct_answer_first: true,
    required_sections: ['thesis', 'claims_with_evidence', 'conclusion'],
    optional_sections: ['counterargument'],
    forbidden_sections: ['excessive_narrative', 'unrelated_periods'],
    max_bullets: 8, max_words: 400,
    citation_policy: { ...DEFAULT_CITATION, min_source_cards: 3, must_cite_each_main_claim: true },
    context_policy: { prefer_doc_types: ['synthesis', 'event'], rule_context_required: false, provenance_required: true, allow_weak_context: false },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: false, allowed_periods: [], forbidden_expansion: ['tangential_analysis'], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: false, should_ask_clarifying_question: false, safe_partial_allowed: true },
    output_shape: 'claim_evidence',
  },
  follow_up: {
    direct_answer_first: true,
    required_sections: ['clarification_request'],
    optional_sections: [],
    forbidden_sections: ['speculative_answer'],
    max_bullets: 2, max_words: 80,
    citation_policy: { citations_required: false, min_source_cards: 0, allow_flagged_source_gap: true, must_cite_each_main_claim: false },
    context_policy: { prefer_doc_types: [], rule_context_required: false, provenance_required: false, allow_weak_context: true },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: [], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: false, should_ask_clarifying_question: true, safe_partial_allowed: false },
    output_shape: 'clarification_question',
  },
  negative_gap: {
    direct_answer_first: false,
    required_sections: ['data_limitation_warning'],
    optional_sections: ['safe_related_info'],
    forbidden_sections: ['confident_answer', 'fabricated_data'],
    max_bullets: 3, max_words: 120,
    citation_policy: { citations_required: false, min_source_cards: 0, allow_flagged_source_gap: true, must_cite_each_main_claim: false },
    context_policy: { prefer_doc_types: [], rule_context_required: false, provenance_required: false, allow_weak_context: true },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: ['fabrication'], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: true, should_ask_clarifying_question: false, safe_partial_allowed: true },
    output_shape: 'insufficient_data',
  },
  out_of_scope: {
    direct_answer_first: false,
    required_sections: ['scope_notice'],
    optional_sections: ['suggested_topics'],
    forbidden_sections: ['fabricated_answer'],
    max_bullets: 2, max_words: 80,
    citation_policy: { citations_required: false, min_source_cards: 0, allow_flagged_source_gap: true, must_cite_each_main_claim: false },
    context_policy: { prefer_doc_types: [], rule_context_required: false, provenance_required: false, allow_weak_context: true },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: [], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: true, should_ask_clarifying_question: false, safe_partial_allowed: false },
    output_shape: 'insufficient_data',
  },
  citation_source: {
    direct_answer_first: true,
    required_sections: ['source_identification', 'evidence_summary'],
    optional_sections: ['source_limitations'],
    forbidden_sections: ['fabricated_source', 'speculative_claim'],
    max_bullets: 5, max_words: 200,
    citation_policy: { ...DEFAULT_CITATION, min_source_cards: 1, must_cite_each_main_claim: true },
    context_policy: { prefer_doc_types: ['event', 'synthesis'], rule_context_required: false, provenance_required: true, allow_weak_context: false },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: ['fabricated_source'], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: false, should_ask_clarifying_question: false, safe_partial_allowed: true },
    output_shape: 'bullet_explanation',
  },
  hallucination_trap: {
    direct_answer_first: false,
    required_sections: ['refusal_statement'],
    optional_sections: ['safe_alternative'],
    forbidden_sections: ['fabricated_answer', 'fabricated_data', 'confident_answer'],
    max_bullets: 2, max_words: 80,
    citation_policy: { citations_required: false, min_source_cards: 0, allow_flagged_source_gap: true, must_cite_each_main_claim: false },
    context_policy: { prefer_doc_types: [], rule_context_required: false, provenance_required: false, allow_weak_context: true },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: ['fabrication'], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: true, should_ask_clarifying_question: false, safe_partial_allowed: false },
    output_shape: 'insufficient_data',
  },
  unclear: {
    direct_answer_first: false,
    required_sections: ['clarification_request'],
    optional_sections: [],
    forbidden_sections: ['speculative_answer'],
    max_bullets: 2, max_words: 60,
    citation_policy: { citations_required: false, min_source_cards: 0, allow_flagged_source_gap: true, must_cite_each_main_claim: false },
    context_policy: { prefer_doc_types: [], rule_context_required: false, provenance_required: false, allow_weak_context: true },
    focus_policy: { answer_only_asked_scope: true, avoid_extra_background: true, allowed_periods: [], forbidden_expansion: [], must_include_keywords: [], avoid_keywords: [] },
    negative_gap_policy: { should_abstain: false, should_ask_clarifying_question: true, safe_partial_allowed: false },
    output_shape: 'clarification_question',
  },
};

/** Build an answer plan from intent classification. */
export function buildAnswerPlan(query: string, classification: IntentClassification): AnswerPlan {
  const template = PLAN_TEMPLATES[classification.intent];
  let answerStatus: AnswerStatus = 'answerable';
  if (classification.intent === 'out_of_scope') answerStatus = 'out_of_scope';
  else if (classification.intent === 'negative_gap') answerStatus = 'insufficient_data';
  else if (classification.intent === 'hallucination_trap') answerStatus = 'out_of_scope';
  else if (classification.intent === 'unclear' || classification.intent === 'follow_up') answerStatus = 'needs_clarification';

  return { query, intent: classification.intent, answer_status: answerStatus, ...template };
}
