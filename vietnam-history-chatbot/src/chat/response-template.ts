/**
 * Response Template Registry — Stage 8B1 (No-API, deterministic)
 *
 * Provides per-intent answer templates that enforce:
 * - output shape and section structure
 * - word/bullet limits (non-lan-man)
 * - citation and rule-context policies
 * - abstention / clarification triggers
 *
 * This module NEVER calls an LLM or API.
 */

import type { AnswerIntent } from './answer-planner.js';

// ─── Template Schema ────────────────────────────────────────

export interface ResponseTemplate {
  template_id: string;
  intent: AnswerIntent;
  output_shape: 'short_direct' | 'bullet_explanation' | 'comparison_table' | 'timeline' | 'claim_evidence' | 'clarification_question' | 'insufficient_data';
  direct_answer_first: boolean;
  required_sections: string[];
  optional_sections: string[];
  forbidden_sections: string[];
  max_words: number;
  max_bullets: number;
  citation_required: boolean;
  rule_context_required: boolean;
  provenance_preferred: boolean;
  allow_partial_answer: boolean;
  must_abstain_if_insufficient: boolean;
  should_ask_clarification: boolean;
  non_lan_man_rules: string[];
  forbidden_expansion_rules: string[];
  notes: string;
}

// ─── Registry ───────────────────────────────────────────────

const TEMPLATES: ResponseTemplate[] = [
  {
    template_id: 'TPL_FACT',
    intent: 'fact',
    output_shape: 'short_direct',
    direct_answer_first: true,
    required_sections: ['direct_answer'],
    optional_sections: ['brief_context'],
    forbidden_sections: ['extended_analysis', 'other_period_events', 'full_biography'],
    max_words: 120,
    max_bullets: 3,
    citation_required: true,
    rule_context_required: false,
    provenance_preferred: true,
    allow_partial_answer: false,
    must_abstain_if_insufficient: false,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Trả lời thẳng câu hỏi trong 1–2 câu đầu',
      'Không mở rộng sang giai đoạn/sự kiện khác',
      'Không thêm bối cảnh dài nếu không được hỏi',
    ],
    forbidden_expansion_rules: [
      'Không kể toàn bộ lịch sử sự kiện liên quan',
      'Không phân tích ý nghĩa nếu chỉ hỏi "là gì/khi nào/ở đâu"',
    ],
    notes: 'Dùng cho câu hỏi sự kiện trực tiếp: ai/gì/ở đâu/khi nào.',
  },
  {
    template_id: 'TPL_WHY_MEANING',
    intent: 'why_meaning',
    output_shape: 'bullet_explanation',
    direct_answer_first: true,
    required_sections: ['direct_answer', 'reasons_or_meaning', 'evidence'],
    optional_sections: ['impact'],
    forbidden_sections: ['unrelated_timeline', 'other_period_events'],
    max_words: 220,
    max_bullets: 5,
    citation_required: true,
    rule_context_required: false,
    provenance_preferred: true,
    allow_partial_answer: true,
    must_abstain_if_insufficient: false,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Trả lời trực tiếp "vì sao/ý nghĩa" trước',
      'Liệt kê nguyên nhân/ý nghĩa theo bullet ngắn gọn',
      'Mỗi bullet có dẫn chứng cụ thể',
      'Không kể lại toàn bộ diễn biến sự kiện',
    ],
    forbidden_expansion_rules: [
      'Không liệt kê sự kiện phụ không liên quan đến nguyên nhân/ý nghĩa',
      'Không mở rộng sang giai đoạn khác nếu không cần',
    ],
    notes: 'Dùng cho "vì sao", "nguyên nhân", "ý nghĩa", "tác động".',
  },
  {
    template_id: 'TPL_COMPARISON',
    intent: 'comparison',
    output_shape: 'comparison_table',
    direct_answer_first: true,
    required_sections: ['shared_context', 'differences', 'conclusion'],
    optional_sections: ['similarities'],
    forbidden_sections: ['unrelated_events', 'excessive_narrative', 'full_history_both_sides'],
    max_words: 280,
    max_bullets: 6,
    citation_required: true,
    rule_context_required: true,
    provenance_preferred: true,
    allow_partial_answer: false,
    must_abstain_if_insufficient: false,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Nêu kết luận so sánh ngắn gọn trước',
      'Dùng bảng hoặc bullet đối chiếu theo tiêu chí',
      'Mỗi tiêu chí so sánh phải có cả hai bên',
      'Không kể dài toàn bộ lịch sử từng bên',
    ],
    forbidden_expansion_rules: [
      'Không kể chi tiết chỉ một bên mà bỏ bên kia',
      'Không thêm sự kiện ngoài phạm vi so sánh',
      'Không phân tích sâu nếu chỉ hỏi "khác nhau thế nào"',
    ],
    notes: 'Dùng cho "so sánh", "khác nhau", "giống và khác". Rule context bắt buộc.',
  },
  {
    template_id: 'TPL_TIMELINE',
    intent: 'timeline',
    output_shape: 'timeline',
    direct_answer_first: false,
    required_sections: ['chronological_events'],
    optional_sections: ['brief_intro', 'significance'],
    forbidden_sections: ['detailed_analysis', 'unrelated_events', 'unasked_period_events'],
    max_words: 280,
    max_bullets: 8,
    citation_required: true,
    rule_context_required: false,
    provenance_preferred: true,
    allow_partial_answer: true,
    must_abstain_if_insufficient: false,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Liệt kê mốc theo thứ tự thời gian',
      'Mỗi mốc: năm + sự kiện + ý nghĩa ngắn',
      'Không thêm phân tích chi tiết nếu không hỏi',
      'Không mở rộng sang giai đoạn ngoài phạm vi câu hỏi',
    ],
    forbidden_expansion_rules: [
      'Không chèn sự kiện ngoài timeline được hỏi',
      'Không biến timeline thành bài phân tích',
    ],
    notes: 'Dùng cho "diễn biến", "quá trình", "các mốc", "trình tự".',
  },
  {
    template_id: 'TPL_ENTITY_ROLE',
    intent: 'entity_role',
    output_shape: 'bullet_explanation',
    direct_answer_first: true,
    required_sections: ['role_description', 'evidence'],
    optional_sections: ['limitations', 'context'],
    forbidden_sections: ['excessive_biography', 'unrelated_events', 'full_life_story'],
    max_words: 200,
    max_bullets: 5,
    citation_required: true,
    rule_context_required: false,
    provenance_preferred: true,
    allow_partial_answer: true,
    must_abstain_if_insufficient: false,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Nêu vai trò/đóng góp chính trước',
      'Dẫn chứng cụ thể cho mỗi điểm',
      'Tránh kể toàn bộ tiểu sử',
      'Chỉ nói về vai trò trong phạm vi câu hỏi',
    ],
    forbidden_expansion_rules: [
      'Không kể tiểu sử từ nhỏ đến lớn',
      'Không liệt kê mọi sự kiện nhân vật tham gia',
    ],
    notes: 'Dùng cho "vai trò", "đóng góp", "ảnh hưởng của", "nhiệm vụ của".',
  },
  {
    template_id: 'TPL_DISAMBIGUATION',
    intent: 'disambiguation',
    output_shape: 'bullet_explanation',
    direct_answer_first: true,
    required_sections: ['direct_correction', 'distinction', 'evidence'],
    optional_sections: ['common_confusion'],
    forbidden_sections: ['extended_narrative', 'unrelated_events', 'tangential_events'],
    max_words: 220,
    max_bullets: 4,
    citation_required: true,
    rule_context_required: true,
    provenance_preferred: true,
    allow_partial_answer: false,
    must_abstain_if_insufficient: false,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Mở đầu bằng câu sửa nhầm: "Không nên hiểu X là Y" hoặc "Hai khái niệm này khác nhau"',
      'Nêu rõ điểm khác biệt quan trọng nhất',
      'Dẫn chứng ngắn gọn',
      'Không kể lại toàn bộ lịch sử hai khái niệm',
    ],
    forbidden_expansion_rules: [
      'Không biến phân biệt thành bài so sánh chi tiết',
      'Không thêm sự kiện phụ không phục vụ phân biệt',
    ],
    notes: 'Dùng cho "có phải", "phân biệt", "không đồng nghĩa". Rule context bắt buộc.',
  },
  {
    template_id: 'TPL_TEACHER_STYLE',
    intent: 'teacher_style_analysis',
    output_shape: 'claim_evidence',
    direct_answer_first: true,
    required_sections: ['thesis', 'claims_with_evidence', 'conclusion'],
    optional_sections: ['counterargument'],
    forbidden_sections: ['excessive_narrative', 'unrelated_periods', 'tangential_analysis'],
    max_words: 400,
    max_bullets: 8,
    citation_required: true,
    rule_context_required: false,
    provenance_preferred: true,
    allow_partial_answer: true,
    must_abstain_if_insufficient: false,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Nêu luận điểm chính (thesis) trước',
      'Mỗi luận điểm kèm dẫn chứng cụ thể',
      'Giới hạn 2–4 luận điểm, không liệt kê dài',
      'Kết luận ngắn gọn, chốt lại luận điểm',
      'Không kể chuyện lan man hay nhắc lại toàn bộ sự kiện',
    ],
    forbidden_expansion_rules: [
      'Không biến thành bài văn dài',
      'Không nhắc lại toàn bộ diễn biến sự kiện khi chỉ cần phân tích',
      'Không thêm giai đoạn/sự kiện ngoài phạm vi phân tích',
    ],
    notes: 'Dùng cho "phân tích", "chứng minh", "nhận xét", "đánh giá". Yêu cầu cấu trúc luận điểm–dẫn chứng.',
  },
  {
    template_id: 'TPL_FOLLOW_UP',
    intent: 'follow_up',
    output_shape: 'clarification_question',
    direct_answer_first: false,
    required_sections: ['clarification_request'],
    optional_sections: [],
    forbidden_sections: ['speculative_answer', 'guessed_reference'],
    max_words: 80,
    max_bullets: 2,
    citation_required: false,
    rule_context_required: false,
    provenance_preferred: false,
    allow_partial_answer: false,
    must_abstain_if_insufficient: false,
    should_ask_clarification: true,
    non_lan_man_rules: [
      'Không đoán "nó/sự kiện đó" là gì khi chưa có memory',
      'Hỏi lại rõ ràng: "Bạn đang hỏi về sự kiện/nhân vật nào?"',
      'Tối đa 1–2 câu hỏi lại',
    ],
    forbidden_expansion_rules: [
      'Không tự đoán ngữ cảnh và trả lời dài',
      'Không retrieve/generate broad answer',
    ],
    notes: 'Dùng khi query chứa đại từ chỉ định ("nó", "sự kiện đó") mà chưa có session memory.',
  },
  {
    template_id: 'TPL_NEGATIVE_GAP',
    intent: 'negative_gap',
    output_shape: 'insufficient_data',
    direct_answer_first: false,
    required_sections: ['data_limitation_warning'],
    optional_sections: ['safe_related_info'],
    forbidden_sections: ['confident_answer', 'fabricated_data', 'fabricated_citation'],
    max_words: 120,
    max_bullets: 3,
    citation_required: false,
    rule_context_required: false,
    provenance_preferred: false,
    allow_partial_answer: true,
    must_abstain_if_insufficient: true,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Nói rõ dữ liệu không đủ chi tiết để trả lời',
      'Có thể cung cấp thông tin liên quan tổng quát nếu có',
      'Không bịa số liệu hoặc trích dẫn',
      'Không trả lời tự tin khi thiếu dữ liệu',
    ],
    forbidden_expansion_rules: [
      'Không fabricate citation',
      'Không trả lời như thể có dữ liệu đầy đủ',
    ],
    notes: 'Dùng khi câu hỏi yêu cầu mức chi tiết vượt quá corpus (số liệu từng xã, danh sách tên, v.v.).',
  },
  {
    template_id: 'TPL_OUT_OF_SCOPE',
    intent: 'out_of_scope',
    output_shape: 'insufficient_data',
    direct_answer_first: false,
    required_sections: ['scope_notice'],
    optional_sections: ['suggested_topics'],
    forbidden_sections: ['fabricated_answer', 'confident_answer_outside_data'],
    max_words: 80,
    max_bullets: 2,
    citation_required: false,
    rule_context_required: false,
    provenance_preferred: false,
    allow_partial_answer: false,
    must_abstain_if_insufficient: true,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Nói rõ phạm vi hệ thống: lịch sử Việt Nam 1858–2000',
      'Gợi ý chủ đề có thể hỏi',
      'Không trả lời ngoài dữ liệu',
    ],
    forbidden_expansion_rules: [
      'Không cố gắng trả lời câu hỏi ngoài phạm vi',
      'Không fabricate thông tin',
    ],
    notes: 'Dùng khi câu hỏi ngoài phạm vi dữ liệu (sau 2000, phi lịch sử, v.v.).',
  },
  {
    template_id: 'TPL_UNCLEAR',
    intent: 'unclear',
    output_shape: 'clarification_question',
    direct_answer_first: false,
    required_sections: ['clarification_request'],
    optional_sections: [],
    forbidden_sections: ['speculative_answer', 'broad_history_dump'],
    max_words: 60,
    max_bullets: 2,
    citation_required: false,
    rule_context_required: false,
    provenance_preferred: false,
    allow_partial_answer: false,
    must_abstain_if_insufficient: false,
    should_ask_clarification: true,
    non_lan_man_rules: [
      'Hỏi lại một câu ngắn gọn',
      'Gợi ý cách đặt câu hỏi rõ hơn',
      'Không retrieve/generate broad answer',
    ],
    forbidden_expansion_rules: [
      'Không tự suy luận và trả lời dài',
    ],
    notes: 'Dùng khi query quá ngắn hoặc thiếu đối tượng cụ thể.',
  },
  {
    template_id: 'TPL_CITATION_SOURCE',
    intent: 'citation_source',
    output_shape: 'bullet_explanation',
    direct_answer_first: true,
    required_sections: ['source_identification', 'evidence_summary'],
    optional_sections: ['source_limitations'],
    forbidden_sections: ['fabricated_source', 'speculative_claim'],
    max_words: 200,
    max_bullets: 5,
    citation_required: true,
    rule_context_required: false,
    provenance_preferred: true,
    allow_partial_answer: true,
    must_abstain_if_insufficient: false,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Nêu rõ nguồn tài liệu nào đã được sử dụng',
      'Trích dẫn cụ thể từ corpus',
      'Không bịa nguồn hoặc trích dẫn giả',
      'Nếu thiếu nguồn, nói rõ giới hạn',
    ],
    forbidden_expansion_rules: [
      'Không fabricate citation',
      'Không tự tạo nguồn không có trong corpus',
    ],
    notes: 'Dùng khi người dùng hỏi về nguồn/căn cứ/bằng chứng cho một nhận định.',
  },
  {
    template_id: 'TPL_HALLUCINATION_TRAP',
    intent: 'hallucination_trap',
    output_shape: 'insufficient_data',
    direct_answer_first: false,
    required_sections: ['refusal_statement'],
    optional_sections: ['safe_alternative'],
    forbidden_sections: ['fabricated_answer', 'fabricated_data', 'confident_answer'],
    max_words: 80,
    max_bullets: 2,
    citation_required: false,
    rule_context_required: false,
    provenance_preferred: false,
    allow_partial_answer: false,
    must_abstain_if_insufficient: true,
    should_ask_clarification: false,
    non_lan_man_rules: [
      'Từ chối bịa đặt thông tin',
      'Giải thích rằng hệ thống chỉ trả lời dựa trên nguồn có sẵn',
      'Gợi ý cách hỏi phù hợp hơn nếu cần',
    ],
    forbidden_expansion_rules: [
      'Không fabricate bất kỳ thông tin nào',
      'Không trả lời tự tin khi được yêu cầu bịa',
    ],
    notes: 'Dùng khi người dùng yêu cầu bịa/đoán/suy luận không có nguồn.',
  },
];

// ─── Lookup ─────────────────────────────────────────────────

const templateMap = new Map<AnswerIntent, ResponseTemplate>();
for (const t of TEMPLATES) templateMap.set(t.intent, t);

/** Get the response template for a given intent. */
export function getTemplate(intent: AnswerIntent): ResponseTemplate {
  return templateMap.get(intent) ?? templateMap.get('fact')!;
}

/** Get all registered templates. */
export function getAllTemplates(): ResponseTemplate[] {
  return [...TEMPLATES];
}

/** Validate a template has all required fields. */
export function validateTemplate(t: ResponseTemplate): { valid: boolean; missing: string[] } {
  const required: (keyof ResponseTemplate)[] = [
    'template_id', 'intent', 'output_shape', 'direct_answer_first',
    'required_sections', 'optional_sections', 'forbidden_sections',
    'max_words', 'max_bullets', 'citation_required', 'rule_context_required',
    'provenance_preferred', 'allow_partial_answer', 'must_abstain_if_insufficient',
    'should_ask_clarification', 'non_lan_man_rules', 'forbidden_expansion_rules', 'notes',
  ];
  const missing = required.filter(k => t[k] === undefined || t[k] === null);
  return { valid: missing.length === 0, missing };
}
