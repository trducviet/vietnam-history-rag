/**
 * Capability Classifier — PATCH 9I
 *
 * Post-pipeline classifier that inspects the RAG pipeline results
 * (guard, routing, evidence, context, confidence) to determine
 * the capability bucket, answer policy, and upgrade signals.
 *
 * Does NOT use benchmark case IDs.
 * Does NOT call external APIs.
 */

import type { CapabilityDecision, CapabilityBucket, AnswerPolicy, EvidenceQuality, UpgradeSignal } from './capability-types.js';
import type { ScopeGuardResult, RoutingResult, ContextBundle, ConfidenceLevel, QueryFrame } from '../shared/types.js';

// ─── Classifier Input ────────────────────────────────────────

export interface ClassifierInput {
  query: string;
  guardResult: ScopeGuardResult;
  routing: RoutingResult;
  contextBundle: ContextBundle;
  confidence: ConfidenceLevel;
  citationCount: number;
  /** Evidence diagnostics from evidence selector */
  evidenceDiagnostics?: {
    primary_count: number;
    supporting_count: number;
    contrast_count: number;
    excluded_count: number;
    warnings: string[];
  };
}

// ─── Normalization ───────────────────────────────────────────

function norm(text: string): string {
  return text.toLowerCase().normalize('NFKC')
    .replace(/[–—]/g, '-')
    .replace(/\s+/g, ' ')
    .trim();
}

// ─── Query Pattern Detectors ─────────────────────────────────

const DATE_PATTERNS = [
  'năm nào', 'ngày nào', 'khi nào', 'thời gian nào', 'vào năm nào',
  'được ký năm nào', 'diễn ra năm nào', 'xảy ra năm nào',
  'thành lập năm nào', 'ban hành năm nào', 'được đọc ngày nào',
  'ra đời năm nào', 'gia nhập năm nào', 'được thông qua năm nào',
];

const ACTOR_PATTERNS = [
  'ai đọc', 'ai ban', 'ai ký', 'ai chỉ huy', 'ai lãnh đạo',
  'ai chủ trì', 'ai là', 'ai viết', 'ai thành lập', 'ai sáng lập',
  'nhân vật nào', 'vua nào', 'tướng nào', 'chủ tịch nào',
  'gắn với ai', 'do ai', 'ai đứng đầu',
];

const LOCATION_PATTERNS = [
  'ở đâu', 'tại đâu', 'địa điểm nào', 'diễn ra ở', 'tại thành phố nào',
  'nơi nào', 'vùng nào',
];

const TIMELINE_PATTERNS = [
  'diễn biến', 'tiến trình', 'các giai đoạn', 'qua các năm',
  'theo thời gian', 'lộ trình', 'quá trình',
  'mốc chính', 'các mốc', 'các sự kiện chính',
  'giai đoạn nào', 'từng bước', 'nêu vài mốc',
];

const COMPARISON_MARKERS = [
  'khác nhau', 'so sánh', 'giống và khác',
  'khác biệt', 'điểm khác', 'khác gì',
  'giữa.*và', 'giữa.*với',
];

/** Check if query has comparison pattern (includes regex-like patterns) */
function hasComparisonPattern(q: string): boolean {
  if (COMPARISON_MARKERS.some(m => q.includes(m))) return true;
  // "A khác B như thế nào" — regex check
  if (/khác.*như thế nào/.test(q)) return true;
  // "A khác B thế nào" (without "như") — regex check
  if (/khác.*thế nào/.test(q)) return true;
  // "A khác B ở điểm nào" — regex check
  if (/khác.*ở.*điểm/.test(q)) return true;
  // "A và B khác" 
  if (/và.*khác/.test(q)) return true;
  return false;
}

const DISAMBIGUATION_MARKERS = [
  'có phải.*không', 'có phải là', 'đúng không', 'có giống',
  'cùng một', 'có liên quan',
];

const BROAD_FREEFORM_MARKERS = [
  'ý nghĩa', 'vai trò', 'tầm quan trọng', 'đánh giá',
  'tác động', 'ảnh hưởng', 'bài học', 'nhận xét',
  'vì sao', 'tại sao', 'lý do', 'nguyên nhân sâu xa',
  'bước ngoặt', 'ý nghĩa lịch sử',
];

/** 9I-R: Broad process patterns — query asks for a process/expansion broader than a single event */
const BROAD_PROCESS_PATTERNS = [
  'mở rộng', 'quá trình', 'như thế nào',
  'đánh chiếm', 'bình định', 'xâm lược',
];

/** 9I-R: Completeness/detail markers — query demands exhaustive detail */
const COMPLETENESS_MARKERS = [
  'đầy đủ', 'toàn diện', 'từng', 'nêu đầy đủ',
  'từng tướng', 'từng giai đoạn', 'từng bước',
  'so sánh toàn diện', 'liệt kê đầy đủ',
];

/** 9I-R: Relation query markers — query asks about connection between entities */
const RELATION_PATTERNS = [
  'liên hệ gì với', 'có liên quan gì đến', 'gắn với chiến lược',
  'có liên quan gì với', 'liên quan gì đến', 'liên hệ gì đến',
  'gắn với', 'quan hệ gì với', 'liên quan đến',
];

const CORPUS_GAP_MARKERS = [
  'toàn văn', 'nguyên văn', 'đầy đủ nội dung', 'chi tiết từng ngày',
  'danh sách đầy đủ', 'tất cả các', 'thống kê cụ thể', 'số liệu chính xác',
  'bao nhiêu người', 'tổng số', 'chi tiết các điều khoản',
  'tổng tư lệnh chính thức', 'từng chiến dịch lớn', 'toàn bộ quá trình',
  'nội dung đầy đủ', 'liệt kê tất cả', 'trình bày chi tiết toàn bộ',
];

const CLARIFICATION_PRONOUNS = [
  'nó', 'sự kiện đó', 'hiệp định đó', 'chiến dịch đó',
  'sự kiện này', 'hiệp định này', 'chiến dịch này',
  'trận đó', 'cuộc chiến đó', 'phong trào này', 'hiệp định ấy',
  'ở đó',
];

// Named entities that make a pronoun-containing query specific enough
const SPECIFICITY_ENTITIES = [
  'điện biên phủ', 'genève', 'paris', 'hồ chí minh', 'asean', 'apec',
  'cách mạng tháng tám', 'đổi mới', 'mậu thân', 'biên giới',
  'tuyên ngôn độc lập', 'cần vương', 'việt minh', 'đông du',
  'ngô đình diệm', 'võ nguyên giáp', 'nguyễn ái quốc',
  'lam sơn 719', 'đường 9', 'linebacker', 'hiệp định nhâm tuất',
  '1858', '1945', '1954', '1975', '1986', '1930',
];

// ─── Main Classifier ────────────────────────────────────────

export function classifyCapability(input: ClassifierInput): CapabilityDecision {
  const q = norm(input.query);
  const guard = input.guardResult;
  const routing = input.routing;
  const frame = routing.query_frame;
  const ctx = input.contextBundle;
  const conf = input.confidence;
  const evDiag = input.evidenceDiagnostics;

  // ── Rule 1: NEEDS_CLARIFICATION ──
  if (guard.decision === 'needs_clarification') {
    return makeDecision('NEEDS_CLARIFICATION', 'ASK_CLARIFICATION', 'AMBIGUOUS_QUERY', {
      confidenceCeiling: 'low',
      citationsAllowed: false,
      citationsRequired: false,
      reasons: ['Guard: needs_clarification', guard.reason ?? ''],
    });
  }

  // Also detect inline clarification needs (pronoun without entity)
  const hasVaguePronoun = CLARIFICATION_PRONOUNS.some(p => q.includes(p));
  const hasSpecificEntity = SPECIFICITY_ENTITIES.some(e => q.includes(e));
  if (hasVaguePronoun && !hasSpecificEntity && q.split(/\s+/).length <= 8) {
    return makeDecision('NEEDS_CLARIFICATION', 'ASK_CLARIFICATION', 'AMBIGUOUS_QUERY', {
      confidenceCeiling: 'low',
      citationsAllowed: false,
      citationsRequired: false,
      reasons: ['Inline pronoun detection without entity'],
    });
  }

  // ── Rule 2: OUT_OF_SCOPE ──
  if (guard.decision === 'out_of_scope') {
    return makeDecision('OUT_OF_SCOPE', 'REFUSE_OOS', 'OUT_OF_SCOPE', {
      confidenceCeiling: 'low',
      citationsAllowed: false,
      citationsRequired: false,
      reasons: ['Guard: out_of_scope', guard.reason ?? ''],
    });
  }

  // ── Evidence quality assessment ──
  const primaryCount = evDiag?.primary_count ?? ctx.primary_docs.length;
  const supportingCount = evDiag?.supporting_count ?? ctx.supporting_docs.length;
  const hasPrimary = primaryCount > 0;
  const hasWarnings = (evDiag?.warnings.length ?? 0) > 0;
  const hasEvidenceWarnings = (ctx.evidence_warnings?.length ?? 0) > 0;
  const intent = routing.intent;
  const frameIntent = frame?.intent;

  // Detect comparison side coverage
  const compSides = frame?.comparison_sides;
  const isComparison = intent === 'comparison' || frameIntent === 'comparison';
  const isDisambiguation = frameIntent === 'disambiguation' || frameIntent === 'misconception_check';

  // Check if answer shows honest partial markers
  const hasSideMissing = ctx.evidence_warnings?.some(w =>
    w.includes('side') || w.includes('Side') || w.includes('missing') || w.includes('insufficient')
  ) ?? false;

  // ── Rule 12: CORPUS_GAP ──
  if (CORPUS_GAP_MARKERS.some(m => q.includes(m))) {
    return makeDecision('CORPUS_GAP', 'CORPUS_GAP_NOTICE', 'CORPUS_GAP', {
      confidenceCeiling: 'low',
      citationsRequired: false,
      mustAcknowledgeMissingEvidence: true,
      upgradeSignals: ['needs_corpus_expansion'],
      reasons: ['Query requests detailed/complete data likely absent from corpus'],
    });
  }

  // ── Rule 3: SAFE_DATE_LOOKUP ──
  if (DATE_PATTERNS.some(p => q.includes(p))) {
    if (hasPrimary && input.citationCount > 0) {
      return makeDecision('SAFE_DATE_LOOKUP', 'FULL_ANSWER', 'STRONG_DIRECT', {
        confidenceCeiling: 'high',
        citationsRequired: true,
        reasons: ['Date lookup with direct evidence'],
      });
    }
    // Has date pattern but weak evidence
    return makeDecision('SAFE_DATE_LOOKUP', 'FULL_ANSWER_OR_CAUTION',
      hasPrimary ? 'DIRECT_BUT_NARROW' : 'LOW_RELEVANCE', {
        confidenceCeiling: 'medium',
        citationsRequired: true,
        upgradeSignals: hasPrimary ? [] : ['needs_vector_retrieval'],
        reasons: ['Date lookup — evidence ' + (hasPrimary ? 'narrow' : 'weak')],
      });
  }

  // ── Rule 4: SAFE_ACTOR_LOOKUP ──
  if (ACTOR_PATTERNS.some(p => q.includes(p))) {
    // 9I-R: structured role queries always need HONEST_PARTIAL
    // because corpus lacks structured role-relation data
    const ROLE_TERMS = ['tổng tư lệnh', 'tư lệnh trưởng', 'chính ủy', 'tổng chỉ huy'];
    const hasRoleTerm = ROLE_TERMS.some(r => q.includes(r));
    if (hasRoleTerm) {
      return makeDecision('HONEST_PARTIAL_REQUIRED', 'HONEST_PARTIAL', 'LOW_RELEVANCE', {
        confidenceCeiling: 'low',
        mustAcknowledgeMissingEvidence: true,
        upgradeSignals: ['needs_structured_role_relation'],
        reasons: ['Actor lookup with structured role term but only narrow evidence'],
      });
    }
    if (hasPrimary && input.citationCount > 0) {
      return makeDecision('SAFE_ACTOR_LOOKUP', 'FULL_ANSWER', 'STRONG_DIRECT', {
        confidenceCeiling: 'high',
        citationsRequired: true,
        reasons: ['Actor lookup with direct evidence'],
      });
    }
    if (hasPrimary) {
      return makeDecision('SAFE_ACTOR_LOOKUP', 'FULL_ANSWER_OR_CAUTION', 'DIRECT_BUT_NARROW', {
        confidenceCeiling: 'medium',
        citationsRequired: true,
        reasons: ['Actor lookup — evidence narrow'],
      });
    }
    return makeDecision('HONEST_PARTIAL_REQUIRED', 'HONEST_PARTIAL', 'LOW_RELEVANCE', {
      confidenceCeiling: 'low',
      mustAcknowledgeMissingEvidence: true,
      upgradeSignals: ['needs_vector_retrieval', 'needs_structured_role_relation'],
      reasons: ['Actor lookup — no direct primary evidence'],
    });
  }

  // ── Rule 5: SAFE_LOCATION_LOOKUP ──
  if (LOCATION_PATTERNS.some(p => q.includes(p))) {
    if (hasPrimary && input.citationCount > 0) {
      return makeDecision('SAFE_LOCATION_LOOKUP', 'FULL_ANSWER', 'STRONG_DIRECT', {
        confidenceCeiling: 'high',
        citationsRequired: true,
        reasons: ['Location lookup with direct evidence'],
      });
    }
    return makeDecision('SAFE_LOCATION_LOOKUP', 'FULL_ANSWER_OR_CAUTION',
      hasPrimary ? 'DIRECT_BUT_NARROW' : 'LOW_RELEVANCE', {
        confidenceCeiling: 'medium',
        citationsRequired: true,
        reasons: ['Location lookup — evidence ' + (hasPrimary ? 'narrow' : 'weak')],
      });
  }

  // ── Rule 9: SAFE_DISAMBIGUATION ──
  if (isDisambiguation || DISAMBIGUATION_MARKERS.some(m => q.includes(m))) {
    if (compSides && hasPrimary) {
      if (hasSideMissing) {
        return makeDecision('HONEST_PARTIAL_REQUIRED', 'HONEST_PARTIAL', 'ONE_SIDE_MISSING', {
          confidenceCeiling: 'medium',
          mustAcknowledgeMissingEvidence: true,
          upgradeSignals: ['needs_side_specific_retrieval'],
          reasons: ['Disambiguation — one side missing'],
        });
      }
      return makeDecision('SAFE_DISAMBIGUATION', 'FULL_ANSWER', 'STRONG_DIRECT', {
        confidenceCeiling: 'high',
        citationsRequired: true,
        reasons: ['Disambiguation with both-side evidence'],
      });
    }
    if (hasPrimary) {
      return makeDecision('SAFE_DISAMBIGUATION', 'FULL_ANSWER_OR_CAUTION', 'DIRECT_BUT_NARROW', {
        confidenceCeiling: 'medium',
        citationsRequired: true,
        reasons: ['Disambiguation — evidence present but no sides parsed'],
      });
    }
    return makeDecision('HONEST_PARTIAL_REQUIRED', 'HONEST_PARTIAL', 'LOW_RELEVANCE', {
      confidenceCeiling: 'low',
      mustAcknowledgeMissingEvidence: true,
      reasons: ['Disambiguation — no primary evidence'],
    });
  }

  // ── Rule 8: SAFE_COMPARISON ──
  if (isComparison || hasComparisonPattern(q)) {
    if (compSides) {
      // 9I-R: Comparison needs evidence for BOTH sides → require primaryCount >= 2
      if (hasSideMissing || !hasPrimary || primaryCount < 2) {
        return makeDecision('HONEST_PARTIAL_REQUIRED', 'HONEST_PARTIAL', 'ONE_SIDE_MISSING', {
          confidenceCeiling: 'medium',
          mustAcknowledgeMissingEvidence: true,
          upgradeSignals: ['needs_side_specific_retrieval'],
          reasons: ['Comparison — ' + (hasSideMissing ? 'one side missing' : primaryCount < 2 ? 'only one primary doc for two-sided comparison' : 'no primary evidence')],
        });
      }
      return makeDecision('SAFE_COMPARISON', 'FULL_ANSWER', 'STRONG_DIRECT', {
        confidenceCeiling: 'high',
        citationsRequired: true,
        reasons: ['Comparison with both-side evidence'],
      });
    }
    // Comparison-like but no sides parsed
    return makeDecision('LOW_EVIDENCE_FREEFORM', 'LOW_EVIDENCE_CAUTION', 'LOW_RELEVANCE', {
      confidenceCeiling: 'medium',
      upgradeSignals: ['needs_llm_synthesis'],
      reasons: ['Comparison — sides not parsed'],
    });
  }

  // ── Rule 7: SAFE_TIMELINE ──
  if (TIMELINE_PATTERNS.some(p => q.includes(p)) || intent === 'timeline' || frameIntent === 'timeline') {
    if (hasPrimary && primaryCount >= 2) {
      return makeDecision('SAFE_TIMELINE', 'FULL_ANSWER', 'STRONG_DIRECT', {
        confidenceCeiling: 'high',
        citationsRequired: true,
        reasons: ['Timeline with multiple primary docs'],
      });
    }
    if (hasPrimary) {
      return makeDecision('SAFE_TIMELINE', 'FULL_ANSWER_OR_CAUTION', 'DIRECT_BUT_NARROW', {
        confidenceCeiling: 'medium',
        citationsRequired: true,
        upgradeSignals: ['needs_vector_retrieval'],
        reasons: ['Timeline — only 1 primary doc'],
      });
    }
    return makeDecision('LOW_EVIDENCE_FREEFORM', 'LOW_EVIDENCE_CAUTION', 'LOW_RELEVANCE', {
      confidenceCeiling: 'low',
      upgradeSignals: ['needs_vector_retrieval', 'needs_llm_synthesis'],
      reasons: ['Timeline — no primary evidence'],
    });
  }

  // ── 9I-R Guard A: Completeness/detail demand ──
  // "đầy đủ/toàn diện/từng" with narrow evidence → HONEST_PARTIAL or CORPUS_GAP
  // Must fire BEFORE broad freeform check to catch "đầy đủ vai trò của từng..."
  if (COMPLETENESS_MARKERS.some(m => q.includes(m)) && primaryCount <= 1) {
    return makeDecision('HONEST_PARTIAL_REQUIRED', 'HONEST_PARTIAL', 'LOW_RELEVANCE', {
      confidenceCeiling: 'low',
      mustAcknowledgeMissingEvidence: true,
      upgradeSignals: ['needs_corpus_expansion', 'needs_structured_role_relation'],
      reasons: ['Completeness demand but only narrow evidence available'],
    });
  }

  // ── 9I-R Guard B: Relation query without relation evidence ──
  if (RELATION_PATTERNS.some(p => q.includes(p)) && primaryCount <= 1) {
    return makeDecision('HONEST_PARTIAL_REQUIRED', 'HONEST_PARTIAL', 'LOW_RELEVANCE', {
      confidenceCeiling: 'low',
      mustAcknowledgeMissingEvidence: true,
      upgradeSignals: ['needs_corpus_expansion', 'needs_structured_role_relation'],
      reasons: ['Relation query but only one entity evidence found'],
    });
  }

  // ── 9I-R Guard C: Broad process question with single narrow event ──
  if (BROAD_PROCESS_PATTERNS.filter(p => q.includes(p)).length >= 2 && primaryCount <= 1) {
    return makeDecision('HONEST_PARTIAL_REQUIRED', 'HONEST_PARTIAL', 'LOW_RELEVANCE', {
      confidenceCeiling: 'low',
      mustAcknowledgeMissingEvidence: true,
      upgradeSignals: ['needs_corpus_expansion'],
      reasons: ['Broad process question with only single narrow event evidence'],
    });
  }

  // ── 9I-R Guard D: Comparison-like query with narrow evidence and no sides parsed ──
  if (hasComparisonPattern(q) && !compSides && primaryCount <= 1) {
    return makeDecision('HONEST_PARTIAL_REQUIRED', 'HONEST_PARTIAL', 'ONE_SIDE_MISSING', {
      confidenceCeiling: 'low',
      mustAcknowledgeMissingEvidence: true,
      upgradeSignals: ['needs_side_specific_retrieval'],
      reasons: ['Comparison-like query with narrow evidence and no sides parsed'],
    });
  }

  // ── Rule 11: LOW_EVIDENCE_FREEFORM ──
  if (BROAD_FREEFORM_MARKERS.some(m => q.includes(m))) {
    if (hasPrimary && conf !== 'low') {
      return makeDecision('SAFE_DIRECT_FACT', 'FULL_ANSWER_OR_CAUTION', 'DIRECT_BUT_NARROW', {
        confidenceCeiling: 'medium',
        citationsRequired: true,
        upgradeSignals: ['needs_llm_synthesis'],
        reasons: ['Broad question with some direct evidence'],
      });
    }
    return makeDecision('LOW_EVIDENCE_FREEFORM', 'LOW_EVIDENCE_CAUTION', 'LOW_RELEVANCE', {
      confidenceCeiling: 'low',
      upgradeSignals: ['needs_llm_synthesis', 'needs_vector_retrieval'],
      reasons: ['Broad/conceptual question — retrieval insufficient'],
    });
  }

  // ── Rule 13: RETRIEVAL_WEAK ──
  if (hasPrimary && conf === 'low' && !hasEvidenceWarnings) {
    return makeDecision('RETRIEVAL_WEAK', 'FULL_ANSWER_OR_CAUTION', 'LOW_RELEVANCE', {
      confidenceCeiling: 'medium',
      citationsRequired: true,
      upgradeSignals: ['needs_vector_retrieval'],
      reasons: ['In-scope but low confidence retrieval'],
    });
  }

  // ── Rule 10: HONEST_PARTIAL_REQUIRED ──
  if (!hasPrimary || conf === 'low') {
    return makeDecision('HONEST_PARTIAL_REQUIRED', 'HONEST_PARTIAL', 'LOW_RELEVANCE', {
      confidenceCeiling: 'low',
      mustAcknowledgeMissingEvidence: true,
      upgradeSignals: ['needs_vector_retrieval'],
      reasons: ['No primary evidence or low confidence'],
    });
  }

  // ── Rule 6: SAFE_DIRECT_FACT (default for in-scope with evidence) ──
  const evQuality: EvidenceQuality = (primaryCount >= 2 || (hasPrimary && supportingCount >= 1))
    ? 'STRONG_DIRECT' : 'DIRECT_BUT_NARROW';
  return makeDecision('SAFE_DIRECT_FACT', 'FULL_ANSWER', evQuality, {
    confidenceCeiling: conf === 'high' ? 'high' : 'medium',
    citationsRequired: true,
    reasons: ['Direct fact with evidence'],
  });
}

// ─── Decision Builder ────────────────────────────────────────

interface DecisionOptions {
  confidenceCeiling?: 'high' | 'medium' | 'low';
  citationsAllowed?: boolean;
  citationsRequired?: boolean;
  mustAcknowledgeMissingEvidence?: boolean;
  upgradeSignals?: UpgradeSignal[];
  reasons?: string[];
}

function makeDecision(
  bucket: CapabilityBucket,
  policy: AnswerPolicy,
  evidenceQuality: EvidenceQuality,
  opts: DecisionOptions = {},
): CapabilityDecision {
  return {
    bucket,
    policy,
    evidenceQuality,
    confidenceCeiling: opts.confidenceCeiling ?? 'medium',
    citationsAllowed: opts.citationsAllowed ?? true,
    citationsRequired: opts.citationsRequired ?? false,
    mustAcknowledgeMissingEvidence: opts.mustAcknowledgeMissingEvidence ?? false,
    upgradeSignals: opts.upgradeSignals ?? [],
    reasons: opts.reasons ?? [],
  };
}
