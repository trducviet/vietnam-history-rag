/**
 * Session Memory Schema — Stage 8C1 (No-API, deterministic)
 *
 * Stores only lightweight conversational focus signals. It does not rewrite
 * queries, call APIs, create embeddings, or bypass citation/grounding rules.
 */

import type { AnswerIntent, AnswerStatus, IntentClassification } from './answer-planner.js';
import { classifyIntent } from './answer-planner.js';
import {
  detectMemoryReset,
  evaluateMemoryReadPolicy,
  evaluateMemoryWritePolicy,
  hasExplicitHistoricalAnchor,
  hasFollowUpReference,
  shouldDecayMemory,
} from './memory-policy.js';

export type MemoryPeriodLabel = '1930-1945' | '1945-1954' | '1954-1975' | 'cross_period' | 'unknown';
export type MemorySource = 'user_query' | 'assistant_answer' | 'retrieval_context' | 'manual';
export type MemoryEntityType = 'event' | 'person' | 'organization' | 'place' | 'concept' | 'agreement' | 'campaign' | 'period' | 'document';
export type ConversationScope = 'single_event' | 'comparison' | 'timeline' | 'teacher_style' | 'broad' | 'unknown';
export type MemorySafetyState =
  | 'memory_safe_to_use'
  | 'memory_low_confidence'
  | 'memory_conflict'
  | 'memory_unavailable'
  | 'memory_should_reset'
  | 'memory_should_ask_clarification';

export interface ActiveTopic {
  topic_id: string;
  label: string;
  confidence: number;
  source: MemorySource;
  last_updated_turn: number;
}

export interface ActivePeriod {
  label: MemoryPeriodLabel;
  years: number[];
  confidence: number;
  last_updated_turn: number;
}

export interface ActiveEntity {
  text: string;
  type: MemoryEntityType;
  confidence: number;
  aliases: string[];
  last_updated_turn: number;
}

export interface ActiveEvent {
  event_label: string;
  year: string;
  period: MemoryPeriodLabel;
  confidence: number;
  last_updated_turn: number;
}

export interface ConversationFocus {
  scope: ConversationScope;
  confidence: number;
}

export interface SessionMemorySafety {
  memory_reliable: boolean;
  needs_clarification: boolean;
  last_negative_gap: boolean;
  last_out_of_scope: boolean;
  memory_conflict: boolean;
}

export interface SessionMemoryState {
  session_id: string;
  turn_count: number;
  active_topic: ActiveTopic | null;
  active_period: ActivePeriod;
  active_entities: ActiveEntity[];
  active_events: ActiveEvent[];
  active_documents: ActiveEntity[];
  active_primary_person: ActiveEntity | null;
  active_primary_organization: ActiveEntity | null;
  active_primary_event: ActiveEvent | null;
  active_primary_document: ActiveEntity | null;
  active_comparison_targets: string[];
  last_supported_claims: string[];
  last_cited_docs: string[];
  last_citation_focus: string | null;
  last_answer_focus: string | null;
  last_intent: AnswerIntent | 'unknown';
  last_safe_mode: 'none' | 'negative_gap' | 'out_of_scope' | 'hallucination_trap' | 'unclear';
  last_turn_was_oos: boolean;
  last_turn_was_hallucination_trap: boolean;
  last_turn_was_unclear: boolean;
  last_user_question: string;
  last_user_intent: AnswerIntent | 'unknown';
  last_answer_summary: string;
  last_answer_status: AnswerStatus | 'unknown';
  last_cited_doc_ids: string[];
  last_cited_source_ids: string[];
  last_rule_context_used: boolean;
  unresolved_references: string[];
  conversation_focus: ConversationFocus;
  safety: SessionMemorySafety;
  created_at: string;
  updated_at: string;
}

export interface MemorySignals {
  query: string;
  intent: AnswerIntent;
  answer_status: AnswerStatus;
  topic: ActiveTopic | null;
  period: ActivePeriod;
  entities: ActiveEntity[];
  events: ActiveEvent[];
  cited_doc_ids: string[];
  cited_source_ids: string[];
  rule_context_used: boolean;
  unresolved_references: string[];
  conversation_focus: ConversationFocus;
  answer_summary: string;
  confidence: number;
  citation_weak: boolean;
  unresolved_follow_up: boolean;
}

export interface MemoryTurnInput {
  session_id?: string;
  turn_id?: number;
  user_query: string;
  answer_focus?: {
    intent_result?: Partial<IntentClassification> | Record<string, unknown>;
    answer_plan?: Record<string, unknown>;
    rendered_template_preview?: Record<string, unknown>;
    focus_check_result?: Record<string, unknown>;
    answer_status?: string;
    citation_policy_satisfied?: boolean;
    rule_context_used?: boolean;
    context_weak_warning?: boolean;
    should_ask_clarification?: boolean;
    should_abstain?: boolean;
    [key: string]: unknown;
  };
  context_doc_ids?: string[];
  cited_source_ids?: string[];
  answer_summary?: string;
  require_safe_citation?: boolean;
  now?: string;
}

export interface FollowUpResolution {
  resolved: boolean;
  referent_text: string;
  referent_type: 'topic' | 'event' | 'entity' | 'period' | 'summary' | 'none';
  confidence: number;
  resolution_source: 'active_topic' | 'active_event' | 'active_entity' | 'active_period' | 'last_answer_summary' | 'none';
  should_ask_clarification: boolean;
  reason: string;
}

const TOPIC_PATTERNS: Array<{ pattern: RegExp; label: string; type: MemoryEntityType; aliases?: string[]; year?: string }> = [
  { pattern: /điện\s*biên\s*phủ\s*1954/i, label: 'Điện Biên Phủ 1954', type: 'event', aliases: ['Chiến dịch Điện Biên Phủ'], year: '1954' },
  { pattern: /điện\s*biên\s*phủ\s*trên\s*không|hà\s*nội\s*-\s*điện\s*biên\s*phủ\s*trên\s*không/i, label: 'Điện Biên Phủ trên không 1972', type: 'event', aliases: ['12 ngày đêm 1972'], year: '1972' },
  { pattern: /điện\s*biên\s*phủ(?!\s*trên\s*không)/i, label: 'Điện Biên Phủ 1954', type: 'event', aliases: ['Chiến dịch Điện Biên Phủ'], year: '1954' },
  { pattern: /gen[eè]ve\s*1954|giơ-ne-vơ\s*1954|hiệp\s*định\s*gen[eè]ve/i, label: 'Genève 1954', type: 'agreement', aliases: ['Hiệp định Genève 1954'], year: '1954' },
  { pattern: /paris\s*1973|hiệp\s*định\s*paris/i, label: 'Paris 1973', type: 'agreement', aliases: ['Hiệp định Paris 1973'], year: '1973' },
  { pattern: /việt\s*minh/i, label: 'Việt Minh', type: 'organization', aliases: ['Mặt trận Việt Minh'] },
  { pattern: /cách\s*mạng\s*tháng\s*tám/i, label: 'Cách mạng Tháng Tám', type: 'event', aliases: ['Tổng khởi nghĩa Tháng Tám'], year: '1945' },
  { pattern: /tổng\s*tiến\s*công\s*(và\s*nổi\s*dậy\s*)?mùa\s*xuân\s*1975|xuân\s*1975/i, label: 'Tổng tiến công mùa Xuân 1975', type: 'campaign', aliases: ['Đại thắng mùa Xuân 1975'], year: '1975' },
  { pattern: /chiến\s*dịch\s*hồ\s*chí\s*minh/i, label: 'Chiến dịch Hồ Chí Minh', type: 'campaign', aliases: ['Chiến dịch Hồ Chí Minh 1975'], year: '1975' },
  { pattern: /mậu\s*thân\s*1968/i, label: 'Mậu Thân 1968', type: 'event', aliases: ['Tổng tiến công và nổi dậy Tết Mậu Thân'], year: '1968' },
  { pattern: /đồng\s*khởi\s*1959\s*[-–]\s*1960|đồng\s*khởi/i, label: 'Phong trào Đồng Khởi 1959-1960', type: 'event', aliases: ['Đồng Khởi'], year: '1960' },
  { pattern: /biên\s*giới\s*thu\s*đông\s*1950/i, label: 'Chiến dịch Biên giới Thu Đông 1950', type: 'campaign', aliases: ['Chiến dịch Biên giới'], year: '1950' },
  { pattern: /chiến\s*dịch\s*biên\s*giới\s*(1950)?|biên\s*giới\s*1950/i, label: 'Chiến dịch Biên giới Thu Đông 1950', type: 'campaign', aliases: ['Chiến dịch Biên giới'], year: '1950' },
  { pattern: /hậu\s*phương\s*miền\s*bắc/i, label: 'hậu phương miền Bắc', type: 'concept', aliases: ['miền Bắc hậu phương'] },
  { pattern: /tiền\s*tuyến\s*miền\s*nam/i, label: 'tiền tuyến miền Nam', type: 'concept', aliases: ['miền Nam tiền tuyến'] },
  { pattern: /cương\s*lĩnh\s*chính\s*trị/i, label: 'Cương lĩnh chính trị đầu tiên', type: 'document', aliases: ['Cương lĩnh 1930'], year: '1930' },
  { pattern: /luận\s*cương\s*chính\s*trị/i, label: 'Luận cương chính trị tháng 10/1930', type: 'document', aliases: ['Luận cương 1930'], year: '1930' },
  { pattern: /hội\s*nghị\s*trung\s*ương\s*(6|7|8)/i, label: 'Hội nghị Trung ương 6/7/8', type: 'event', aliases: ['HNTW 6/7/8'], year: '1941' },
  { pattern: /cần\s*vương/i, label: 'Phong trào Cần Vương', type: 'event', aliases: ['Cần Vương'] },
  { pattern: /xô\s*viết\s*nghệ\s*tĩnh/i, label: 'Xô viết Nghệ Tĩnh 1930-1931', type: 'event', aliases: ['Xô viết Nghệ Tĩnh'], year: '1930' },
  { pattern: /trường\s*sơn|đường\s*trường\s*sơn/i, label: 'Đường Trường Sơn', type: 'concept', aliases: ['Tuyến vận tải Trường Sơn'] },
  { pattern: /đổi\s*mới\s*1986|đổi\s*mới/i, label: 'Đổi mới 1986', type: 'event', aliases: ['Công cuộc Đổi mới'], year: '1986' },
  { pattern: /tuyên\s*ngôn\s*độc\s*lập/i, label: 'Tuyên ngôn Độc lập 1945', type: 'document', aliases: ['Tuyên ngôn Độc lập'], year: '1945' },
  { pattern: /đảng\s*cộng\s*sản(\s*việt\s*nam)?|thành\s*lập\s*đảng/i, label: 'Đảng Cộng sản Việt Nam', type: 'organization', aliases: ['Đảng Cộng sản'], year: '1930' },
  { pattern: /mặt\s*trận\s*dân\s*tộc\s*giải\s*phóng(\s*miền\s*nam)?|giải\s*phóng\s*miền\s*nam/i, label: 'Mặt trận Dân tộc Giải phóng miền Nam', type: 'organization', aliases: ['Mặt trận Giải phóng miền Nam'], year: '1960' },
  { pattern: /nguyễn\s*ái\s*quốc/i, label: 'Nguyễn Ái Quốc', type: 'person', aliases: ['Hồ Chí Minh'] },
  { pattern: /hồ\s*chí\s*minh/i, label: 'Hồ Chí Minh', type: 'person', aliases: ['Nguyễn Ái Quốc'] },
  { pattern: /trần\s*phú/i, label: 'Trần Phú', type: 'person', aliases: [] },
  { pattern: /bảo\s*đại/i, label: 'Bảo Đại', type: 'person', aliases: [] },
  { pattern: /ngô\s*đình\s*diệm/i, label: 'Ngô Đình Diệm', type: 'person', aliases: [] },
  { pattern: /võ\s*nguyên\s*giáp/i, label: 'Võ Nguyên Giáp', type: 'person', aliases: [] },
  { pattern: /kháng\s*chiến\s*chống\s*pháp/i, label: 'kháng chiến chống Pháp', type: 'period', aliases: ['chống Pháp'] },
  { pattern: /kháng\s*chiến\s*chống\s*mỹ/i, label: 'kháng chiến chống Mỹ', type: 'period', aliases: ['chống Mỹ'] },
  { pattern: /chiến\s*dịch\s*tây\s*nguyên|tây\s*nguyên/i, label: 'Chiến dịch Tây Nguyên', type: 'campaign', aliases: ['Tây Nguyên'], year: '1975' },
  // Patch 11A-FIX: Add key historical date patterns for memory focus tracking
  { pattern: /30\s*\/\s*4\s*\/?\s*1975|30\s*tháng\s*4\s*năm?\s*1975/i, label: 'Ngày 30/4/1975', type: 'event', aliases: ['Ngày giải phóng miền Nam', '30/4/1975'], year: '1975' },
  { pattern: /2\s*\/\s*9\s*\/?\s*1945|2\s*tháng\s*9\s*năm?\s*1945/i, label: 'Ngày 2/9/1945', type: 'event', aliases: ['Tuyên ngôn Độc lập', 'Quốc khánh'], year: '1945' },
  { pattern: /mậu\s*thân(?!\s*1968)/i, label: 'Mậu Thân 1968', type: 'event', aliases: ['Tổng tiến công Tết Mậu Thân'], year: '1968' },
  { pattern: /hội\s*nghị\s*thành\s*lập\s*đảng|3\s*\/\s*2\s*\/?\s*1930/i, label: 'Hội nghị thành lập Đảng 3/2/1930', type: 'event', aliases: ['Thành lập Đảng Cộng sản Việt Nam'], year: '1930' },
];

function nowIso(input?: MemoryTurnInput): string {
  return input?.now ?? new Date().toISOString();
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number(value.toFixed(4))));
}

function slug(text: string): string {
  return text.toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

function normalizePlain(text: string): string {
  return text.toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
}

function periodFromYear(year: number): MemoryPeriodLabel {
  if (year >= 1930 && year <= 1945) return '1930-1945';
  if (year >= 1946 && year <= 1954) return '1945-1954';
  if (year >= 1955 && year <= 1975) return '1954-1975';
  return 'unknown';
}

function detectYears(query: string): number[] {
  return [...new Set((query.match(/\b(18\d{2}|19\d{2}|20\d{2})\b/g) ?? []).map(Number))];
}

function detectPeriod(query: string, entities: ActiveEntity[], turn: number): ActivePeriod {
  const years = [...new Set([
    ...detectYears(query),
    ...entities.map(entity => Number((entity.aliases.join(' ') + ' ' + entity.text).match(/\b(18\d{2}|19\d{2}|20\d{2})\b/)?.[1])).filter(Number.isFinite),
  ])].sort((a, b) => a - b);
  const labels = [...new Set(years.map(periodFromYear).filter(label => label !== 'unknown'))];
  return {
    label: labels.length > 1 ? 'cross_period' : labels[0] ?? 'unknown',
    years,
    confidence: years.length > 0 ? 0.9 : 0.35,
    last_updated_turn: turn,
  };
}

function detectEntities(query: string, turn: number): ActiveEntity[] {
  const entities: ActiveEntity[] = [];
  for (const item of TOPIC_PATTERNS) {
    if (!item.pattern.test(query)) continue;
    entities.push({
      text: item.label,
      type: item.type,
      confidence: 0.9,
      aliases: item.aliases ?? [],
      last_updated_turn: turn,
    });
  }
  const deduped = dedupeEntities(entities);
  return deduped.filter(entity => {
    if (entity.type !== 'person') return true;
    return !deduped.some(other =>
      other !== entity
      && (other.type === 'event' || other.type === 'campaign')
      && normalizePlain(other.text).includes(normalizePlain(entity.text)));
  });
}

function dedupeEntities(entities: ActiveEntity[]): ActiveEntity[] {
  const byText = new Map<string, ActiveEntity>();
  for (const entity of entities) {
    const key = slug(entity.text);
    const existing = byText.get(key);
    if (!existing || entity.confidence > existing.confidence) byText.set(key, entity);
  }
  return [...byText.values()];
}

function detectEvents(entities: ActiveEntity[], period: ActivePeriod, turn: number): ActiveEvent[] {
  return entities
    .filter(entity => entity.type === 'event' || entity.type === 'campaign' || entity.type === 'agreement')
    .map(entity => {
      const year = entity.text.match(/\b(18\d{2}|19\d{2}|20\d{2})\b/)?.[1]
        ?? entity.aliases.join(' ').match(/\b(18\d{2}|19\d{2}|20\d{2})\b/)?.[1]
        ?? '';
      return {
        event_label: entity.text,
        year,
        period: year ? periodFromYear(Number(year)) : period.label,
        confidence: entity.confidence,
        last_updated_turn: turn,
      };
    });
}

function detectTopic(query: string, intent: AnswerIntent, entities: ActiveEntity[], turn: number): ActiveTopic | null {
  if (entities.length === 0) return null;
  const label = (intent === 'comparison' || intent === 'disambiguation' || entities.length > 1)
    ? entities.slice(0, 3).map(entity => entity.text).join(' vs ')
    : entities[0].text;
  return {
    topic_id: slug(label),
    label,
    confidence: entities.length > 1 || hasExplicitHistoricalAnchor(query) ? 0.9 : 0.78,
    source: 'user_query',
    last_updated_turn: turn,
  };
}

function detectConversationScope(intent: AnswerIntent, query: string): ConversationScope {
  if (intent === 'comparison' || intent === 'disambiguation') return 'comparison';
  if (intent === 'timeline') return 'timeline';
  if (intent === 'teacher_style_analysis') return 'teacher_style';
  if (/vai\s*trò|ý\s*nghĩa|là\s*gì|vì\s*sao/i.test(query)) return 'single_event';
  return 'unknown';
}

function getAnswerStatus(input: MemoryTurnInput, classification: IntentClassification): AnswerStatus {
  const raw = input.answer_focus?.answer_status
    ?? input.answer_focus?.answer_plan?.answer_status
    ?? input.answer_focus?.rendered_template_preview?.answer_status;
  if (raw === 'answerable' || raw === 'partially_answerable' || raw === 'needs_clarification' || raw === 'insufficient_data' || raw === 'out_of_scope') {
    return raw;
  }
  if (classification.intent === 'negative_gap') return 'insufficient_data';
  if (classification.intent === 'out_of_scope') return 'out_of_scope';
  if (classification.intent === 'follow_up' || classification.intent === 'unclear') return 'needs_clarification';
  return 'answerable';
}

function summarize(input: MemoryTurnInput, topic: ActiveTopic | null): string {
  const explicit = input.answer_summary?.trim();
  if (explicit) return explicit.slice(0, 240);
  return topic ? `Trọng tâm lượt trước: ${topic.label}.` : '';
}

export function createEmptySessionMemory(session_id = `session_${Date.now()}`, now = new Date().toISOString()): SessionMemoryState {
  return {
    session_id,
    turn_count: 0,
    active_topic: null,
    active_period: { label: 'unknown', years: [], confidence: 0, last_updated_turn: 0 },
    active_entities: [],
    active_events: [],
    active_documents: [],
    active_primary_person: null,
    active_primary_organization: null,
    active_primary_event: null,
    active_primary_document: null,
    active_comparison_targets: [],
    last_supported_claims: [],
    last_cited_docs: [],
    last_citation_focus: null,
    last_answer_focus: null,
    last_intent: 'unknown',
    last_safe_mode: 'none',
    last_turn_was_oos: false,
    last_turn_was_hallucination_trap: false,
    last_turn_was_unclear: false,
    last_user_question: '',
    last_user_intent: 'unknown',
    last_answer_summary: '',
    last_answer_status: 'unknown',
    last_cited_doc_ids: [],
    last_cited_source_ids: [],
    last_rule_context_used: false,
    unresolved_references: [],
    conversation_focus: { scope: 'unknown', confidence: 0 },
    safety: {
      memory_reliable: true,
      needs_clarification: false,
      last_negative_gap: false,
      last_out_of_scope: false,
      memory_conflict: false,
    },
    created_at: now,
    updated_at: now,
  };
}

export function computeMemoryConfidence(signals: Omit<MemorySignals, 'confidence'>): number {
  let score = 0.35;
  if (signals.topic) score += 0.22;
  if (signals.entities.length > 0) score += 0.14;
  if (signals.events.length > 0) score += 0.12;
  if (signals.period.label !== 'unknown') score += 0.08;
  if (signals.cited_source_ids.length > 0) score += 0.08;
  if (signals.rule_context_used) score += 0.04;
  if (signals.answer_status === 'partially_answerable') score -= 0.06;
  // Patch 11A-FIX: Reduce citation_weak penalty when high-confidence entities
  // are present. Entities clearly detected from explicit historical anchors
  // (like 'Genève 1954', 'Việt Minh') provide strong topic signal even without citations.
  if (signals.citation_weak) {
    const hasHighConfidenceEntity = signals.entities.some(e => e.confidence >= 0.8);
    score -= hasHighConfidenceEntity ? 0.08 : 0.18;
  }
  if (signals.intent === 'negative_gap' || signals.intent === 'out_of_scope' || signals.intent === 'unclear' || signals.intent === 'hallucination_trap') score = Math.min(score, 0.25);
  if (signals.intent === 'follow_up' && signals.unresolved_follow_up) score = Math.min(score, 0.35);
  return clamp01(score);
}

export function extractMemorySignalsFromAnswerFocus(input: MemoryTurnInput): MemorySignals {
  const turn = input.turn_id ?? 1;
  const query = input.user_query;
  const classified = input.answer_focus?.intent_result?.intent
    ? input.answer_focus.intent_result as IntentClassification
    : classifyIntent(query);
  const intent = classified.intent;
  const answer_status = getAnswerStatus(input, classified);
  const entities = detectEntities(query, turn);
  const period = detectPeriod(query, entities, turn);
  const events = detectEvents(entities, period, turn);
  const topic = detectTopic(query, intent, entities, turn);
  const unresolved_follow_up = intent === 'follow_up' && !topic && entities.length === 0;
  const unresolved_references = hasFollowUpReference(query) && unresolved_follow_up ? [query] : [];
  const citationSatisfied = input.answer_focus?.citation_policy_satisfied !== false;
  const sourceIds = [...new Set(input.cited_source_ids ?? [])];
  const docIds = [...new Set(input.context_doc_ids ?? [])];
  const citation_weak = input.require_safe_citation !== false
    && (!citationSatisfied || (sourceIds.length === 0 && !['negative_gap', 'out_of_scope', 'unclear', 'follow_up'].includes(intent)));
  const base: Omit<MemorySignals, 'confidence'> = {
    query,
    intent,
    answer_status,
    topic,
    period,
    entities,
    events,
    cited_doc_ids: docIds,
    cited_source_ids: sourceIds,
    rule_context_used: input.answer_focus?.rule_context_used === true,
    unresolved_references,
    conversation_focus: {
      scope: detectConversationScope(intent, query),
      confidence: topic || entities.length ? 0.82 : 0.35,
    },
    answer_summary: summarize(input, topic),
    citation_weak,
    unresolved_follow_up,
  };
  return { ...base, confidence: computeMemoryConfidence(base) };
}

function decayValue(value: number, amount = 0.1): number {
  return clamp01(value - amount);
}

function decayMemory(memory: SessionMemoryState): SessionMemoryState {
  return {
    ...memory,
    active_topic: memory.active_topic ? { ...memory.active_topic, confidence: decayValue(memory.active_topic.confidence) } : null,
    active_period: { ...memory.active_period, confidence: decayValue(memory.active_period.confidence) },
    active_entities: memory.active_entities.map(entity => ({ ...entity, confidence: decayValue(entity.confidence) })),
    active_events: memory.active_events.map(event => ({ ...event, confidence: decayValue(event.confidence) })),
    conversation_focus: { ...memory.conversation_focus, confidence: decayValue(memory.conversation_focus.confidence) },
  };
}

function mergeEntities(existing: ActiveEntity[], incoming: ActiveEntity[]): ActiveEntity[] {
  return dedupeEntities([...incoming, ...existing]).slice(0, 6);
}

function mergeEvents(existing: ActiveEvent[], incoming: ActiveEvent[]): ActiveEvent[] {
  const byLabel = new Map<string, ActiveEvent>();
  for (const event of [...incoming, ...existing]) {
    const key = slug(event.event_label);
    const prev = byLabel.get(key);
    if (!prev || event.confidence > prev.confidence) byLabel.set(key, event);
  }
  return [...byLabel.values()].slice(0, 4);
}

export function updateSessionMemoryFromTurn(memory: SessionMemoryState, input: MemoryTurnInput): SessionMemoryState {
  const now = nowIso(input);
  const nextTurn = input.turn_id ?? memory.turn_count + 1;
  const reset = detectMemoryReset(input.user_query);
  let working = { ...memory, turn_count: nextTurn, updated_at: now };

  if (reset.should_reset && reset.target === 'all') {
    working = createEmptySessionMemory(memory.session_id, memory.created_at);
    working.turn_count = nextTurn;
    working.updated_at = now;
  } else if (reset.should_reset || reset.should_decay || shouldDecayMemory(working, nextTurn)) {
    working = decayMemory(working);
  }

  const signals = extractMemorySignalsFromAnswerFocus({ ...input, turn_id: nextTurn });
  const writePolicy = evaluateMemoryWritePolicy(working, signals, input);
  const safety = {
    memory_reliable: writePolicy.allow,
    needs_clarification: signals.answer_status === 'needs_clarification' || signals.intent === 'unclear' || signals.unresolved_follow_up,
    last_negative_gap: signals.intent === 'negative_gap' || signals.answer_status === 'insufficient_data',
    last_out_of_scope: signals.intent === 'out_of_scope' || signals.answer_status === 'out_of_scope',
    memory_conflict: reset.reason === 'correction_or_topic_refinement' && signals.entities.length === 0,
  };
  const safeMode = signals.intent === 'negative_gap' || signals.intent === 'out_of_scope' || signals.intent === 'hallucination_trap' || signals.intent === 'unclear'
    ? signals.intent
    : 'none';

  const next: SessionMemoryState = {
    ...working,
    last_user_question: input.user_query,
    last_user_intent: signals.intent,
    last_answer_summary: signals.answer_summary,
    last_answer_status: signals.answer_status,
    last_cited_doc_ids: signals.cited_doc_ids,
    last_cited_source_ids: signals.cited_source_ids,
    last_rule_context_used: signals.rule_context_used,
    unresolved_references: signals.unresolved_references,
    last_supported_claims: signals.answer_summary ? [signals.answer_summary] : (signals.topic ? [`Trọng tâm: ${signals.topic.label}`] : []),
    last_cited_docs: signals.cited_doc_ids,
    last_citation_focus: signals.topic?.label ?? working.last_citation_focus ?? null,
    last_answer_focus: signals.topic?.label ?? working.last_answer_focus ?? null,
    last_intent: signals.intent,
    last_safe_mode: safeMode,
    last_turn_was_oos: signals.intent === 'out_of_scope' || signals.answer_status === 'out_of_scope',
    last_turn_was_hallucination_trap: signals.intent === 'hallucination_trap',
    last_turn_was_unclear: signals.intent === 'unclear' || signals.unresolved_follow_up,
    safety,
    updated_at: now,
  };

  if (!writePolicy.allow) return next;
  // Patch 11A-FIX: Preserve high individual entity/event confidence (≥0.8) from
  // explicit detection rather than clamping to overall memory confidence which is
  // artificially lowered by citation_weak penalty. This ensures entities like
  // 'Nguyễn Ái Quốc' (person, 0.9) remain usable for follow-up resolution.
  const entityConfidence = (e: { confidence: number }) =>
    e.confidence >= 0.8 ? Math.max(e.confidence * 0.95, signals.confidence) : Math.min(e.confidence, signals.confidence);
  const mergedEntities = mergeEntities(next.active_entities, signals.entities.map(entity => ({ ...entity, confidence: entityConfidence(entity) })));
  const mergedEvents = mergeEvents(next.active_events, signals.events.map(event => ({ ...event, confidence: entityConfidence(event) })));
  const activeDocuments = mergedEntities.filter(entity => entity.type === 'document' || entity.type === 'agreement').slice(0, 4);
  const comparisonTargets = signals.topic?.label.includes(' vs ')
    ? signals.topic.label.split(/\s+vs\s+/i).map(item => item.trim()).filter(Boolean)
    : signals.entities.length > 1
      ? signals.entities.slice(0, 4).map(entity => entity.text)
      : next.active_comparison_targets;

  return {
    ...next,
    active_topic: signals.topic ? { ...signals.topic, confidence: signals.confidence } : next.active_topic,
    active_period: { ...signals.period, confidence: Math.max(signals.period.confidence, signals.confidence - 0.05) },
    active_entities: mergedEntities,
    active_events: mergedEvents,
    active_documents: activeDocuments,
    active_primary_person: mergedEntities.find(entity => entity.type === 'person') ?? next.active_primary_person,
    active_primary_organization: mergedEntities.find(entity => entity.type === 'organization') ?? next.active_primary_organization,
    active_primary_event: mergedEvents[0] ?? next.active_primary_event,
    active_primary_document: activeDocuments[0] ?? next.active_primary_document,
    active_comparison_targets: comparisonTargets,
    conversation_focus: signals.conversation_focus,
    safety: {
      memory_reliable: true,
      needs_clarification: false,
      last_negative_gap: false,
      last_out_of_scope: false,
      memory_conflict: false,
    },
  };
}

export function resolveFollowUpReference(query: string, memory: SessionMemoryState): FollowUpResolution {
  const q = query.toLowerCase().normalize('NFKC').trim();

  if (memory.last_answer_status === 'insufficient_data' || memory.last_answer_status === 'out_of_scope' || memory.safety.last_negative_gap || memory.safety.last_out_of_scope) {
    return { resolved: false, referent_text: '', referent_type: 'none', confidence: 0, resolution_source: 'none', should_ask_clarification: true, reason: 'last_answer_not_safe_for_memory_resolution' };
  }

  if (!hasFollowUpReference(q)) {
    return { resolved: false, referent_text: '', referent_type: 'none', confidence: 0, resolution_source: 'none', should_ask_clarification: false, reason: 'query_has_no_follow_up_reference' };
  }

  if (/hai\s*sự\s*kiện\s*đó/i.test(q)) {
    const candidates = memory.active_events.filter(event => event.confidence >= 0.75);
    if (candidates.length === 2) {
      return { resolved: true, referent_text: candidates.map(event => event.event_label).join(' và '), referent_type: 'event', confidence: Math.min(...candidates.map(event => event.confidence)), resolution_source: 'active_event', should_ask_clarification: false, reason: 'exactly_two_active_events' };
    }
    return { resolved: false, referent_text: '', referent_type: 'none', confidence: 0, resolution_source: 'none', should_ask_clarification: true, reason: 'two_event_reference_without_exactly_two_candidates' };
  }

  if (/giai\s*đoạn\s*đó/i.test(q)) {
    if (memory.active_period.label !== 'unknown' && memory.active_period.confidence >= 0.75) {
      return { resolved: true, referent_text: memory.active_period.label, referent_type: 'period', confidence: memory.active_period.confidence, resolution_source: 'active_period', should_ask_clarification: false, reason: 'active_period_high_confidence' };
    }
    return { resolved: false, referent_text: '', referent_type: 'none', confidence: memory.active_period.confidence, resolution_source: 'none', should_ask_clarification: true, reason: 'active_period_missing_or_low_confidence' };
  }

  if (/(sau\s*đó|tiếp\s*theo|còn\s+sau\s+đó|còn\s+giai\s*đoạn\s+sau)/i.test(q)) {
    return { resolved: false, referent_text: memory.active_topic?.label ?? '', referent_type: memory.active_topic ? 'topic' : 'none', confidence: memory.active_topic?.confidence ?? 0, resolution_source: memory.active_topic ? 'active_topic' : 'none', should_ask_clarification: true, reason: 'chronology_follow_up_requires_query_rewriter_stage' };
  }

  const eventCandidates = memory.active_events.filter(event => event.confidence >= 0.75);
  if (/sự\s*kiện\s*đó/i.test(q) && eventCandidates.length !== 1) {
    return { resolved: false, referent_text: '', referent_type: 'none', confidence: eventCandidates[0]?.confidence ?? 0, resolution_source: 'none', should_ask_clarification: true, reason: 'ambiguous_event_reference' };
  }
  if (eventCandidates.length === 1 && /(nó|sự\s*kiện\s*đó)/i.test(q)) {
    return { resolved: true, referent_text: eventCandidates[0].event_label, referent_type: 'event', confidence: eventCandidates[0].confidence, resolution_source: 'active_event', should_ask_clarification: false, reason: 'single_high_confidence_active_event' };
  }

  if (memory.active_topic && memory.active_topic.confidence >= 0.75) {
    return { resolved: true, referent_text: memory.active_topic.label, referent_type: 'topic', confidence: memory.active_topic.confidence, resolution_source: 'active_topic', should_ask_clarification: false, reason: 'active_topic_high_confidence' };
  }

  const entityCandidates = memory.active_entities.filter(entity => entity.confidence >= 0.75);
  if (entityCandidates.length === 1) {
    return { resolved: true, referent_text: entityCandidates[0].text, referent_type: 'entity', confidence: entityCandidates[0].confidence, resolution_source: 'active_entity', should_ask_clarification: false, reason: 'single_high_confidence_active_entity' };
  }

  if (memory.last_answer_summary && memory.conversation_focus.confidence >= 0.75) {
    return { resolved: true, referent_text: memory.last_answer_summary, referent_type: 'summary', confidence: memory.conversation_focus.confidence, resolution_source: 'last_answer_summary', should_ask_clarification: false, reason: 'last_answer_summary_high_confidence' };
  }

  return { resolved: false, referent_text: '', referent_type: 'none', confidence: 0, resolution_source: 'none', should_ask_clarification: true, reason: 'no_reliable_memory_candidate' };
}

export function shouldUseMemoryForFollowUp(query: string, memory: SessionMemoryState): boolean {
  const resolution = resolveFollowUpReference(query, memory);
  return evaluateMemoryReadPolicy(query, memory, resolution).allow && resolution.resolved;
}

export function shouldAskClarificationFromMemory(query: string, memory: SessionMemoryState): boolean {
  const resolution = resolveFollowUpReference(query, memory);
  const decision = evaluateMemoryReadPolicy(query, memory, resolution);
  return resolution.should_ask_clarification || decision.safety_state === 'memory_should_ask_clarification' || decision.safety_state === 'memory_low_confidence';
}

export function resetSessionMemory(memory: SessionMemoryState, now = new Date().toISOString()): SessionMemoryState {
  return createEmptySessionMemory(memory.session_id, now);
}
