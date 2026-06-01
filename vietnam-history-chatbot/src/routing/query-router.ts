/**
 * Query Router — classifies user query intent and determines which indexes
 * to search, using gpt-5-mini / gpt-5.4-mini with structured output.
 *
 * Routing policy (from implementation plan v2):
 *  - fact/date/actor/location → Event Index priority
 *  - explanation/comparison/timeline/cause_effect → Synthesis Index + Events
 *  - entity_profile → Entity links + Synthesis person profiles
 *  - multi_hop → Both indexes
 */

import OpenAI from 'openai';
import { config } from '../shared/config.js';
import { buildChatCompletionParams, shouldCallLLM } from '../llm/openai-chat-compat.js';
import { isCloudRouterDisabled } from '../runtime/no-cloud-guard.js';
import type { QueryIntent, QueryFrame, RoutingResult, MetadataFilter, RuntimeDocSource } from '../shared/types.js';
import { buildQueryFrame } from './query-frame-builder.js';



// ─── Routing Prompt ──────────────────────────────────────────

const ROUTING_SYSTEM_PROMPT = `Bạn là một bộ phân loại câu hỏi lịch sử Việt Nam (1858–2000).

Nhiệm vụ: phân tích câu hỏi của người dùng và trả về JSON object với các trường:
- intent: phân loại ý định câu hỏi
- target_indexes: danh sách index cần tìm kiếm
- metadata_filters: bộ lọc metadata nếu có
- estimated_complexity: độ phức tạp ước tính
- reasoning: giải thích ngắn gọn lý do phân loại

Các loại intent:
- "fact_lookup": tra cứu sự kiện cụ thể, chi tiết factual
- "date_lookup": hỏi ngày tháng, năm xảy ra sự kiện
- "actor_lookup": hỏi về nhân vật liên quan đến sự kiện
- "location_lookup": hỏi về địa điểm
- "entity_profile": hỏi tổng quan về một nhân vật/tổ chức
- "explanation": hỏi giải thích nguyên nhân, ý nghĩa, hệ quả
- "comparison": so sánh giữa các sự kiện/giai đoạn
- "timeline": hỏi về chuỗi sự kiện theo thời gian
- "cause_effect": hỏi quan hệ nhân-quả
- "multi_hop": câu hỏi cần kết hợp nhiều nguồn thông tin

Quy tắc target_indexes:
- fact_lookup, date_lookup, actor_lookup, location_lookup → ["event"]
- explanation, cause_effect → ["synthesis", "event"]
- comparison → ["synthesis", "event"]
- timeline → ["synthesis", "event"]
- entity_profile → ["synthesis", "event"]
- multi_hop → ["event", "synthesis"]

Quy tắc metadata_filters (nếu phát hiện từ câu hỏi):
- Nếu câu hỏi đề cập năm cụ thể → set year_min, year_max
- Nếu câu hỏi đề cập giai đoạn → set period_label
- Luôn set canonical_only: true

Quy tắc estimated_complexity:
- "simple": tra cứu đơn giản, 1 sự kiện
- "moderate": cần 2-3 documents, giải thích
- "complex": so sánh, timeline, multi_hop, cần nhiều sources

Trả về ĐÚNG JSON object, không kèm text khác.`;

// ─── Intent → Index Mapping (fallback) ───────────────────────

const INTENT_TO_INDEXES: Record<QueryIntent, ('event' | 'synthesis')[]> = {
  fact_lookup: ['event'],
  date_lookup: ['event'],
  actor_lookup: ['event'],
  location_lookup: ['event'],
  entity_profile: ['synthesis', 'event'],
  explanation: ['synthesis', 'event'],
  comparison: ['synthesis', 'event'],
  timeline: ['synthesis', 'event'],
  cause_effect: ['synthesis', 'event'],
  multi_hop: ['event', 'synthesis'],
};

const VALID_INTENTS = new Set<QueryIntent>([
  'fact_lookup', 'date_lookup', 'actor_lookup', 'location_lookup',
  'entity_profile', 'explanation', 'comparison', 'timeline',
  'cause_effect', 'multi_hop',
]);

// ─── QueryFrame Alignment Helper ───────────────────────────────

/**
 * Map a QueryFrameIntent (fine-grained) onto a QueryIntent (legacy LLM type).
 * QueryIntent is a smaller set; we pick the closest match.
 */
function frameIntentToQueryIntent(frameIntent: QueryFrame['intent']): QueryIntent {
  switch (frameIntent) {
    case 'fact_lookup':       return 'fact_lookup';
    case 'date_lookup':       return 'date_lookup';
    case 'actor_lookup':
    case 'actor_date_lookup': return 'actor_lookup';
    case 'location_lookup':   return 'location_lookup';
    case 'organization_lookup':
    case 'movement_lookup':   return 'entity_profile';
    case 'treaty_lookup':
    case 'clause_lookup':
    case 'conference_lookup':
    case 'campaign_lookup':
    case 'sub_event_lookup':  return 'fact_lookup';
    case 'significance_lookup':
    case 'explanation':       return 'explanation';
    case 'cause_effect':      return 'cause_effect';
    case 'comparison':        return 'comparison';
    case 'timeline':          return 'timeline';
    // disambiguation & misconception_check: no direct QueryIntent — use multi_hop
    // because it opens both indexes and allows context builder to work well
    case 'disambiguation':
    case 'misconception_check': return 'multi_hop';
    case 'out_of_scope':      return 'fact_lookup'; // fallback; retrieval will find nothing useful
    default:                  return 'fact_lookup';
  }
}

/**
 * Resolve the preferred target indexes for a given QueryFrame.
 *
 * Priority:
 *  1. frame.constraints.prefer_index if explicitly set
 *  2. Intent-based defaults (synthesis+event for broad queries)
 */
function resolvePreferredIndexes(
  frame: QueryFrame
): RuntimeDocSource[] {
  if (frame.constraints?.prefer_index?.length) {
    return frame.constraints.prefer_index;
  }
  switch (frame.intent) {
    case 'timeline':
    case 'comparison':
    case 'explanation':
    case 'cause_effect':
    case 'significance_lookup':
    case 'misconception_check':
      return ['synthesis', 'event'];
    case 'disambiguation':
      // disambiguation is usually event-first but may need synthesis for context
      return frame.constraints?.requires_contrast ? ['event', 'synthesis'] : ['event'];
    default:
      return ['event'];
  }
}

/**
 * Align base RoutingResult intent/indexes/complexity with QueryFrame signals.
 *
 * Patch 7D-2 precision guards:
 * - Never override metadata_filters (narrow rules preserved).
 * - frame.confidence 'low' → attach only, no override.
 * - If query has a date marker (ngày nào/khi nào/năm nào) and frame intent
 *   is NOT misconception/comparison/disambiguation → keep date_lookup + ['event'].
 *   This prevents "Chiến dịch X bắt đầu vào ngày nào?" from being routed as timeline.
 * - If query has a location/actor marker and frame is not comparison/misconception,
 *   keep location_lookup or actor_lookup + ['event'].
 * - Only change indexes when frame brings genuine synthesis signal.
 */
export function alignRoutingWithQueryFrame(
  base: RoutingResult,
  frame: QueryFrame
): RoutingResult {
  // Always attach the frame
  const result: RoutingResult = { ...base, query_frame: frame };

  // Low confidence: only attach, do not override
  if (frame.confidence === 'low') {
    result.reasoning = `${base.reasoning} | [QueryFrame: low-confidence, no override]`;
    return result;
  }

  // ── Precision guards ──
  // These check whether the query itself has strong signals that must dominate.
  // Note: frame.intent already incorporates these in Patch 7D-2 (date guard in detectIntent).
  // The alignment guard here is a belt-and-suspenders check in case LLM routing set
  // a different base.intent than the frame.

  const isDateDominatedIntent =
    frame.intent === 'date_lookup' ||
    frame.intent === 'actor_lookup' ||
    frame.intent === 'location_lookup' ||
    frame.intent === 'organization_lookup';

  // If the frame itself detected a simple lookup intent, do not force synthesis indexes
  // (they would hurt precision for simple factual lookups)
  const isBroadContextIntent =
    frame.intent === 'timeline' ||
    frame.intent === 'comparison' ||
    frame.intent === 'misconception_check' ||
    frame.intent === 'explanation' ||
    frame.intent === 'cause_effect' ||
    frame.intent === 'significance_lookup';

  // ── Intent alignment ──
  const alignedIntent = frameIntentToQueryIntent(frame.intent);
  // Only update intent if base is a plain fallback or if frame brings a better signal
  const isDefaultFallback = base.intent === 'fact_lookup';
  if (isDefaultFallback || alignedIntent !== base.intent) {
    result.intent = alignedIntent;
  }

  // ── Index alignment ──
  // GUARD 1: narrow year filter → keep event lane
  const hasNarrowYearFilter =
    base.metadata_filters.year_min !== undefined ||
    base.metadata_filters.year_max !== undefined;

  // Patch 7L-F: Misconception queries often contain a WRONG year (e.g., "1975 phải không?").
  // Year filter would block correct evidence (e.g., Đổi Mới/1986). Clear year filters.
  if (frame.intent === 'misconception_check' && hasNarrowYearFilter) {
    delete result.metadata_filters.year_min;
    delete result.metadata_filters.year_max;
  }

  const hasNarrowYearFilterAfterMiscon =
    result.metadata_filters.year_min !== undefined ||
    result.metadata_filters.year_max !== undefined;

  // GUARD 2: simple lookup intent → keep event lane (don't add synthesis)
  if (hasNarrowYearFilterAfterMiscon || isDateDominatedIntent) {
    // Keep ['event'] — do not let frame push to synthesis
    // (synthesis does not help for date/actor/location point lookups)
    result.target_indexes = ['event'];
  } else if (isBroadContextIntent) {
    // Only push synthesis when frame genuinely needs broad context
    result.target_indexes = resolvePreferredIndexes(frame);
  }
  // Otherwise: keep base.target_indexes as-is

  // ── Complexity alignment ──
  switch (frame.intent) {
    case 'timeline':
    case 'comparison':
      result.estimated_complexity = 'complex';
      break;
    case 'disambiguation':
    case 'misconception_check':
    case 'explanation':
    case 'cause_effect':
    case 'significance_lookup':
      if (result.estimated_complexity === 'simple') {
        result.estimated_complexity = 'moderate';
      }
      break;
    default:
      break;
  }

  // ── Reasoning append ──
  const frameTag = `[QueryFrame: intent=${frame.intent}, confidence=${frame.confidence}` +
    (frame.answer_focus.action ? `, action=${frame.answer_focus.action}` : '') +
    (frame.contrast_focus?.action ? `, contrast.action=${frame.contrast_focus.action}` : '') +
    ']';
  result.reasoning = `${base.reasoning} | ${frameTag}`;

  return result;
}




// ─── Router Implementation ──────────────────────────────────

/**
 * Route a user query to determine intent, target indexes, and metadata filters.
 * Uses LLM for classification, with rule-based fallback if LLM fails.
 */
export async function routeQuery(query: string): Promise<RoutingResult> {
  // Narrow Paris-era rules take absolute priority over LLM routing.
  // ORDER MATTERS: US withdrawal MUST be checked BEFORE Paris signing.
  const qNorm = query.toLowerCase().normalize('NFKC');

  // Build the QueryFrame once for this query — used by all return paths
  const frame = buildQueryFrame(query);

  if (isUSWithdrawalAfterParisQuery(qNorm)) {
    const base: RoutingResult = {
      intent: 'fact_lookup',
      target_indexes: ['event'],
      metadata_filters: { canonical_only: true, year_min: 1973, year_max: 1973 },
      estimated_complexity: 'simple',
      reasoning: 'Narrow rule: US withdrawal query → year filter 1973 → prefer EVT_0339',
    };
    return alignRoutingWithQueryFrame(base, frame);
  }
  if (isParisTreatySigningQuery(qNorm)) {
    const base: RoutingResult = {
      intent: 'date_lookup',
      target_indexes: ['event'],
      metadata_filters: { canonical_only: true, year_min: 1973, year_max: 1973 },
      estimated_complexity: 'simple',
      reasoning: 'Narrow rule: Paris treaty signing date query → year filter 1973',
    };
    return alignRoutingWithQueryFrame(base, frame);
  }

  if (isCloudRouterDisabled()) {
    return routeWithRules(query, frame);
  }

  try {
    return await routeWithLLM(query, frame);
  } catch (error) {
    console.warn('⚠️  LLM routing failed, using rule-based fallback:', (error as Error).message);
    return routeWithRules(query, frame);
  }
}

/** LLM-based routing using gpt-5-mini */
async function routeWithLLM(query: string, frame: QueryFrame): Promise<RoutingResult> {
  if (!shouldCallLLM(config.openaiApiKey)) {
    throw new Error('LLM disabled or no API key — falling back to rules');
  }

  const openai = new OpenAI({ apiKey: config.openaiApiKey });

  const params = buildChatCompletionParams({
    model: config.routerModel,
    messages: [
      { role: 'system', content: ROUTING_SYSTEM_PROMPT },
      { role: 'user', content: query },
    ],
    maxTokens: 300,
    temperature: 0,
    responseFormat: { type: 'json_object' },
    purpose: 'routing',
  });

  const response = await openai.chat.completions.create(params as any);

  const content = response.choices[0]?.message?.content;
  if (!content) throw new Error('Empty response from router model');

  const parsed = JSON.parse(content) as Record<string, unknown>;
  const base = validateRoutingResult(parsed, query);
  return alignRoutingWithQueryFrame(base, frame);
}

/** Validate and normalize LLM output */
function validateRoutingResult(raw: Record<string, unknown>, rawQuery: string): RoutingResult {
  const intent = raw.intent as string;
  if (!VALID_INTENTS.has(intent as QueryIntent)) {
    throw new Error(`Invalid intent: ${intent}`);
  }

  const validIntent = intent as QueryIntent;
  const targetIndexes = Array.isArray(raw.target_indexes)
    ? raw.target_indexes.filter((i: unknown) => i === 'event' || i === 'synthesis') as ('event' | 'synthesis')[]
    : INTENT_TO_INDEXES[validIntent];

  const rawFilters = (raw.metadata_filters || {}) as Record<string, unknown>;
  const metadata_filters: MetadataFilter = {
    canonical_only: true,
    ...(typeof rawFilters.year_min === 'number' ? { year_min: rawFilters.year_min } : {}),
    ...(typeof rawFilters.year_max === 'number' ? { year_max: rawFilters.year_max } : {}),
    ...(typeof rawFilters.period_label === 'string' ? { period_label: rawFilters.period_label } : {}),
    ...(typeof rawFilters.doc_source === 'string' ? { doc_source: rawFilters.doc_source as 'event' | 'synthesis' } : {}),
  };

  const complexity = ['simple', 'moderate', 'complex'].includes(raw.estimated_complexity as string)
    ? raw.estimated_complexity as 'simple' | 'moderate' | 'complex'
    : 'moderate';

  return {
    intent: validIntent,
    target_indexes: targetIndexes.length > 0 ? targetIndexes : INTENT_TO_INDEXES[validIntent],
    metadata_filters,
    estimated_complexity: complexity,
    reasoning: typeof raw.reasoning === 'string' ? raw.reasoning : '',
    query_frame: buildQueryFrame(rawQuery),
  };
}

// ─── Referenced Year Detection ───────────────────────────────

/**
 * Detect queries where the year mentioned is a referenced/planned/future year
 * rather than the actual event metadata year.
 *
 * Example: "Điều khoản nào của Hiệp định Genève dự kiến tổng tuyển cử ... năm 1956?"
 * → 1956 is the planned election year, but the event is the 1954 Geneva Accords.
 *   Using 1956 as year filter would miss the correct documents.
 */
function isReferencedYearNotEventYear(query: string): boolean {
  const hasClauseKeyword = /điều khoản|quy định|dự kiến|theo hiệp định|cam kết|thỏa thuận/.test(query);
  const hasTreatyKeyword = /genève|geneve|giơnevơ|giơ-ne-vơ|hiệp định/.test(query);
  const hasFutureAction = /tổng tuyển cử|thống nhất|bầu cử|trưng cầu/.test(query);

  // Pattern 1: Treaty clause referencing a future year
  if (hasClauseKeyword && hasTreatyKeyword) return true;

  // Pattern 2: Treaty + planned future action + specific year
  if (hasTreatyKeyword && hasFutureAction) return true;

  // Pattern 3: "dự kiến" with any year = planned/future reference
  if (/dự kiến/.test(query) && /\d{4}/.test(query)) return true;

  return false;
}

// ─── Paris Treaty Specific Rules ─────────────────────────────

/**
 * Detect queries specifically asking about the signing date of the Paris Agreement
 * (Hiệp định Paris về chấm dứt chiến tranh, ký ngày 27-1-1973).
 *
 * This is a narrow rule to add year_min=1973, year_max=1973 so that
 * "Hội nghị Paris về Campuchia" (EVT_0404, year=1991) is excluded from results.
 *
 * Conditions (must ALL be true):
 * - query mentions "hiệp định paris" (not just "paris")
 * - query contains a signing/date interrogative keyword
 */
function isParisTreatySigningQuery(query: string): boolean {
  // Must explicitly mention "hiệp định paris"
  if (!/hiệp định paris/.test(query)) return false;

  // Guard: if query is primarily about US withdrawal / disambiguation,
  // do NOT route as Paris signing. Let the withdrawal rule handle it.
  if (/rút quân|rút khỏi|rút toàn bộ|mỹ rút/.test(query)) return false;

  // Must be asking about signing date / when it happened.
  // NOTE: \b does not work correctly with Vietnamese diacritics (ỹ, ý are
  // non-\w in JS regex), so we use space/punctuation-aware patterns instead.
  const hasDateKeyword = /(?:^|\s)ký(?:\s|,|\?|$)|được ký|khi nào|ngày nào|thời điểm nào|vào ngày|ngày ký/.test(query);
  return hasDateKeyword;
}

/**
 * Detect queries asking about US troop withdrawal after the Paris Agreement
 * (Mỹ rút quân, EVT_0339, year=1973).
 *
 * Triggers when query mentions withdrawal action + US/America reference.
 * Also triggers for disambiguation patterns like "khác với việc ký Hiệp định Paris".
 * Apply year_min=1973 to exclude unrelated events.
 *
 * NOTE: \b word boundaries break with Vietnamese diacritics (ỹ, ý, etc.
 * are non-\w in JS regex). Use space/punctuation-aware patterns instead.
 */
function isUSWithdrawalAfterParisQuery(query: string): boolean {
  const hasWithdrawal = /rút quân|rút khỏi|rút toàn bộ|rút hết|quân mỹ rút|mỹ rút/.test(query);
  // Match "mỹ" or "hoa kỳ" or "quân mỹ" without relying on \b
  const hasMy = /(?:^|\s|,)mỹ(?:\s|,|\?|!|$)|hoa kỳ|quân mỹ/.test(query);
  return hasWithdrawal && hasMy;
}

// ─── Rule-Based Fallback ─────────────────────────────────────

/** Simple keyword-based routing when LLM is unavailable */
export function routeWithRules(query: string, frame?: QueryFrame): RoutingResult {
  const q = query.toLowerCase().normalize('NFKC');

  let intent: QueryIntent = 'fact_lookup';
  let complexity: 'simple' | 'moderate' | 'complex' = 'simple';

  // Date patterns
  if (/năm nào|ngày nào|khi nào|thời gian|bao giờ|vào năm/.test(q)) {
    intent = 'date_lookup';
  }
  // Actor patterns
  else if (/ai đã|ai là|nhân vật|lãnh đạo|tướng|chủ tịch|vua/.test(q)) {
    intent = 'actor_lookup';
  }
  // Location patterns
  else if (/ở đâu|tại đâu|địa điểm|nơi nào/.test(q)) {
    intent = 'location_lookup';
  }
  // Entity profile patterns
  else if (/tiểu sử|cuộc đời|sự nghiệp|hồ sơ|profile/.test(q)) {
    intent = 'entity_profile';
    complexity = 'moderate';
  }
  // Comparison patterns
  else if (/so sánh|khác nhau|khác gì|giống nhau|khác biệt|tương đồng/.test(q)) {
    intent = 'comparison';
    complexity = 'complex';
  }
  // Timeline patterns
  else if (/diễn biến|quá trình|các sự kiện|theo thứ tự|timeline|chuỗi/.test(q)) {
    intent = 'timeline';
    complexity = 'complex';
  }
  // Cause-effect patterns
  else if (/tại sao|vì sao|nguyên nhân|hệ quả|dẫn đến|kết quả|ảnh hưởng/.test(q)) {
    intent = 'cause_effect';
    complexity = 'moderate';
  }
  // Explanation patterns
  else if (/giải thích|ý nghĩa|vai trò|tầm quan trọng|như thế nào/.test(q)) {
    intent = 'explanation';
    complexity = 'moderate';
  }

  // Extract year hints — with special-case overrides for narrow topic rules
  const metadata_filters: MetadataFilter = { canonical_only: true };

  // Narrow rule: US withdrawal after Paris → lock to 1973
  // NOTE: Must be checked BEFORE Paris signing rule (disambiguation queries match both)
  if (isUSWithdrawalAfterParisQuery(q)) {
    metadata_filters.year_min = 1973;
    metadata_filters.year_max = 1973;
  }
  // Narrow rule: Paris treaty signing → lock to 1973 to avoid false positive EVT_0404 (1991)
  else if (isParisTreatySigningQuery(q)) {
    metadata_filters.year_min = 1973;
    metadata_filters.year_max = 1973;
    intent = 'date_lookup';
  }
  // General year extraction — skip if year is a referenced/planned year
  // 9A4-FIX2: Also skip narrow year filter for comparison intent (spans multiple years/events)
  else if (!isReferencedYearNotEventYear(q) && intent !== 'comparison') {
    const yearMatch = q.match(/năm\s*(\d{4})/);
    if (yearMatch) {
      const year = parseInt(yearMatch[1], 10);
      metadata_filters.year_min = year;
      metadata_filters.year_max = year;
    }
    const rangeMatch = q.match(/(\d{4})\s*[-–]\s*(\d{4})/);
    if (rangeMatch) {
      metadata_filters.year_min = parseInt(rangeMatch[1], 10);
      metadata_filters.year_max = parseInt(rangeMatch[2], 10);
    }
  }

  const base: RoutingResult = {
    intent,
    target_indexes: INTENT_TO_INDEXES[intent],
    metadata_filters,
    estimated_complexity: complexity,
    reasoning: `Rule-based routing: matched intent "${intent}"`,
  };

  // Apply QueryFrame alignment if available (passed from routeQuery caller)
  const resolvedFrame = frame ?? buildQueryFrame(query);
  return alignRoutingWithQueryFrame(base, resolvedFrame);
}
