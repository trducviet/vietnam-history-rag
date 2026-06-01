/**
 * Focus Precision Helper (Patch 7L / 7L-A / 7L-B / 7L-C)
 *
 * Deterministic helpers for improving primary evidence selection precision.
 * Used by EvidenceSelector, Context Builder, Answer Verifier, and Pipeline.
 *
 * Policies:
 * - Start/first/invasion queries → prefer early action docs, demote late/withdrawal
 * - Misconception queries → prefer subject focus over mistaken term
 * - Treaty-specific focus → Paris docs for Paris queries, Genève for Genève
 * - Location queries → reject generic tokens, prefer concrete place names
 * - Focus profiles → domain-specific positive/negative term sets for targeting
 * - Token-aware matching → prevents substring false positives (7L-C)
 *
 * No document IDs. No API calls. No benchmark leakage.
 */

import type { QueryFrame, BaseDocument } from '../shared/types.js';

// ─── Normalization ──────────────────────────────────────────────

export function normalizeForFocus(text: string): string {
  return text
    .toLowerCase()
    .normalize('NFKC')
    .replace(/\s+/g, ' ')
    .trim();
}

// ─── Token-Aware Matching (Patch 7L-C) ──────────────────────────

/**
 * Tokenize Vietnamese text into word tokens.
 * Splits on whitespace and strips punctuation from edges.
 */
export function tokenizeVietnamese(text: string): string[] {
  return normalizeForFocus(text)
    .split(/\s+/)
    .map(t => t.replace(/^[,.";:!?()\[\]{}]+|[,.";:!?()\[\]{}]+$/g, ''))
    .filter(t => t.length > 0);
}

/**
 * Check if a single token exists at a word boundary in the text.
 * Uses space/start/end boundaries for Vietnamese (space-separated syllables).
 * Note: For Vietnamese compound words (e.g., 'hiến pháp' = constitution),
 * use domain-specific functions like hasFranceEntity() instead.
 */
export function hasToken(text: string, token: string): boolean {
  const norm = normalizeForFocus(text);
  const tNorm = normalizeForFocus(token);
  const escaped = tNorm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`(?:^|\\s)${escaped}(?:\\s|$)`, 'u');
  return re.test(norm);
}

/**
 * Check if a multi-word phrase exists as a contiguous sequence in the text.
 * The phrase must appear at word boundaries (space-delimited).
 */
export function hasPhrase(text: string, phrase: string): boolean {
  const norm = normalizeForFocus(text);
  const pNorm = normalizeForFocus(phrase);
  if (pNorm.includes(' ')) {
    // Multi-word: check contiguous presence at boundaries
    const escaped = pNorm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(`(?:^|\\s)${escaped}(?:\\s|$)`, 'u');
    return re.test(norm);
  }
  // Single word: delegate to hasToken
  return hasToken(text, phrase);
}

/**
 * Check if any of the given terms/phrases match in text using token-aware matching.
 * Each term may be a single token or a multi-word phrase.
 */
export function hasAnyTokenOrPhrase(text: string, terms: string[]): boolean {
  return terms.some(t => hasPhrase(text, t));
}

/**
 * Words that when immediately preceding 'pháp' indicate it's NOT the country France.
 * E.g., 'hiến pháp', 'biện pháp', 'giải pháp', 'pháp lý', 'pháp luật'
 */
const PHAP_FALSE_POSITIVE_PREFIXES = [
  'hiến', 'biện', 'giải', 'phương', 'hợp',
];
const PHAP_FALSE_POSITIVE_SUFFIXES = [
  'lý', 'luật', 'quyền', 'chế', 'điển',
];

/**
 * Check if query refers to France entity (not compound words like 'hiến pháp').
 */
export function hasFranceEntity(text: string): boolean {
  const tokens = tokenizeVietnamese(text);
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === 'pháp') {
      const prev = i > 0 ? tokens[i - 1] : '';
      const next = i < tokens.length - 1 ? tokens[i + 1] : '';
      // Check it's not a compound word
      if (PHAP_FALSE_POSITIVE_PREFIXES.includes(prev)) continue;
      if (PHAP_FALSE_POSITIVE_SUFFIXES.includes(next)) continue;
      return true;
    }
  }
  // Also check explicit phrases
  const norm = normalizeForFocus(text);
  return norm.includes('thực dân pháp') || norm.includes('quân pháp') ||
    norm.includes('liên quân pháp') || norm.includes('pháp - tây ban nha') ||
    norm.includes('pháp tây ban nha') || norm.includes('nước pháp') ||
    norm.includes('đế quốc pháp');
}

// ─── Action Lexicons ────────────────────────────────────────────

/** Actions indicating start/beginning/first — used for start queries */
const START_ACTION_TERMS = [
  'nổ súng', 'xâm lược', 'mở đầu', 'đầu tiên', 'bắt đầu',
  'khởi đầu', 'khởi phát', 'mở màn', 'tấn công đầu tiên',
  'tiến công đầu tiên', 'chiếm',
];

/** Actions indicating end/late/withdrawal — demoted for start queries */
const END_ACTION_TERMS = [
  'rút quân', 'kết thúc', 'hoàn toàn', 'cuối cùng', 'chấm dứt',
  'rời khỏi', 'triệt thoái', 'rút khỏi', 'di tản', 'đầu hàng',
];

/** Subject focus terms for common misconception topics */
const MISCONCEPTION_FOCUS_TERMS: Record<string, string[]> = {
  'doi moi': ['đổi mới', 'đại hội vi', 'đại hội 6', '1986', 'cải cách kinh tế'],
  'tuyen ngon doc lap': ['tuyên ngôn độc lập', 'ba đình', '2/9/1945', '2 tháng 9'],
  'dien bien phu': ['điện biên phủ', 'de castries', 'võ nguyên giáp'],
  'cach mang thang tam': ['cách mạng tháng tám', 'tổng khởi nghĩa', '19/8/1945'],
};

/** Treaty name → normalized forms mapping */
const TREATY_FOCUS_MAP: Record<string, string[]> = {
  'paris': ['paris', 'hiệp định paris', 'hiep dinh paris', '1973'],
  'geneve': ['genève', 'geneva', 'geneve', 'hiệp định genève', 'hiep dinh geneve', '1954'],
  'nham_tuat': ['nhâm tuất', 'nham tuat', 'hiệp ước nhâm tuất', '1862'],
  'patenotre': ['patenôtre', 'patenotre', 'hiệp ước patenôtre', '1884'],
  'giap_tuat': ['giáp tuất', 'giap tuat', '1874'],
};

// ─── Focus Profiles (Patch 7L-B) ────────────────────────────────

/**
 * Domain-specific focus profile for precision targeting.
 * Used by pipeline (query expansion), evidence selector (scoring),
 * and answer verifier (repair).
 */
export interface FocusProfile {
  id: string;
  /** Terms whose presence in a doc BOOSTS its relevance */
  positive_terms: string[];
  /** Terms whose presence WITHOUT positive terms DEMOTES the doc */
  negative_terms: string[];
  /** Extra query terms to prepend for retrieval expansion */
  expansion_terms: string;
  /** Terms the answer MUST contain (at least one) for quality */
  answer_must_contain_any?: string[];
}

const FOCUS_PROFILES: Record<string, FocusProfile> = {
  french_invasion_start: {
    id: 'french_invasion_start',
    positive_terms: [
      '1858', 'đà nẵng', 'tourane', 'liên quân pháp', 'tây ban nha',
      'nổ súng', 'xâm lược', 'mở đầu', 'bán đảo sơn trà',
      'rigault de genouilly', 'liên quân pháp - tây ban nha',
    ],
    negative_terms: [
      'hải phòng', '1883', '1946', 'rút quân', 'toàn quốc kháng chiến',
      'kết thúc', '1956', '1954', 'genève', 'điện biên phủ',
      'lời kêu gọi', 'hồ chí minh kêu gọi',
    ],
    expansion_terms: 'Đà Nẵng Tourane 1858 liên quân Pháp Tây Ban Nha nổ súng xâm lược Việt Nam đầu tiên',
    answer_must_contain_any: ['1858', 'đà nẵng', 'tourane', 'sơn trà'],
  },
  doi_moi_misconception: {
    id: 'doi_moi_misconception',
    positive_terms: [
      'đổi mới', 'đại hội vi', 'đại hội 6', '1986', 'cải cách kinh tế',
      'nguyễn văn linh', 'kinh tế thị trường',
    ],
    negative_terms: [
      '1975', 'chiến dịch hồ chí minh', 'giải phóng sài gòn', '30/4/1975',
      '30 tháng 4', 'chiến dịch mùa xuân',
    ],
    expansion_terms: 'Đổi Mới Đại hội VI 1986 cải cách kinh tế',
    answer_must_contain_any: ['đổi mới', '1986', 'đại hội vi', 'đại hội 6'],
  },
  treaty_paris_focus: {
    id: 'treaty_paris_focus',
    positive_terms: [
      'paris', 'hiệp định paris', '1973', 'quân mỹ', 'rút quân',
      'chấm dứt chiến tranh', 'lập lại hòa bình',
      'kissinger', 'lê đức thọ', '27/1/1973', '27 tháng 1',
    ],
    negative_terms: [
      'genève', 'geneva', '1954', 'tổng tuyển cử', 'điện biên phủ',
      'chia cắt hai miền',
    ],
    expansion_terms: 'Hiệp định Paris 1973 quân Mỹ rút khỏi Việt Nam chấm dứt chiến tranh',
    answer_must_contain_any: ['paris', '1973'],
  },
  treaty_geneve_focus: {
    id: 'treaty_geneve_focus',
    positive_terms: [
      'genève', 'geneva', 'giơnevơ', '1954', 'tổng tuyển cử',
      'vĩ tuyến 17', 'đình chiến', 'ngừng bắn',
    ],
    negative_terms: [
      'paris', '1973', 'kissinger', 'lê đức thọ',
    ],
    expansion_terms: 'Hiệp định Genève 1954 đình chiến vĩ tuyến 17',
    answer_must_contain_any: ['genève', 'geneva', 'giơnevơ', '1954'],
  },
};

/**
 * Detect which focus profile applies to this query.
 * Uses token-aware matching (7L-C) to prevent false positives.
 * Returns undefined if no profile matches.
 */
export function detectFocusProfile(query: string, queryFrame?: QueryFrame): FocusProfile | undefined {
  const qNorm = normalizeForFocus(query);

  // French invasion start (7L-C: token-aware France entity detection)
  // Must have REAL France entity, not compound words like 'hiến pháp'
  const hasFrench = hasFranceEntity(query);
  const hasInvasion = qNorm.includes('xâm lược') || qNorm.includes('nổ súng') ||
    qNorm.includes('tấn công đầu tiên') || qNorm.includes('mở đầu xâm lược');
  const hasVietnam = qNorm.includes('việt nam') || qNorm.includes('cảng') ||
    qNorm.includes('đà nẵng') || qNorm.includes('tourane') ||
    qNorm.includes('sơn trà');
  const hasStartContext = qNorm.includes('năm nào') || qNorm.includes('cảng nào') ||
    qNorm.includes('đầu tiên') || qNorm.includes('mở đầu');
  // Require France entity + invasion action, OR France entity + start context + Vietnam target
  if (hasFrench && (hasInvasion || (hasStartContext && hasVietnam))) {
    return FOCUS_PROFILES.french_invasion_start;
  }

  // Đổi Mới misconception: "đổi mới" + misconception marker
  const hasDoiMoi = qNorm.includes('đổi mới');
  const hasMisconception = qNorm.includes('phải không') || qNorm.includes('đúng không') ||
    qNorm.includes('có phải') || qNorm.includes('có đúng là') ||
    queryFrame?.intent === 'misconception_check';
  if (hasDoiMoi && hasMisconception) {
    return FOCUS_PROFILES.doi_moi_misconception;
  }

  // Treaty Paris: mentions Paris but NOT comparison/disambiguation with Genève
  const hasParis = qNorm.includes('hiệp định paris') || hasToken(query, 'paris');
  const hasGeneve = qNorm.includes('genève') || qNorm.includes('geneva') || qNorm.includes('giơnevơ');
  const isComparison = queryFrame?.intent === 'comparison' || queryFrame?.intent === 'disambiguation';
  if (hasParis && !hasGeneve && !isComparison) {
    return FOCUS_PROFILES.treaty_paris_focus;
  }

  // Treaty Genève: mentions Genève but NOT comparison with Paris
  if (hasGeneve && !hasParis && !isComparison) {
    return FOCUS_PROFILES.treaty_geneve_focus;
  }

  return undefined;
}

// ─── Treaty Subtopic Focus (Patch 7M) ───────────────────────────

export type TreatySubtopic =
  | 'signing_location'
  | 'demarcation_line'
  | 'regrouping_transfer'
  | 'general_election'
  | 'us_withdrawal'
  | 'general_treaty'
  | 'unknown';

export interface TreatySubtopicFocus {
  treaty?: 'geneve' | 'paris' | 'nham_tuat' | 'patenotre' | string;
  subtopic: TreatySubtopic;
  positive_terms: string[];
  /** 7M-B: Strong positive terms — required for primary force. Weak terms alone are insufficient. */
  strong_positive_terms?: string[];
  negative_terms: string[];
  answer_must_contain_any: string[];
  expansion_terms: string;
}

/**
 * Detect fine-grained treaty subtopic focus from the query.
 * Returns null if no treaty is mentioned or subtopic is too generic.
 *
 * No hard-coded doc IDs. No case IDs. Uses domain vocabulary only.
 */
export function detectTreatySubtopicFocus(query: string): TreatySubtopicFocus | null {
  const qNorm = normalizeForFocus(query);

  const hasGeneve = qNorm.includes('genève') || qNorm.includes('geneva') ||
    qNorm.includes('giơnevơ') || qNorm.includes('hiệp định genève');
  const hasParis = qNorm.includes('hiệp định paris') || qNorm.includes('paris');

  // ── Genève subtopics ──
  if (hasGeneve && !hasParis) {
    // A. Signing location
    if (qNorm.includes('ký tại đâu') || qNorm.includes('được ký ở đâu') ||
        qNorm.includes('được ký tại') || qNorm.includes('địa điểm ký')) {
      return {
        treaty: 'geneve',
        subtopic: 'signing_location',
        positive_terms: ['genève', 'geneva', 'giơnevơ', 'thụy sĩ'],
        negative_terms: ['vĩ tuyến', 'ranh giới', 'tổng tuyển cử', 'tập kết'],
        answer_must_contain_any: ['genève', 'geneva', 'giơnevơ', 'thụy sĩ'],
        expansion_terms: 'Hiệp định Genève 1954 ký tại Genève Thụy Sĩ',
      };
    }

    // B. Demarcation line
    if (qNorm.includes('ranh giới tạm thời') || qNorm.includes('giới tuyến quân sự') ||
        (qNorm.includes('vĩ tuyến') && !qNorm.includes('ký tại')) ||
        qNorm.includes('chia cắt ở đâu') || qNorm.includes('tạm thời ở đâu')) {
      return {
        treaty: 'geneve',
        subtopic: 'demarcation_line',
        positive_terms: [
          'vĩ tuyến 17', 'sông bến hải', 'giới tuyến quân sự tạm thời',
          'ranh giới tạm thời', 'quảng trị', 'chia cắt',
        ],
        negative_terms: ['tổng tuyển cử', 'paris', '1973'],
        answer_must_contain_any: [
          'vĩ tuyến 17', 'sông bến hải', 'giới tuyến quân sự tạm thời',
          'ranh giới tạm thời', 'quảng trị',
        ],
        expansion_terms: 'vĩ tuyến 17 sông Bến Hải giới tuyến quân sự tạm thời ranh giới tạm thời Hiệp định Genève 1954',
      };
    }

    // C. Regrouping / transfer
    if (qNorm.includes('tập kết') || qNorm.includes('chuyển quân') ||
        qNorm.includes('quân đội hai bên') || qNorm.includes('khu vực tập kết') ||
        qNorm.includes('300 ngày')) {
      return {
        treaty: 'geneve',
        subtopic: 'regrouping_transfer',
        positive_terms: [
          'tập kết', 'chuyển quân', 'đình chiến', 'quân đội hai bên',
          'khu vực tập kết', '300 ngày', 'ngừng bắn',
        ],
        // 7M-B: Strong terms required for primary force — weak terms alone are insufficient
        strong_positive_terms: [
          'tập kết', 'chuyển quân', '300 ngày', 'khu vực tập kết',
          'chuyển ra bắc', 'chuyển vào nam',
        ],
        negative_terms: ['tổng tuyển cử', 'năm 1956', 'paris'],
        answer_must_contain_any: [
          'tập kết', 'chuyển quân', 'đình chiến', 'quân đội hai bên', '300 ngày',
        ],
        expansion_terms: 'tập kết chuyển quân đình chiến quân đội hai bên 300 ngày ngừng bắn Hiệp định Genève 1954',
      };
    }

    // D. General election
    if (qNorm.includes('tổng tuyển cử') || qNorm.includes('thống nhất đất nước') ||
        qNorm.includes('dự kiến năm nào') ||
        (qNorm.includes('1956') && !qNorm.includes('tập kết'))) {
      return {
        treaty: 'geneve',
        subtopic: 'general_election',
        positive_terms: ['tổng tuyển cử', 'thống nhất', '1956', 'dự kiến'],
        negative_terms: ['tập kết', 'chuyển quân', 'paris'],
        answer_must_contain_any: ['tổng tuyển cử', '1956', 'thống nhất'],
        expansion_terms: 'tổng tuyển cử thống nhất đất nước dự kiến năm 1956 Hiệp định Genève 1954',
      };
    }

    // Generic Genève — no specific subtopic detected
    return null;
  }

  // ── Paris subtopics ──
  if (hasParis && !hasGeneve) {
    // A. US withdrawal
    if (qNorm.includes('quân mỹ') || qNorm.includes('rút quân') ||
        qNorm.includes('mỹ rút') || qNorm.includes('chấm dứt chiến tranh') ||
        qNorm.includes('lập lại hòa bình')) {
      return {
        treaty: 'paris',
        subtopic: 'us_withdrawal',
        positive_terms: [
          'paris', '1973', 'quân mỹ', 'rút quân', 'chấm dứt chiến tranh',
          'lập lại hòa bình', 'kissinger', 'lê đức thọ',
        ],
        negative_terms: ['genève', 'tổng tuyển cử', '1956', '1954'],
        answer_must_contain_any: ['paris', '1973', 'rút quân', 'chấm dứt chiến tranh'],
        expansion_terms: 'Hiệp định Paris 1973 quân Mỹ rút quân chấm dứt chiến tranh lập lại hòa bình',
      };
    }

    // Generic Paris — handled by treaty_paris_focus profile
    return null;
  }

  return null;
}

// ─── Timeline Topic Focus (Patch 7M) ────────────────────────────

export interface TimelineTopicFocus {
  topic: string;
  positive_terms: string[];
  negative_terms: string[];
  min_year?: number;
  answer_must_contain_any: string[];
  expansion_terms: string;
}

/**
 * Detect timeline topic focus for topic-coherence enforcement.
 * Returns null if query is not a timeline or has no clear topic.
 *
 * No hard-coded doc IDs. No case IDs.
 */
export function detectTimelineTopicFocus(query: string, queryFrame?: QueryFrame): TimelineTopicFocus | null {
  const qNorm = normalizeForFocus(query);
  const isTimeline = queryFrame?.intent === 'timeline' ||
    qNorm.includes('các mốc') || qNorm.includes('timeline') ||
    qNorm.includes('dòng thời gian') || qNorm.includes('quá trình');

  if (!isTimeline) return null;

  // Đổi Mới timeline
  if (qNorm.includes('đổi mới')) {
    return {
      topic: 'doi_moi',
      positive_terms: [
        'đổi mới', '1986', 'đại hội vi', 'đại hội 6', 'cải cách kinh tế',
        'kinh tế thị trường', 'mở cửa', 'hội nhập', 'nguyễn văn linh',
      ],
      negative_terms: [
        'xô viết nghệ tĩnh', '1930', '1955', 'cần vương',
        'điện biên phủ', '1954',
      ],
      min_year: 1975, // background from 1975, but primary focus is 1986+
      answer_must_contain_any: ['đổi mới', '1986', 'đại hội vi', 'đại hội 6'],
      expansion_terms: 'Đổi Mới Đại hội VI 1986 cải cách kinh tế kinh tế thị trường mở cửa hội nhập',
    };
  }

  return null;
}

/** Generic tokens that should NOT be used as location answers (7L-C: added 'đầu') */
export const GENERIC_LOCATION_TOKENS = [
  'cửa', 'hội', 'đầu', 'thành', 'điểm', // Patch 7L-F: added 'thành', 'điểm'
  'vĩ tuyến', 'khu vực', 'chiến dịch', 'hiệp định',
  'phong trào', 'sự kiện', 'nước', 'miền', 'ranh giới', 'giới tuyến',
  'quân', 'quân đội', 'chính phủ', 'nhân dân', 'đảng', 'mặt trận',
  'bắc', 'nam', 'đông', 'tây', // bare directional words without context
];

/** Known concrete place names for location validation */
export const KNOWN_PLACE_NAMES = [
  'hà nội', 'ba đình', 'sài gòn', 'tp hồ chí minh', 'huế', 'đà nẵng',
  'tourane', 'hải phòng', 'hồng kông', 'hương cảng', 'cửu long',
  'pắc bó', 'cao bằng', 'genève', 'geneva', 'thụy sĩ',
  'điện biên phủ', 'việt bắc', 'biên giới việt trung',
  'khe sanh', 'pleiku', 'tân sơn nhất', 'đà lạt',
  'quảng trị', 'quảng nam', 'bình dương', 'long an',
  'phnom penh', 'paris', 'bắc kinh', 'moskva', 'washington',
  'cần thơ', 'vĩnh long', 'mỹ tho', 'biên hòa',
  'dinh độc lập', 'dinh thống nhất',
  'sơn trà', 'bán đảo sơn trà', // 7L-C
];

/** Location alias groups — any member satisfies any other (7L-C) */
export const LOCATION_ALIAS_GROUPS: string[][] = [
  ['hồng kông', 'hong kong', 'hương cảng'],
  ['genève', 'geneva', 'giơnevơ'],
  ['sài gòn', 'tp hồ chí minh', 'thành phố hồ chí minh'],
  ['đà nẵng', 'tourane'],
];

// ─── Query Focus Extraction ─────────────────────────────────────

export interface QueryFocusResult {
  /** Primary terms the answer MUST address */
  primary_terms: string[];
  /** Action terms from the query (nổ súng, rút quân, ký kết...) */
  action_terms: string[];
  /** Terms that represent a mistaken assumption in misconception queries */
  mistaken_terms: string[];
  /** Treaty focus if query mentions a specific treaty */
  treaty_focus?: 'paris' | 'geneve' | 'nham_tuat' | 'patenotre' | 'giap_tuat' | string;
  /** Matched focus profile for domain-specific targeting (Patch 7L-B) */
  focus_profile?: FocusProfile;
  /** Query wants a location answer */
  wants_location: boolean;
  /** Query wants a date answer */
  wants_date: boolean;
  /** Query is a misconception/yes-no check */
  is_misconception: boolean;
  /** Query asks about start/first/invasion/beginning */
  is_start_or_first_query: boolean;
}

export function extractQueryFocus(input: {
  query: string;
  queryFrame?: QueryFrame;
}): QueryFocusResult {
  const { query, queryFrame } = input;
  const qNorm = normalizeForFocus(query);

  const result: QueryFocusResult = {
    primary_terms: [],
    action_terms: [],
    mistaken_terms: [],
    wants_location: false,
    wants_date: false,
    is_misconception: false,
    is_start_or_first_query: false,
  };

  // ── Detect location want ──
  result.wants_location =
    qNorm.includes('ở đâu') || qNorm.includes('tại đâu') ||
    qNorm.includes('địa điểm nào') || qNorm.includes('cảng nào') ||
    qNorm.includes('được ký tại đâu') || qNorm.includes('ký tại đâu') ||
    qNorm.includes('diễn ra ở') || qNorm.includes('tổ chức ở đâu') ||
    queryFrame?.expected_answer_type === 'location';

  // ── Detect date want ──
  result.wants_date =
    qNorm.includes('năm nào') || qNorm.includes('khi nào') ||
    qNorm.includes('ngày nào') || qNorm.includes('bao giờ') ||
    queryFrame?.expected_answer_type === 'date';

  // ── Detect misconception ──
  result.is_misconception =
    queryFrame?.intent === 'misconception_check' ||
    qNorm.includes('phải không') || qNorm.includes('đúng không') ||
    qNorm.includes('có phải') || qNorm.includes('có đúng là');

  // ── Detect start/first query ──
  const startActions = ['campaign_start', 'invasion', 'attack', 'invasion_or_attack'];
  result.is_start_or_first_query = START_ACTION_TERMS.some(t =>
    qNorm.includes(normalizeForFocus(t))
  ) || (queryFrame?.answer_focus?.action != null && startActions.includes(queryFrame.answer_focus.action));

  // ── Extract action terms ──
  for (const term of [...START_ACTION_TERMS, ...END_ACTION_TERMS]) {
    if (qNorm.includes(normalizeForFocus(term))) {
      result.action_terms.push(term);
    }
  }

  // ── Treaty focus ──
  for (const [key, variants] of Object.entries(TREATY_FOCUS_MAP)) {
    if (variants.some(v => qNorm.includes(normalizeForFocus(v)))) {
      result.treaty_focus = key;
      break;
    }
  }

  // ── Primary terms (subject focus) ──
  // For misconception: extract the subject before "phải không" / "đúng không"
  if (result.is_misconception) {
    // Extract subject: text before the misconception marker
    const miscMarkers = ['phải không', 'đúng không', 'có phải', 'có đúng là'];
    for (const marker of miscMarkers) {
      const idx = qNorm.indexOf(marker);
      if (idx > 0) {
        const subject = qNorm.slice(0, idx).trim();
        // Extract meaningful nouns/entities from subject
        const subjectWords = subject.split(/\s+/).filter(w => w.length > 2);
        result.primary_terms.push(...subjectWords.slice(0, 5));
        break;
      }
    }

    // Extract mistaken terms (year or entity that's incorrect)
    const yearMatch = qNorm.match(/(?:năm|từ năm|từ)\s+(\d{4})/);
    if (yearMatch) {
      result.mistaken_terms.push(yearMatch[1]);
    }

    // For "Đổi Mới bắt đầu từ năm 1975 phải không?"
    // primary_terms = ["đổi", "mới", "bắt", "đầu", "từ", "năm", "1975"]
    // But the real focus is "đổi mới" — check misconception focus map
    for (const [key, focusTerms] of Object.entries(MISCONCEPTION_FOCUS_TERMS)) {
      if (qNorm.includes(key)) {
        // Add the canonical focus terms as primary
        result.primary_terms = [...new Set([...focusTerms, ...result.primary_terms])];
        break;
      }
    }
  }

  // ── Extract key entity terms from query ──
  if (result.primary_terms.length === 0) {
    // Use queryFrame's answer_focus if available
    if (queryFrame?.answer_focus) {
      const focus = queryFrame.answer_focus;
      if (focus.treaty_names) result.primary_terms.push(...focus.treaty_names);
      if (focus.campaign_names) result.primary_terms.push(...focus.campaign_names);
      if (focus.movement_names) result.primary_terms.push(...focus.movement_names);
      if (focus.actor) result.primary_terms.push(...focus.actor);
    }
  }

  // ── Patch 7L-B: Detect focus profile ──
  result.focus_profile = detectFocusProfile(query, queryFrame);

  return result;
}

// ─── Document Focus Scoring ─────────────────────────────────────

export interface DocFocusScore {
  score: number;
  matched_terms: string[];
  missing_terms: string[];
  penalties: string[];
}

/**
 * Score how well a document matches the query focus.
 * Returns a score modifier (can be negative for penalties).
 */
export function scoreDocumentFocus(
  doc: BaseDocument,
  focus: QueryFocusResult
): DocFocusScore {
  const docText = normalizeForFocus(
    `${doc.title} ${doc.summary} ${doc.text_for_embedding || ''}`
  );

  let score = 0;
  const matched: string[] = [];
  const missing: string[] = [];
  const penalties: string[] = [];

  // ── Start/first query: check action alignment ──
  if (focus.is_start_or_first_query) {
    // Check if doc mentions start-like actions
    const hasStartAction = START_ACTION_TERMS.some(t =>
      docText.includes(normalizeForFocus(t))
    );
    if (hasStartAction) {
      score += 0.15;
      matched.push('start_action');
    }

    // Check if doc mentions ONLY end-like actions (penalty)
    const hasEndAction = END_ACTION_TERMS.some(t =>
      docText.includes(normalizeForFocus(t))
    );
    const hasNoStartAction = !hasStartAction;

    if (hasEndAction && hasNoStartAction) {
      score -= 0.20;
      penalties.push('end_action_for_start_query');
    }

    // Prefer earlier year for start/first queries
    if (doc.year) {
      // Documents from earlier periods get small boost for "đầu tiên/mở đầu"
      if (doc.year < 1870) score += 0.05;  // Very early colonial period
      if (doc.year > 1950 && hasNoStartAction) score -= 0.05;  // Late period without start action
    }
  }

  // ── Misconception focus: primary terms over mistaken terms ──
  if (focus.is_misconception) {
    // Check primary focus match
    const primaryMatched = focus.primary_terms.filter(t =>
      docText.includes(normalizeForFocus(t))
    );
    if (primaryMatched.length > 0) {
      score += 0.10 * Math.min(primaryMatched.length, 3);
      matched.push(...primaryMatched.map(t => `focus:${t}`));
    } else {
      missing.push('misconception_subject');
    }

    // Check if doc ONLY matches mistaken term but not subject
    const mistakenMatched = focus.mistaken_terms.filter(t =>
      docText.includes(normalizeForFocus(t))
    );
    if (mistakenMatched.length > 0 && primaryMatched.length === 0) {
      score -= 0.15;
      penalties.push('only_mistaken_term_match');
    }
  }

  // ── Treaty focus: ensure treaty name match ──
  if (focus.treaty_focus) {
    const treatyVariants = TREATY_FOCUS_MAP[focus.treaty_focus];
    if (treatyVariants) {
      const treatyMatched = treatyVariants.some(v =>
        docText.includes(normalizeForFocus(v))
      );
      if (treatyMatched) {
        score += 0.10;
        matched.push(`treaty:${focus.treaty_focus}`);
      } else {
        // Doc doesn't mention the queried treaty → significant penalty as primary
        score -= 0.15;
        penalties.push(`treaty_focus_mismatch:${focus.treaty_focus}`);
        missing.push(focus.treaty_focus);
      }
    }
  }

  // ── Patch 7L-B: Focus profile positive/negative scoring ──
  if (focus.focus_profile) {
    const profile = focus.focus_profile;
    const posMatched = profile.positive_terms.filter(t =>
      docText.includes(normalizeForFocus(t))
    );
    const negMatched = profile.negative_terms.filter(t =>
      docText.includes(normalizeForFocus(t))
    );

    if (posMatched.length > 0) {
      const posBoost = 0.08 * Math.min(posMatched.length, 4);
      score += posBoost;
      matched.push(...posMatched.map(t => `profile+:${t}`));
    }

    if (negMatched.length > 0 && posMatched.length === 0) {
      // Doc matches ONLY negative terms → strong demotion
      const negPenalty = 0.12 * Math.min(negMatched.length, 3);
      score -= negPenalty;
      penalties.push(`profile_negative_only:${negMatched.slice(0, 2).join(',')}`);
    } else if (negMatched.length > posMatched.length) {
      // More negative than positive → mild demotion
      score -= 0.05;
      penalties.push('profile_negative_dominant');
    }
  }

  return { score, matched_terms: matched, missing_terms: missing, penalties };
}

// ─── Location Extraction (Precision) ────────────────────────────

/**
 * Extract a quality location from document, rejecting generic tokens.
 *
 * Priority:
 * 1. doc.place_labels (structured metadata)
 * 2. semantic_features.location (if exists)
 * 3. Title/text pattern: "tại/ở <place>"
 * 4. Known place name detection in text
 * 5. Treaty name as location hint (e.g., Genève)
 *
 * Returns null if only generic tokens found.
 */
export function extractPreciseLocation(doc: BaseDocument): string | null {
  // 1. Structured place_labels (highest priority)
  if (doc.place_labels && doc.place_labels.length > 0) {
    const goodLabels = doc.place_labels.filter(l =>
      !isGenericLocationToken(l)
    );
    if (goodLabels.length > 0) return goodLabels[0];
  }

  const text = doc.text_for_embedding || doc.summary;
  const allText = `${doc.title} ${text}`;
  const textNorm = normalizeForFocus(allText);

  // 2. Known place name scan (Patch 7L-A: moved BEFORE regex to avoid "Hội" from "Hội nghị Hồng Kông")
  for (const place of KNOWN_PLACE_NAMES) {
    if (textNorm.includes(place)) {
      // Return the properly-cased version by searching original text
      const placeRegex = new RegExp(place.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'iu');
      const match = allText.match(placeRegex);
      return match ? match[0] : place;
    }
  }

  // 3. Treaty name as location hint (e.g., title mentions Genève → location is Genève)
  const titleNorm = normalizeForFocus(doc.title);
  if (titleNorm.includes('genève') || titleNorm.includes('geneva') || titleNorm.includes('giơnevơ')) return 'Genève';
  if (titleNorm.includes('paris')) return 'Paris';

  // 4. Title/text pattern extraction (regex fallback)
  const locPatterns = [
    /(?:ký tại|diễn ra tại|tổ chức tại|họp tại|thành lập tại|tại)\s+([\p{L}\s]{3,30}?)(?:[,.\s]|$)/gu,
    /(?:ký ở|diễn ra ở|tổ chức ở|họp ở|thành lập ở|ở)\s+([\p{L}\s]{3,30}?)(?:[,.\s]|$)/gu,
    /(?:tấn công|chiếm|đánh)\s+([\p{L}\s]{3,25}?)(?:[,.\s]|$)/gu,
    /(?:cảng)\s+([\p{L}\s]{3,25}?)(?:[,.\s]|$)/gu,
  ];

  for (const pattern of locPatterns) {
    const matches = allText.matchAll(pattern);
    for (const m of matches) {
      const loc = m[1].trim();
      if (!isGenericLocationToken(loc) && loc.length > 2) {
        return loc;
      }
    }
  }

  return null;
}

/**
 * Check if a location token is too generic to be a useful answer.
 */
export function isGenericLocationToken(loc: string): boolean {
  const norm = normalizeForFocus(loc);
  return GENERIC_LOCATION_TOKENS.some(generic =>
    norm === generic || (norm.length <= generic.length + 2 && norm.includes(generic))
  );
}

/**
 * Check if a location is a treaty content term, not a signing location.
 * E.g., "vĩ tuyến 17" is treaty content, not where it was signed.
 */
export function isTreatyContentNotLocation(loc: string): boolean {
  const norm = normalizeForFocus(loc);
  const treatyContentTerms = [
    'vĩ tuyến', 'ranh giới', 'giới tuyến', 'chia cắt',
    'tổng tuyển cử', 'ngừng bắn', 'rút quân', 'trao trả',
  ];
  return treatyContentTerms.some(t => norm.includes(t));
}
