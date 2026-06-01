/**
 * Query Frame Builder (Patch 7D)
 *
 * Builds a structured QueryFrame from a raw user query string.
 *
 * Design principles:
 * - Deterministic: same query → same frame.
 * - No document IDs anywhere in the output.
 * - answer_focus = what the user wants to find.
 * - contrast_focus = what the user explicitly contrasts against.
 * - Evidence role (primary/supporting/contrast) is NOT assigned here;
 *   that is the EvidenceSelector's job (Patch 7E).
 * - All text matching: normalizeVietnameseText + keyword sets (no regex \b with diacritics).
 * - Reuses normalizeVietnameseText from semantic-taxonomy.ts.
 */

import { normalizeVietnameseText } from '../indexing/semantic-taxonomy.js';
import type {
  QueryFrame,
  QueryFrameIntent,
  ExpectedAnswerType,
  QueryFocus,
} from '../shared/types.js';
import { detectEntityProfile } from './entity-collision-map.js';

// ─── Normalization ────────────────────────────────────────────────────────────

/** Normalize a query for matching: remove diacritics, lowercase, collapse spaces */
export function normalizeQueryText(q: string): string {
  return normalizeVietnameseText(q);
}

// ─── Year Extraction ──────────────────────────────────────────────────────────

/** Extract all 4-digit years (1800–2100) from query string */
export function extractYears(query: string): number[] {
  const matches = query.match(/\b(1[89]\d{2}|20\d{2})\b/g) ?? [];
  return [...new Set(matches.map(Number))].sort((a, b) => a - b);
}

// ─── Named Entity Signals ──────────────────────────────────────────────────────

/**
 * Phrase tables for named entities commonly appearing in Vietnamese history queries.
 * All phrases are already lowercased + diacritic-free (match against normalizeQueryText output).
 */
const TREATY_PHRASES: Record<string, string> = {
  'hiep dinh paris': 'Hiệp định Paris',
  'hiep uoc paris': 'Hiệp định Paris',
  'hiep dinh geneve': 'Hiệp định Genève',
  'hiep dinh gioneve': 'Hiệp định Genève',
  'giong nevo': 'Hiệp định Genève',
  'hiep dinh so bo': 'Hiệp định Sơ bộ 1946',
  'hiep uoc patennote': 'Hiệp ước Patenôtre',
  'hoa uoc giam dinh': 'Hoà ước Giáp Tuất',
};

const CAMPAIGN_PHRASES: Record<string, string> = {
  // Patch 9E: ĐBP trên không MUST come before generic ĐBP to match longest first
  'dien bien phu tren khong': 'Điện Biên Phủ trên không',
  'linebacker ii': 'Chiến dịch Linebacker II',
  'linebacker 2': 'Chiến dịch Linebacker II',
  'ha noi 12 ngay dem': 'Điện Biên Phủ trên không',
  '12 ngay dem': 'Điện Biên Phủ trên không',
  'chien dich dien bien phu': 'Chiến dịch Điện Biên Phủ',
  'chien dich ho chi minh': 'Chiến dịch Hồ Chí Minh',
  'tong tien cong tet mau than': 'Tổng tiến công Tết Mậu Thân 1968',
  'tet mau than': 'Tổng tiến công Tết Mậu Thân 1968',
  'chien dich bien gioi': 'Chiến dịch Biên giới',
  'chien dich viet bac': 'Chiến dịch Việt Bắc',
};

const MOVEMENT_PHRASES: Record<string, string> = {
  'viet minh': 'Mặt trận Việt Minh',
  'mat tran viet minh': 'Mặt trận Việt Minh',
  'dong duong cong san dang': 'Đảng Cộng sản Đông Dương',
  'mat tran giai phong mien nam': 'Mặt trận Giải phóng miền Nam',
  'mat tran giai phong': 'Mặt trận Giải phóng miền Nam',
};

/** Check if norm text includes any of the given tokens as substring */
function includes(normText: string, tokens: string[]): boolean {
  return tokens.some(t => normText.includes(t));
}

/** Extract treaty names mentioned in normalized query */
function detectTreatyNames(normQ: string): string[] {
  const found: string[] = [];
  for (const [phrase, label] of Object.entries(TREATY_PHRASES)) {
    if (normQ.includes(phrase)) found.push(label);
  }
  return [...new Set(found)];
}

/** Extract campaign names mentioned in normalized query */
function detectCampaignNames(normQ: string): string[] {
  const found: string[] = [];
  for (const [phrase, label] of Object.entries(CAMPAIGN_PHRASES)) {
    if (normQ.includes(phrase)) found.push(label);
  }
  return [...new Set(found)];
}

/** Extract movement/org names mentioned in normalized query */
function detectMovementNames(normQ: string): string[] {
  const found: string[] = [];
  for (const [phrase, label] of Object.entries(MOVEMENT_PHRASES)) {
    if (normQ.includes(phrase)) found.push(label);
  }
  return [...new Set(found)];
}

// ─── Action Signal Detection ──────────────────────────────────────────────────

/**
 * Detect the primary action the query is asking about.
 * Returns a SemanticAction-compatible string or undefined.
 * Checked in priority order to avoid ambiguity.
 */
function detectQueryAction(normQ: string): string | undefined {
  // withdrawal: must come before treaty_signing (disambiguation queries include both)
  if (includes(normQ, ['rut quan', 'rut khoi', 'rut toan bo', 'my rut', 'quan my rut'])) {
    return 'withdrawal_or_evacuation';
  }
  // Patch 7L-A: invasion / attack / start — must come before treaty_signing
  if (includes(normQ, ['no sung', 'xam luoc', 'tan cong dau tien', 'mo dau xam luoc', 'bat dau xam luoc'])) {
    return 'invasion_or_attack';
  }
  // treaty_signing: explicit signing verb
  if (includes(normQ, ['ky hiep dinh', 'ky ket', 'duoc ky', 'ngay ky', 'van kien ky'])) {
    return 'treaty_signing';
  }
  // treaty_clause: clause/article content
  if (includes(normQ, ['dieu khoan', 'dieu khoan nao', 'quy dinh', 'cam ket', 'thoa thuan'])) {
    return 'treaty_clause';
  }
  // conference
  if (includes(normQ, ['hoi nghi', 'hoi nghi nao'])) {
    return 'conference';
  }
  // accession
  if (includes(normQ, ['gia nhap', 'tro thanh thanh vien', 'kep nap', 'ket nap'])) {
    return 'accession';
  }
  // normalization
  if (includes(normQ, ['binh thuong hoa', 'lap lai quan he', 'khai thong quan he'])) {
    return 'normalization';
  }
  // boundary_or_division
  if (includes(normQ, ['vi tuyen', 'gioi tuyen', 'chia cat', 'chia doi'])) {
    return 'boundary_or_division';
  }
  // campaign / battle
  if (includes(normQ, ['chien dich nao', 'tran nao', 'mo man chien dich'])) {
    return 'campaign';
  }
  // victory / end
  if (includes(normQ, ['chien thang', 'giai phong', 'dau hang', 'ket thuc chien tranh'])) {
    return 'victory_or_end';
  }
  // founding
  if (includes(normQ, ['thanh lap', 'khai sinh', 'ra doi'])) {
    return 'organization_founding';
  }
  // independence
  if (includes(normQ, ['doc lap', 'tuyen ngon', 'tuyen bo doc lap'])) {
    return 'independence_declaration';
  }
  // election/referendum
  if (includes(normQ, ['tuyen cu', 'tong tuyen cu', 'bau cu'])) {
    return 'election_referendum';
  }
  // reform
  if (includes(normQ, ['doi moi', 'cai cach', 'chuyen sang nen kinh te'])) {
    return 'reform';
  }
  return undefined;
}

// ─── Actor Extraction ─────────────────────────────────────────────────────────

/** Detect actor mentions in the query */
function detectActors(normQ: string): string[] {
  const actors: string[] = [];
  if (includes(normQ, ['my', 'hoa ky', 'quan my', 'my rut', 'tong thong my'])) actors.push('Mỹ');
  if (includes(normQ, ['phap', 'quan phap', 'thuc dan phap'])) actors.push('Pháp');
  if (includes(normQ, ['viet nam dan chu cong hoa', 'nuoc viet nam', 'chinh phu'])) actors.push('Việt Nam DCCH');
  if (includes(normQ, ['ho chi minh'])) actors.push('Hồ Chí Minh');
  if (includes(normQ, ['ngo dinh diem'])) actors.push('Ngô Đình Diệm');
  if (includes(normQ, ['vo nguyen giap', 'tuong giap'])) actors.push('Võ Nguyên Giáp');
  if (includes(normQ, ['nguyen van thieu'])) actors.push('Nguyễn Văn Thiệu');
  if (includes(normQ, ['duong van minh'])) actors.push('Dương Văn Minh');
  if (includes(normQ, ['lbj', 'johnson', 'johnson'])) actors.push('Tổng thống Johnson');
  if (includes(normQ, ['nixon'])) actors.push('Nixon');
  if (includes(normQ, ['de castries', 'tuong de castries'])) actors.push('Tướng De Castries');
  return actors;
}

// ─── Organization Extraction ──────────────────────────────────────────────────

function detectOrganizations(normQ: string): string[] {
  const orgs: string[] = [];
  if (includes(normQ, ['asean'])) orgs.push('ASEAN');
  if (includes(normQ, ['apec'])) orgs.push('APEC');
  if (includes(normQ, ['lhq', 'lien hop quoc'])) orgs.push('LHQ');
  if (includes(normQ, ['imf', 'quy tien te quoc te'])) orgs.push('IMF');
  if (includes(normQ, ['ngan hang the gioi', 'world bank'])) orgs.push('WB');
  if (includes(normQ, ['mat tran viet minh', 'viet minh'])) orgs.push('Mặt trận Việt Minh');
  if (includes(normQ, ['mat tran giai phong'])) orgs.push('Mặt trận Giải phóng miền Nam');
  return orgs;
}

// ─── Location Extraction ──────────────────────────────────────────────────────

function detectLocations(normQ: string): string[] {
  const locs: string[] = [];
  if (includes(normQ, ['dien bien phu'])) locs.push('Điện Biên Phủ');
  if (includes(normQ, ['sai gon'])) locs.push('Sài Gòn');
  if (includes(normQ, ['ha noi'])) locs.push('Hà Nội');
  if (includes(normQ, ['hue'])) locs.push('Huế');
  if (includes(normQ, ['khe sanh'])) locs.push('Khe Sanh');
  if (includes(normQ, ['viet bac'])) locs.push('Việt Bắc');
  if (includes(normQ, ['mien bac', 'bac viet nam'])) locs.push('Miền Bắc');
  if (includes(normQ, ['mien nam', 'nam viet nam'])) locs.push('Miền Nam');
  return locs;
}

// ─── Intent Detection ─────────────────────────────────────────────────────────

/**
 * Explicit date markers. When ANY of these appear in the query,
 * intent must be date_lookup UNLESS an explicit comparison/disambiguation/
 * misconception marker is also present.
 *
 * These markers DOMINATE timeline detection.
 * "Chiến dịch X bắt đầu vào ngày nào?" → date_lookup, not timeline.
 */
const DATE_MARKERS = [
  'khi nao', 'ngay nao', 'nam nao', 'vao nam nao', 'thoi diem nao',
  'bao lau', 'bao gio', 'vao ngay', 'vao thang',
];

/** Contrast / disambiguation markers (event-identification: "sự kiện nào ... khác với") */
const CONTRAST_MARKERS = [
  'khac voi', 'khac gi', 'khac gi voi', 'phan biet', 'diem khac nhau',
  'khac nhau nhu the nao', 'co phai la', 'co phai khong', 'dung khong',
  'khong phai', 'chu khong phai',
];

/** Patch 7K: Expanded comparison markers — catches two-entity comparison patterns */
const COMPARISON_MARKERS = [
  'so sanh', 'khac nhau', 'diem khac nhau', 'giong va khac', 'giong nhau',
  'khac biet', 'tuong dong', 'diem khac biet',
  // Two-entity patterns (Patch 7K)
  'khac gi voi', 'khac gi so voi', 'khac voi',
  'co giong nhau khong', 'co phai cung',
];

const TIMELINE_MARKERS = [
  'neu cac moc',        // "nêu các mốc"
  'cac moc chinh',      // "các mốc chính"
  'dong thoi gian',     // "dòng thời gian"
  'timeline',           // English loanword, clearly timeline intent
  'chuoi su kien',      // "chuỗi sự kiện"
  'theo thu tu',        // "theo thứ tự"
  'trinh bay cac moc',  // "trình bày các mốc"
  'toan bo qua trinh',  // "toàn bộ quá trình" (explicitly full process)
  'cac giai doan',      // "các giai đoạn"
  'tung buoc',          // "từng bước"
  'lich su phat trien', // "lịch sử phát triển"
  'cac buoc',           // "các bước"
  // NOTE: 'dien bien' REMOVED — false positive with place name "Điện Biên Phủ"
  // NOTE: 'qua trinh' REMOVED — too generic, matches any process description
  // NOTE: 'dan toi' REMOVED — matches causal explanations, not timeline structure
];


const MISCONCEPTION_MARKERS = [
  'co phai', 'dung khong', 'phai khong', 'co dung la', 'co phai la',
  'vinh vien dung khong', 'co dung khong',
];

const EXPLANATION_MARKERS = [
  'vi sao', 'tai sao', 'y nghia gi', 'co y nghia nhu the nao', 'tac dong gi',
  'nguyen nhan', 'he qua', 'vai tro', 'buoc ngot', 'buoc ngoat',
];

const CAUSE_EFFECT_MARKERS = [
  'nguyen nhan', 'he qua', 'dan den', 'dan toi', 'ket qua', 'anh huong',
  'tai sao', 'vi sao', 'vi the',
];

/**
 * Detect query intent from normalized query text.
 *
 * Priority order (Patch 7D-2 — precision guard):
 *
 *  0. Date guard: if explicit date marker ("ngày nào", "khi nào", "năm nào", "bao lâu")
 *     is present and NO explicit comparison/disambiguation/misconception marker,
 *     return date_lookup immediately. This prevents timeline/explanation over-routing.
 *
 *  1. Misconception check ("đúng không", "có phải")
 *  2. Disambiguation ("khác với" + event question)
 *  3. Comparison ("so sánh", "điểm khác nhau")
 *
 *  4. Explicit timeline markers (BEFORE explanation/cause_effect):
 *     If query has a clear structural timeline phrase ("nêu các mốc", "chuỗi sự kiện", etc.),
 *     return timeline immediately, even if "dẫn tới" or another cause word is also present.
 *     Example: "Nêu các mốc chính dẫn tới Hiệp định Genève 1954."
 *       → timeline (not cause_effect), because "nêu các mốc chính" is explicit.
 *
 *  5. Explanation/cause AFTER timeline:
 *     "vì sao", "tại sao", "ý nghĩa", "hệ quả" — only reached if NO explicit timeline marker.
 *     Example: "Vì sao chiến thắng Điện Biên Phủ có ý nghĩa quyết định?"
 *       → explanation (no explicit timeline marker present).
 *
 *  6–12. Specific lookup types.
 *  13. Date lookup (if not caught by guard above).
 *  14. Fact lookup (default).
 */
export function detectIntent(normQ: string): QueryFrameIntent {
  // ── Step 0: Date guard ──
  // If query has an explicit date interrogative AND no broad-context markers,
  // short-circuit to date_lookup. This protects against:
  //   "Chiến dịch Điện Biên Phủ bắt đầu vào ngày nào?" → date_lookup (not timeline)
  //   "Kết thúc chiến dịch khi nào?" → date_lookup
  const hasDateMarker = DATE_MARKERS.some(m => normQ.includes(m));
  const hasMisconcMarker = MISCONCEPTION_MARKERS.some(m => normQ.includes(m));
  const hasContrastMarker = CONTRAST_MARKERS.some(m => normQ.includes(m));
  const hasComparisonMarker = COMPARISON_MARKERS.some(m => normQ.includes(m));

  if (hasDateMarker && !hasMisconcMarker && !hasContrastMarker && !hasComparisonMarker) {
    return 'date_lookup';
  }

  // ── Step 1: Misconception check ──
  if (hasMisconcMarker) {
    return 'misconception_check';
  }

  // ── Step 2: Event-identification disambiguation (Patch 7K-A — priority over comparison) ──
  // "Sự kiện nào nói về A, khác với B" is NOT comparison; it's disambiguation.
  // This check MUST come before comparison and ignores hasComparisonMarker.
  const EVENT_ID_MARKERS = [
    'su kien nao noi ve', 'su kien nao mo ta', 'su kien nao danh dau',
    'su kien nao lien quan den', 'su kien nao la',
  ];
  const hasEventIdentification = EVENT_ID_MARKERS.some(m => normQ.includes(m));
  const hasEventQuestion =
    normQ.includes('su kien nao') ||
    normQ.includes('cai gi') ||
    normQ.includes('la gi');

  if (hasEventIdentification && hasContrastMarker) {
    return 'disambiguation';
  }

  // Legacy disambiguation: event question + contrast but NOT comparison
  if (hasContrastMarker && hasEventQuestion && !hasComparisonMarker) {
    return 'disambiguation';
  }

  // ── Step 3: Comparison (Patch 7K — broader detection) ──
  // Contrast markers with two entities but no event-question → comparison
  if (hasComparisonMarker) {
    return 'comparison';
  }
  if (hasContrastMarker && !hasEventQuestion) {
    return 'comparison';
  }

  // ── Step 4: Explicit timeline marker (BEFORE explanation/cause_effect) ──
  // An explicit structural timeline phrase always wins over causal co-occurrence.
  // Rationale: "Nêu các mốc chính dẫn tới Hiệp định Genève 1954" has BOTH
  //   "nêu các mốc chính" (explicit timeline) AND "dẫn tới" (cause marker).
  //   The user is asking for a timeline, not an explanation of causation.
  const hasExplicitTimelineMarker = TIMELINE_MARKERS.some(m => normQ.includes(m));
  if (hasExplicitTimelineMarker) {
    return 'timeline';
  }

  // ── Step 5: Explanation / Cause-effect (only reached if NO explicit timeline marker) ──
  // "vì sao", "tại sao", "ý nghĩa" without timeline structure → explanation.
  // Example: "Vì sao chiến thắng Điện Biên Phủ có ý nghĩa quyết định?" → explanation.
  if (EXPLANATION_MARKERS.some(m => normQ.includes(m))) {
    return 'explanation';
  }
  if (CAUSE_EFFECT_MARKERS.some(m => normQ.includes(m))) {
    return 'cause_effect';
  }

  // ── Step 6: Specific lookup types ──

  // Clause lookup
  if (normQ.includes('dieu khoan nao') || normQ.includes('dieu khoan gi')) {
    return 'clause_lookup';
  }

  // Treaty lookup
  if (normQ.includes('hiep dinh nao') || normQ.includes('hiep uoc nao')) {
    return 'treaty_lookup';
  }

  // Conference lookup
  if (normQ.includes('hoi nghi nao') || normQ.includes('hoi nghi gi')) {
    return 'conference_lookup';
  }

  // Campaign lookup
  if (normQ.includes('chien dich nao') || normQ.includes('tran nao')) {
    return 'campaign_lookup';
  }

  // Sub-event lookup
  if (normQ.includes('su kien nao mo ta') || normQ.includes('su kien nao danh dau')) {
    return 'sub_event_lookup';
  }

  // Movement / organization
  if (normQ.includes('phong trao nao') || normQ.includes('to chuc nao') ||
      normQ.includes('mat tran nao') || normQ.includes('dang nao') ||
      normQ.includes('co quan nao')) {
    return 'organization_lookup';
  }

  // Significance (kept below explanation — don't override explanation)
  if (normQ.includes('y nghia') || normQ.includes('tam quan trong') ||
      normQ.includes('buoc ngoat') ||
      (normQ.includes('ket qua') && normQ.includes('chien tranh'))) {
    return 'significance_lookup';
  }

  // Location
  if (normQ.includes('o dau') || normQ.includes('tai dau') ||
      normQ.includes('dia diem nao') || normQ.includes('dien ra o')) {
    return 'location_lookup';
  }

  // Actor (Patch 9E/9E-R/9E-S: expanded actor patterns)
  if (normQ.includes('ai da') || normQ.includes('ai la') ||
      normQ.includes('nguoi nao') || normQ.includes('tuong nao') ||
      normQ.includes('tong thong nao') || normQ.includes('lanh dao nao') ||
      normQ.includes('chu tich nao') || normQ.includes('vua nao') ||
      normQ.includes('do ai') || normQ.includes('ai ban hanh') ||
      normQ.includes('ai lanh dao') || normQ.includes('ai thanh lap') ||
      normQ.includes('ai ky ') || normQ.includes('ai chi huy') ||
      normQ.includes('ai xuong') || normQ.includes('ai ra chieu') ||
      normQ.includes('gan voi vua') ||
      // Patch 9E-R: "X là ai?" / "người lãnh đạo X là ai?"
      normQ.match(/\bla ai\b/) ||
      // "X do ai?" at end of query
      normQ.match(/\bdo ai\s*[?.!]*$/) ||
      // Patch 9E-S: "do X hay Y ban hành/..." — actor choice pattern
      normQ.match(/\bdo\s+\S+.*?\bhay\b.*?\bban hanh\b/)) {
    return 'actor_lookup';
  }

  // Date (fallback — not caught by guard above)
  if (hasDateMarker) {
    return 'date_lookup';
  }

  return 'fact_lookup';
}


// ─── Expected Answer Type ─────────────────────────────────────────────────────

export function detectExpectedAnswerType(
  intent: QueryFrameIntent,
  normQ: string
): ExpectedAnswerType {
  switch (intent) {
    case 'date_lookup': return 'date';
    case 'actor_lookup': return 'actor';
    case 'actor_date_lookup': return 'actor_date';
    case 'location_lookup': return 'location';
    case 'organization_lookup': return 'organization';
    case 'treaty_lookup': return 'treaty';
    case 'clause_lookup': return 'clause';
    case 'conference_lookup': return 'conference';
    case 'campaign_lookup': return 'campaign';
    case 'cause_effect': return 'cause';
    case 'explanation': return 'meaning';
    case 'significance_lookup': return 'meaning';
    case 'comparison': return 'comparison';
    case 'timeline': return 'timeline';
    case 'misconception_check': return 'yes_no_correction';
    case 'disambiguation': return 'event';
    case 'sub_event_lookup': return 'event';
    case 'movement_lookup': return 'event';
    case 'fact_lookup': {
      // Special case: if fact query about a treaty → treaty answer
      if (normQ.includes('hiep dinh') || normQ.includes('hiep uoc')) return 'treaty';
      return 'event';
    }
    case 'out_of_scope': return 'unknown';
    default: return 'unknown';
  }
}

// ─── Prefer Index ─────────────────────────────────────────────────────────────

function detectPreferIndex(intent: QueryFrameIntent): ('event' | 'synthesis')[] {
  switch (intent) {
    case 'timeline':
    case 'comparison':
    case 'explanation':
    case 'cause_effect':
    case 'significance_lookup':
    case 'misconception_check':
      return ['synthesis', 'event'];
    case 'treaty_lookup':
    case 'clause_lookup':
    case 'conference_lookup':
    case 'campaign_lookup':
    case 'sub_event_lookup':
    case 'date_lookup':
    case 'actor_lookup':
    case 'location_lookup':
    case 'organization_lookup':
    case 'fact_lookup':
    case 'disambiguation':
      return ['event'];
    default:
      return ['event'];
  }
}

// ─── Answer Focus Builder ─────────────────────────────────────────────────────

/**
 * Build answer_focus from the full normalized query text.
 * This describes what the user is asking FOR.
 */
export function buildAnswerFocus(normQ: string, intent: QueryFrameIntent): QueryFocus {
  const focus: QueryFocus = {};

  const action = detectQueryAction(normQ);
  if (action) focus.action = action;

  const actors = detectActors(normQ);
  if (actors.length) focus.actor = actors;

  const orgs = detectOrganizations(normQ);
  if (orgs.length) focus.organization = orgs;

  const locs = detectLocations(normQ);
  if (locs.length) focus.location = locs;

  const treatyNames = detectTreatyNames(normQ);
  if (treatyNames.length) focus.treaty_names = treatyNames;

  const campaignNames = detectCampaignNames(normQ);
  if (campaignNames.length) focus.campaign_names = campaignNames;

  const movementNames = detectMovementNames(normQ);
  if (movementNames.length) focus.movement_names = movementNames;

  // Time signals
  const years = extractYears(normQ);
  if (years.length) {
    focus.time = {
      explicit_years: years,
      year_min: years[0],
      year_max: years[years.length - 1],
    };
  }

  return focus;
}

// ─── Contrast Focus Builder ───────────────────────────────────────────────────

/**
 * Build contrast_focus for disambiguation/comparison queries.
 *
 * Strategy:
 * - Split on contrast markers (khác với, khác gì, phân biệt...)
 * - Treat the clause AFTER the contrast marker as the contrast topic.
 * - Extract action/entity from that clause.
 *
 * Returns undefined if no contrast is detected.
 */
export function buildContrastFocus(query: string, normQ: string): QueryFocus | undefined {
  // Find a contrast marker split point
  const rawLower = query.toLowerCase();
  const contrastMarkerPhrases = [
    'khác với việc', 'khác với', 'khác gì với', 'phân biệt với',
    'không phải là', 'chứ không phải',
  ];

  let contrastClause: string | undefined;
  for (const marker of contrastMarkerPhrases) {
    const idx = rawLower.indexOf(marker);
    if (idx !== -1) {
      contrastClause = rawLower.slice(idx + marker.length).trim();
      break;
    }
  }

  if (!contrastClause) return undefined;

  const normContrast = normalizeVietnameseText(contrastClause);
  const focus: QueryFocus = {};

  const contrastAction = detectQueryAction(normContrast);
  if (contrastAction) focus.action = contrastAction;

  const contrastActors = detectActors(normContrast);
  if (contrastActors.length) focus.actor = contrastActors;

  const contrastTreaties = detectTreatyNames(normContrast);
  if (contrastTreaties.length) focus.treaty_names = contrastTreaties;

  const contrastCampaigns = detectCampaignNames(normContrast);
  if (contrastCampaigns.length) focus.campaign_names = contrastCampaigns;

  // If contrast clause is empty of signals, return undefined
  if (
    !focus.action &&
    !focus.actor?.length &&
    !focus.treaty_names?.length &&
    !focus.campaign_names?.length
  ) {
    return undefined;
  }

  return focus;
}

// ─── Comparison Side Extraction (Patch 7K / 7N-D-A) ──────────────────────────
//
// Expected extractions (self-check reference):
//   "Hiến pháp 1946 khác gì so với Hiến pháp 1959?"
//     → sideA="Hiến pháp 1946", sideB="Hiến pháp 1959"
//   "Gia nhập ASEAN và gia nhập APEC có giống nhau không?"
//     → sideA="Gia nhập ASEAN", sideB="gia nhập APEC"
//   "Mặt trận Việt Minh và Mặt trận Liên Việt có phải cùng một tổ chức không?"
//     → sideA="Mặt trận Việt Minh", sideB="Mặt trận Liên Việt"
//   "Bình thường hóa quan hệ Việt-Mỹ có phải là gia nhập ASEAN không?"
//     → sideA="Bình thường hóa quan hệ Việt-Mỹ", sideB="gia nhập ASEAN"
//   "Hiệp ước Nhâm Tuất và Hòa ước Patenôtre có phải cùng một hiệp ước không?"
//     → sideA="Hiệp ước Nhâm Tuất", sideB="Hòa ước Patenôtre"
//   "Tổng khởi nghĩa tháng Tám 1945 khác gì với Chiến dịch Hồ Chí Minh 1975?"
//     → sideA="Tổng khởi nghĩa tháng Tám 1945", sideB="Chiến dịch Hồ Chí Minh 1975"

/**
 * Extract two comparison sides from a query using known Vietnamese markers.
 * Returns null if query is not a two-entity comparison.
 */
export function extractComparisonSides(
  query: string
): { side_a: string; side_b: string; marker: string; comparison_dimension?: string } | null {
  const raw = query.trim();

  // ── Pattern 0: "A và B có phải cùng một X không?" ──
  // MUST run FIRST — before any generic "có phải" pattern.
  // Handles: cùng một, cùng là, là cùng một
  const cungMotMatch = raw.match(
    /^(.+?)\s+và\s+(.+?)\s+có\s+phải\s+(?:là\s+)?(?:cùng\s+(?:là\s+)?(?:một\s+)?)/i
  );
  if (cungMotMatch) {
    return { side_a: cungMotMatch[1].trim(), side_b: cungMotMatch[2].trim(), marker: 'có phải cùng' };
  }

  // ── Pattern 0b (Patch 9E-R): "A và B có phải một không?" ──
  // "Vua Hàm Nghi và vua đầu nhà Nguyễn có phải một không?"
  // → sideA = "Vua Hàm Nghi", sideB = "vua đầu nhà Nguyễn"
  // Key: "và" splits A/B, "có phải một không" is the question tail
  const vaCoPhaiMotMatch = raw.match(
    /^(.+?)\s+và\s+(.+?)\s+có\s+phải\s+(?:là\s+)?một\s+không\s*[?.!]*$/i
  );
  if (vaCoPhaiMotMatch) {
    const sA = vaCoPhaiMotMatch[1].trim();
    const sB = vaCoPhaiMotMatch[2].trim();
    if (sA.length > 2 && sB.length > 2) {
      return { side_a: sA, side_b: sB, marker: 'có phải một' };
    }
  }

  // ── Pattern 1: "A <marker> B" — explicit comparison markers ──
  // Patch 9C: Added 'khác so với', 'khác biệt gì so với', 'khác biệt với'
  const splitMarkers = [
    'khác gì so với', 'khác biệt gì so với', 'khác gì với',
    'khác so với', 'khác biệt với', 'khác với',
    'so sánh', 'phân biệt',
  ];
  for (const marker of splitMarkers) {
    const idx = raw.toLowerCase().indexOf(marker);
    if (idx !== -1) {
      const before = raw.slice(0, idx).trim().replace(/^[\s,]+|[\s,?!.]+$/g, '');
      let after = raw.slice(idx + marker.length).trim().replace(/^[\s,]+|[\s,?!.]+$/g, '');
      // Patch 9C: Strip trailing question phrases like 'như thế nào', 'ở điểm nào', 'ở chỗ nào'
      after = after.replace(/\s*(?:như thế nào|ở điểm nào|ở chỗ nào|ra sao)\s*[?.!]*$/i, '').trim();
      // Patch 9H: Skip if 'before' is a leading question phrase (not a valid sideA)
      const beforeLower = before.toLowerCase();
      if (marker === 'phân biệt' && /(?:vì sao|tại sao|làm sao|như thế nào|cần)/.test(beforeLower)) {
        continue; // Let Pattern 7 handle this
      }
      if (before.length > 2 && after.length > 2) {
        return { side_a: before, side_b: after, marker };
      }
    }
  }

  // ── Pattern 5 (Patch 9C): "A khác B như thế nào?" / "A khác B ở điểm nào?" ──
  // Patch 9G: Pattern 5b first — "A khác B ở/về X (như thế nào)?" with dimension
  const khacDimMatch = raw.match(/^(.+?)\s+khác\s+(.+?)\s+(?:ở|về)\s+(.+?)(?:\s+(?:như thế nào|ra sao))?[?.!]*$/i);
  if (khacDimMatch) {
    const sA = khacDimMatch[1].trim();
    const sB = khacDimMatch[2].trim();
    const dim = khacDimMatch[3].trim().replace(/\s*(?:như thế nào|ra sao)\s*$/i, '').trim();
    // Patch 9H: If sideB is 'nhau', this is Pattern 6 ("khác nhau"), not Pattern 5
    if (sA.length > 2 && sB.length > 2 && dim.length > 1 && sB.toLowerCase() !== 'nhau') {
      return { side_a: sA, side_b: sB, marker: 'khác...ở/về', comparison_dimension: dim };
    }
  }

  const khacNTNMatch = raw.match(/^(.+?)\s+khác\s+(.+?)\s+(?:như thế nào|ở điểm nào|ở chỗ nào|ra sao)\s*[?.!]*$/i);
  if (khacNTNMatch) {
    const sA = khacNTNMatch[1].trim();
    const sB = khacNTNMatch[2].trim();
    // Patch 9H: If sideB starts with 'nhau', this is Pattern 6
    if (sA.length > 2 && sB.length > 2 && !sB.toLowerCase().startsWith('nhau')) {
      return { side_a: sA, side_b: sB, marker: 'khác...như thế nào' };
    }
  }

  // ── Pattern 6 (Patch 9C): "A và B khác nhau như thế nào?" ──
  // Patch 9G: Pattern 6b first — "A và B khác nhau ở/về X (như thế nào)?" with dimension
  const vaKhacDimMatch = raw.match(/^(.+?)\s+và\s+(.+?)\s+khác\s+nhau\s+(?:ở|về)\s+(.+?)(?:\s+(?:như thế nào|ra sao))?[?.!]*$/i);
  if (vaKhacDimMatch) {
    const sA = vaKhacDimMatch[1].trim();
    const sB = vaKhacDimMatch[2].trim();
    const dim = vaKhacDimMatch[3].trim().replace(/\s*(?:như thế nào|ra sao)\s*$/i, '').trim();
    if (sA.length > 2 && sB.length > 2 && dim.length > 1) {
      return { side_a: sA, side_b: sB, marker: 'và...khác nhau ở/về', comparison_dimension: dim };
    }
  }

  const vaKhacMatch = raw.match(/^(.+?)\s+và\s+(.+?)\s+khác\s+nhau\s*(?:như thế nào|ở điểm nào|ở chỗ nào|ra sao)?[?.!]*$/i);
  if (vaKhacMatch) {
    const sA = vaKhacMatch[1].trim();
    const sB = vaKhacMatch[2].trim();
    if (sA.length > 2 && sB.length > 2) {
      return { side_a: sA, side_b: sB, marker: 'và...khác nhau' };
    }
  }

  // ── Pattern 2: "So sánh A và B" / "Phân biệt A với B" ──
  const soSanhMatch = raw.match(/^(?:so sánh|phân biệt)\s+(.+?)\s+(?:và|với)\s+(.+?)\s*[?.!]*$/i);
  if (soSanhMatch) {
    return { side_a: soSanhMatch[1].trim(), side_b: soSanhMatch[2].trim(), marker: 'so sánh/phân biệt' };
  }

  // ── Pattern 3: "A và B có giống nhau không?" ──
  const giongMatch = raw.match(/^(.+?)\s+và\s+(.+?)\s+có\s+giống\s+nhau/i);
  if (giongMatch) {
    return { side_a: giongMatch[1].trim(), side_b: giongMatch[2].trim(), marker: 'có giống nhau' };
  }

  // ── Pattern 3b (Patch 9E-R): "A có giống B không?" / "A có khác B không?" ──
  // No "và" — A is before "có giống/khác", B is after
  const giongNoVaMatch = raw.match(/^(.+?)\s+có\s+(?:giống|khác)\s+(.+?)\s+không\s*[?.!]*$/i);
  if (giongNoVaMatch) {
    const sA = giongNoVaMatch[1].trim();
    const sB = giongNoVaMatch[2].trim();
    if (sA.length > 2 && sB.length > 2) {
      return { side_a: sA, side_b: sB, marker: 'có giống/khác' };
    }
  }

  // ── Pattern 4: "A có phải là B không?" / "A có phải B không?" ──
  const coPhaiMatch = raw.match(/^(.+?)\s+có\s+phải\s+(?:là\s+)?(.+?)\s+không\s*[?.!]*$/i);
  if (coPhaiMatch) {
    const sideBRaw = coPhaiMatch[2].trim();
    // Strip relation fillers from start of sideB
    const sideBCleaned = sideBRaw
      .replace(/^(?:sự kiện|nội dung|tổ chức|phong trào|hiệp ước|hiệp định|một)\s+/i, '')
      .trim();
    const finalSideB = sideBCleaned || sideBRaw;
    // Patch 9E-R: Validate sideB is not a garbage stopword
    const SIDE_STOPWORDS = ['một', 'không', 'phải', 'là', 'cùng', 'cùng một', 'đúng', 'vậy', 'thế'];
    if (SIDE_STOPWORDS.includes(finalSideB.toLowerCase()) || finalSideB.length <= 2) {
      // sideB is garbage — do NOT return this extraction
      // Fall through to next patterns
    } else {
      return { side_a: coPhaiMatch[1].trim(), side_b: finalSideB, marker: 'có phải' };
    }
  }



  // ── Pattern 7 (Patch 9G-R2): "Vì sao cần phân biệt giữa X và Y?" ──
  const viSaoPhanBietMatch = raw.match(/(?:vì sao|tại sao)\s+(?:cần\s+)?phân biệt\s+(?:giữa\s+)?(.+?)\s+và\s+(.+?)\s*[?.!]*$/i);
  if (viSaoPhanBietMatch) {
    const sA = viSaoPhanBietMatch[1].trim();
    const sB = viSaoPhanBietMatch[2].trim();
    if (sA.length > 2 && sB.length > 2) {
      return { side_a: sA, side_b: sB, marker: 'vì sao phân biệt' };
    }
  }

  // ── Pattern 8 (Patch 9G-R2): "Sự khác nhau giữa X và Y là gì?" ──
  const suKhacNhauMatch = raw.match(/sự\s+khác\s+nhau\s+giữa\s+(.+?)\s+và\s+(.+?)\s+là\s+gì\s*[?.!]*$/i);
  if (suKhacNhauMatch) {
    const sA = suKhacNhauMatch[1].trim();
    const sB = suKhacNhauMatch[2].trim();
    if (sA.length > 2 && sB.length > 2) {
      return { side_a: sA, side_b: sB, marker: 'sự khác nhau giữa' };
    }
  }

  return null;
}

// ─── Comparison Side Term Expansion (Patch 7N-B) ─────────────────────────────

/**
 * Domain alias registry for comparison side expansion.
 * Key: normalized keyword. Value: expanded terms for retrieval/matching.
 * No doc IDs. No case IDs.
 */
const SIDE_ALIASES: Array<{ keywords: string[]; expansions: string[] }> = [
  {
    keywords: ['asean', 'gia nhập asean'],
    expansions: ['asean', 'gia nhập asean', 'việt nam gia nhập asean', '1995', 'hội nhập khu vực', 'đông nam á'],
  },
  {
    keywords: ['apec', 'gia nhập apec'],
    expansions: ['apec', 'gia nhập apec', 'việt nam gia nhập apec', '1998', 'châu á thái bình dương', 'diễn đàn hợp tác kinh tế'],
  },
  {
    keywords: ['tổng khởi nghĩa tháng tám', 'cách mạng tháng tám'],
    expansions: ['tổng khởi nghĩa tháng tám', 'cách mạng tháng tám', 'tháng tám 1945', '1945', 'việt minh giành chính quyền', 'giành chính quyền'],
  },
  {
    keywords: ['chiến dịch hồ chí minh'],
    expansions: ['chiến dịch hồ chí minh', '1975', 'giải phóng sài gòn', 'tổng tiến công giải phóng', '30/4', '30-4-1975'],
  },
  {
    keywords: ['hiến pháp 1946'],
    // Patch 9C-R Final: removed 'việt nam dân chủ cộng hòa' — shared with 1959, causes false both-side
    expansions: ['hiến pháp 1946', 'hiến pháp năm 1946', '1946', 'hiến pháp đầu tiên'],
  },
  {
    keywords: ['hiến pháp 1959'],
    // Patch 9C-R Final: removed shared 'việt nam dân chủ cộng hòa'
    expansions: ['hiến pháp 1959', 'hiến pháp năm 1959', '1959', 'hiến pháp nước việt nam dân chủ cộng hòa năm 1959', 'xây dựng chủ nghĩa xã hội', 'miền bắc'],
  },
  {
    keywords: ['cần vương'],
    expansions: ['cần vương', 'hàm nghi', 'tôn thất thuyết', 'phò vua', 'chống pháp', 'văn thân', 'cuối thế kỷ xix'],
  },
  {
    keywords: ['xô viết nghệ tĩnh', 'nghệ tĩnh', 'xô viết nghệ - tĩnh', 'xô viết nghệ-tĩnh'],
    // Patch 9C-R Final: added hyphen/spacing variants for Nghệ Tĩnh
    expansions: ['xô viết nghệ tĩnh', 'xô viết nghệ - tĩnh', 'xô viết nghệ-tĩnh', 'nghệ tĩnh', '1930', '1931', 'công nông', 'chính quyền xô viết', 'phong trào xô viết nghệ tĩnh'],
  },
  {
    keywords: ['bình thường hóa quan hệ việt mỹ', 'bình thường hóa quan hệ việt-mỹ', 'quan hệ việt mỹ', 'quan hệ việt-mỹ'],
    expansions: ['bình thường hóa', 'quan hệ việt mỹ', 'quan hệ việt-mỹ', 'hoa kỳ', 'mỹ'],
  },
  {
    keywords: ['việt minh', 'mặt trận việt minh'],
    expansions: ['việt minh', 'mặt trận việt minh', '1941', 'pắc bó', 'nguyễn ái quốc'],
  },
  {
    keywords: ['liên việt', 'mặt trận liên việt'],
    expansions: ['liên việt', 'mặt trận liên việt', 'hội liên hiệp quốc dân'],
  },
  {
    keywords: ['nhâm tuất', 'hiệp ước nhâm tuất'],
    expansions: ['nhâm tuất', 'hiệp ước nhâm tuất', '1862', 'ba tỉnh miền đông nam kỳ'],
  },
  {
    keywords: ['patenôtre', 'hòa ước patenôtre'],
    expansions: ['patenôtre', 'hòa ước patenôtre', '1884', 'bảo hộ'],
  },
  {
    keywords: ['genève', 'geneva', 'giơnevơ', 'hiệp định genève'],
    expansions: ['genève', 'geneva', 'giơnevơ', 'hiệp định genève', 'hội nghị genève', '1954'],
  },
  {
    keywords: ['paris', 'hiệp định paris'],
    expansions: ['paris', 'hiệp định paris', '1973', 'quân mỹ rút', 'chấm dứt chiến tranh', 'lập lại hòa bình'],
  },
  // Patch 8D: Additional aliases for side classification
  {
    keywords: ['biên giới 1950', 'biên giới thu đông'],
    expansions: ['biên giới 1950', 'biên giới thu đông', 'đường số 4', 'đông khê', 'cao bằng', 'thất khê', 'chiến dịch biên giới'],
  },
  {
    keywords: ['điện biên phủ trên không', '12 ngày đêm'],
    expansions: ['điện biên phủ trên không', 'hà nội 12 ngày đêm', 'linebacker ii', 'b-52', '1972', '12 ngày đêm'],
  },
  {
    keywords: ['chiến thắng điện biên phủ', 'điện biên phủ 1954'],
    expansions: ['điện biên phủ', 'chiến thắng điện biên phủ', '7-5-1954', '1954', 'de castries', 'pháp đầu hàng'],
  },
  // Patch 9C: Additional aliases for comparison side coverage
  {
    keywords: ['khởi nghĩa yên thế', 'yên thế'],
    expansions: ['khởi nghĩa yên thế', 'yên thế', 'hoàng hoa thám', 'đề thám', '1884', '1913'],
  },
  {
    keywords: ['phong trào cần vương'],
    expansions: ['cần vương', 'phong trào cần vương', 'hàm nghi', 'tôn thất thuyết', '1885', '1896'],
  },
  {
    keywords: ['hiệp định sơ bộ', 'sơ bộ 6-3'],
    expansions: ['hiệp định sơ bộ', 'sơ bộ 6-3-1946', '6-3-1946', '1946', 'pháp công nhận'],
  },
  {
    keywords: ['tạm ước', 'tạm ước 14-9'],
    expansions: ['tạm ước', 'tạm ước 14-9-1946', 'modus vivendi', '14-9-1946', 'nhân nhượng'],
  },
  {
    keywords: ['cương lĩnh chính trị đầu tiên', 'cương lĩnh chính trị'],
    expansions: ['cương lĩnh chính trị đầu tiên', 'cương lĩnh', 'nguyễn ái quốc', 'hội nghị thành lập đảng', '2-1930', 'chính cương vắn tắt'],
  },
  {
    keywords: ['luận cương chính trị', 'luận cương 10-1930'],
    expansions: ['luận cương chính trị', 'luận cương', 'trần phú', '10-1930', 'hội nghị trung ương lần thứ nhất'],
  },
  {
    keywords: ['chiến dịch việt bắc', 'việt bắc thu đông', 'việt bắc 1947'],
    expansions: ['chiến dịch việt bắc', 'việt bắc thu đông 1947', '1947', 'thu đông 1947', 'pháp tấn công việt bắc'],
  },
  {
    keywords: ['tết mậu thân', 'tổng tiến công tết mậu thân', 'mậu thân 1968'],
    expansions: ['tết mậu thân', 'tổng tiến công tết mậu thân 1968', '1968', 'tết nguyên đán 1968', 'tổng tiến công và nổi dậy'],
  },
  {
    keywords: ['tuyên ngôn độc lập', 'tuyên ngôn'],
    expansions: ['tuyên ngôn độc lập', '2-9-1945', 'hồ chí minh đọc tuyên ngôn', 'ba đình'],
  },
  {
    keywords: ['đường 9 nam lào', 'đường 9', 'lam sơn 719'],
    expansions: ['đường 9 nam lào', 'đường 9 - nam lào', '1971', 'lam sơn 719', 'nam lào'],
  },
  {
    keywords: ['chiến dịch hồ chí minh 1975'],
    expansions: ['chiến dịch hồ chí minh', '1975', 'giải phóng sài gòn', 'tổng tiến công', '30/4', '30-4-1975'],
  },
];

/**
 * Expand a comparison side string into a list of matching terms for retrieval.
 * Uses domain alias registry. Returns unique lowercase terms.
 */
/**
 * Patch 9C-R Final: Normalize Vietnamese text for matching.
 * Handles hyphen variants, spacing, case, and NFKC normalization.
 */
export function normalizeVietnamesePhrase(input: string): string {
  return input
    .toLowerCase()
    .normalize('NFKC')
    .replace(/[–—]/g, '-')           // normalize dash variants
    .replace(/\s*-\s*/g, ' ')         // remove spaces around hyphens, replace hyphen with space
    .replace(/\s+/g, ' ')             // collapse multiple spaces
    .trim();
}

export function expandComparisonSideTerms(side: string): string[] {
  const sideNorm = normalizeVietnamesePhrase(side);
  const results = new Set<string>();

  // Always include the full normalized side label (important for multi-word matching)
  if (sideNorm.length > 3) results.add(sideNorm);

  // Always include original tokens (but skip very short ones)
  for (const token of sideNorm.split(/\s+/).filter(t => t.length > 2)) {
    results.add(token);
  }

  // Match against alias registry (also normalize keywords)
  for (const entry of SIDE_ALIASES) {
    const matched = entry.keywords.some(kw => sideNorm.includes(normalizeVietnamesePhrase(kw)));
    if (matched) {
      for (const exp of entry.expansions) {
        results.add(normalizeVietnamesePhrase(exp));
      }
    }
  }

  return [...results];
}

// ─── Constraint Builder ───────────────────────────────────────────────────────

function buildConstraints(
  intent: QueryFrameIntent,
  answerFocus: QueryFocus,
  contrastFocus: QueryFocus | undefined
): QueryFrame['constraints'] {
  const constraints: QueryFrame['constraints'] = {
    prefer_index: detectPreferIndex(intent),
  };

  // must_include_semantics: derived from answer_focus action
  if (answerFocus.action) {
    constraints.must_include_semantics = [answerFocus.action];
  }

  // must_not_be_about: derived from contrast_focus action
  // Patch 7K: skip must_not_be_about for comparison — both sides are wanted
  if (contrastFocus?.action && intent !== 'comparison') {
    constraints.must_not_be_about = [contrastFocus.action];
  }

  // requires_contrast
  if (intent === 'comparison' || intent === 'disambiguation') {
    constraints.requires_contrast = true;
  }

  // requires_correction
  if (intent === 'misconception_check') {
    constraints.requires_correction = true;
  }

  return constraints;
}

// ─── Confidence Scoring ───────────────────────────────────────────────────────

function scoreConfidence(
  intent: QueryFrameIntent,
  answerFocus: QueryFocus
): 'low' | 'medium' | 'high' {
  let score = 0;
  if (intent !== 'fact_lookup') score += 1;   // non-default intent = more signals
  if (answerFocus.action) score += 2;
  if (answerFocus.treaty_names?.length) score += 2;
  if (answerFocus.campaign_names?.length) score += 2;
  if (answerFocus.actor?.length) score += 1;
  if (answerFocus.time?.explicit_years?.length) score += 1;

  if (score >= 5) return 'high';
  if (score >= 2) return 'medium';
  return 'low';
}

// ─── Main Builder ─────────────────────────────────────────────────────────────

/**
 * Build a structured QueryFrame from a raw user query.
 *
 * This is the main export. Called by query-router.ts after base routing.
 * No document IDs in output. No external calls. Deterministic.
 */
export function buildQueryFrame(query: string): QueryFrame {
  const normQ = normalizeQueryText(query);
  const intent = detectIntent(normQ);
  const expected_answer_type = detectExpectedAnswerType(intent, normQ);
  const answer_focus = buildAnswerFocus(normQ, intent);
  const contrast_focus = buildContrastFocus(query, normQ);
  const constraints = buildConstraints(intent, answer_focus, contrast_focus);
  const confidence = scoreConfidence(intent, answer_focus);

  // Patch 7K: Extract comparison sides
  // Patch 8D: Extract for ANY intent — two-sided queries may be routed as multi_hop/fact_lookup
  const comparison_sides = extractComparisonSides(query) ?? undefined;

  // Patch 9E: Entity collision detection
  const entityProfile = detectEntityProfile(query);
  const entity_profile_field = entityProfile ? {
    id: entityProfile.id,
    canonical_name: entityProfile.canonical_name,
    expected_year: entityProfile.expected_year,
    expansion_terms: entityProfile.expansion_terms,
    actor_hints: entityProfile.actor_hints,
  } : undefined;

  const reasoning: string[] = [`intent=${intent}`];
  if (answer_focus.action) reasoning.push(`answer.action=${answer_focus.action}`);
  if (contrast_focus?.action) reasoning.push(`contrast.action=${contrast_focus.action}`);
  if (answer_focus.treaty_names?.length) reasoning.push(`treaty=${answer_focus.treaty_names.join(',')}`);
  if (answer_focus.time?.explicit_years?.length) reasoning.push(`years=${answer_focus.time.explicit_years.join(',')}`);
  if (comparison_sides) reasoning.push(`comparison: [${comparison_sides.side_a}] vs [${comparison_sides.side_b}]`);
  if (entityProfile) reasoning.push(`entity_profile=${entityProfile.id}`);

  return {
    intent,
    answer_focus,
    ...(contrast_focus && { contrast_focus }),
    expected_answer_type,
    constraints,
    ...(comparison_sides && { comparison_sides }),
    ...(entity_profile_field && { entity_profile: entity_profile_field }),
    confidence,
    reasoning,
  };
}
