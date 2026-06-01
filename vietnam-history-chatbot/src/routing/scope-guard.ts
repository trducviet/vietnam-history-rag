/**
 * Scope & Ambiguity Guard — deterministic pre-retrieval filter.
 * Detects out-of-scope queries and ambiguous/vague queries before retrieval.
 *
 * Patch 7J: New file.
 * Patch 9D: Major expansion — lexical collision protection, foreign history,
 *           dynasty/monarch detection, in-scope allowlist, improved normalization.
 */

import type { ScopeGuardResult } from '../shared/types.js';

// ─── Normalization ───────────────────────────────────────────

function norm(text: string): string {
  return text.toLowerCase().normalize('NFKC')
    .replace(/[–—]/g, '-')
    .replace(/\s*-\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// ─── In-Scope Allowlist (Patch 9D) ──────────────────────────
// These terms, when present, strongly indicate the query is about
// Vietnamese modern/contemporary history within corpus scope.
// Check BEFORE any OOS blocking to prevent false positives.

const IN_SCOPE_ALLOWLIST = [
  // Core scope markers (1858-2000 period)
  'pháp xâm lược', 'pháp nổ súng', 'pháp chiếm', 'pháp tấn công',
  'nam kỳ', 'bắc kỳ', 'trung kỳ', 'đông dương',
  'pháp thuộc', 'thuộc địa', 'thực dân pháp',
  // Anti-colonial movements
  'cần vương', 'chiếu cần vương', 'hàm nghi', 'tôn thất thuyết',
  'đông du', 'đông kinh nghĩa thục', 'phan bội châu', 'phan châu trinh',
  'duy tân hội', 'phong trào yêu nước',
  // Party & revolution
  'đảng cộng sản việt nam', 'nguyễn ái quốc', 'hồ chí minh',
  'hội nghị hợp nhất', 'cương lĩnh chính trị',
  'xô viết nghệ tĩnh', 'xô viết nghệ tĩnh',
  // Aug revolution & independence
  'cách mạng tháng tám', 'tổng khởi nghĩa', 'việt minh',
  'tuyên ngôn độc lập', '2 9 1945', 'ba đình',
  // First Indochina war
  'toàn quốc kháng chiến', 'chiến dịch việt bắc',
  'chiến dịch biên giới', 'điện biên phủ',
  'genève', 'giơnevơ', 'hiệp định genève',
  // Vietnam war era
  'đồng khởi', 'mậu thân', 'tết mậu thân',
  'việt nam hóa chiến tranh', 'mỹ rút quân',
  'linebacker', 'điện biên phủ trên không', '12 ngày đêm',
  'đường 9 nam lào', 'lam sơn 719',
  'hiệp định paris', 'paris 1973',
  // Reunification & modern
  'chiến dịch hồ chí minh', 'giải phóng sài gòn', '30 4 1975',
  'thống nhất đất nước',
  'đổi mới', 'bình thường hóa', 'asean', 'apec',
  // Key figures in corpus scope
  'ngô đình diệm', 'võ nguyên giáp', 'lê duẩn', 'trường chinh',
  'phạm văn đồng', 'dương văn minh',
  // Key years in scope
  '1858', '1884', '1930', '1941', '1945', '1946', '1947', '1950',
  '1954', '1964', '1968', '1971', '1972', '1973', '1975',
  '1986', '1995', '1998', '2000',
  // Explicit Vietnam context
  'lịch sử việt nam', 'việt nam dân chủ cộng hòa',
  'cải cách ruộng đất ở việt nam', 'cải cách ruộng đất miền bắc',
  'cải cách ruộng đất tại việt nam', 'cải cách ruộng đất việt nam',
];

// ─── Pre-1858 / Ancient Patterns (Patch 9D expanded) ────────

/** Pre-1858 topics clearly outside corpus scope */
const PRE_1858_KEYWORDS = [
  'bạch đằng', 'năm 938', 'ngô quyền',
  'quang trung', 'tây sơn', 'quân thanh',
  'nhà trần', 'nhà lý', 'nhà lê sơ', 'nhà đinh', 'nhà hồ', 'nhà mạc',
  'tiền lê', 'triệu đà', 'an dương vương', 'hùng vương', 'bắc thuộc',
  'hai bà trưng', 'bà triệu', 'lý thường kiệt', 'trần hưng đạo',
  'nguyên mông', 'quân nguyên',
  'chi lăng', 'xương giang',
];

/** Specific pre-1858 years */
const PRE_1858_YEARS = [938, 1010, 1075, 1288, 1418, 1427, 1428, 1789];

// ─── Lexical Collision Rules (Patch 9D) ─────────────────────
// Detect when a query uses a term that exists in corpus but refers to
// a different historical entity outside corpus scope.

interface CollisionRule {
  /** Terms that trigger this collision check */
  triggers: string[];
  /** If ANY of these are present, query is IN-SCOPE (modern entity) */
  inScopeMarkers: string[];
  /** If ANY of these are present, query is OUT-OF-SCOPE (pre-1858 entity) */
  outOfScopeMarkers: string[];
  /** Reason code for OOS */
  reasonCode: string;
  /** Human reason */
  reason: string;
}

const COLLISION_RULES: CollisionRule[] = [
  {
    triggers: ['lam sơn'],
    inScopeMarkers: ['lam sơn 719', 'đường 9', 'nam lào', '1971', 'vnch', 'quân lực'],
    outOfScopeMarkers: ['lê lợi', 'nhà minh', '1418', '1427', '1428', 'khởi nghĩa lam sơn'],
    reasonCode: 'LEXICAL_COLLISION_OUT_OF_SCOPE',
    reason: 'Khởi nghĩa Lam Sơn (Lê Lợi, thế kỷ XV) nằm ngoài phạm vi dữ liệu. Hệ thống có dữ liệu về chiến dịch Lam Sơn 719 (1971).',
  },
  {
    triggers: ['cải cách ruộng đất'],
    inScopeMarkers: ['việt nam', 'miền bắc', '1953', '1954', '1956', 'bắc việt'],
    outOfScopeMarkers: ['trung quốc', 'mao trạch đông', 'trung hoa'],
    reasonCode: 'FOREIGN_HISTORY_NOT_VIETNAM',
    reason: 'Cải cách ruộng đất ở Trung Quốc nằm ngoài phạm vi. Hệ thống có dữ liệu về cải cách ruộng đất ở miền Bắc Việt Nam.',
  },
];

// ─── Dynasty / Monarch Out-of-Scope (Patch 9D) ─────────────

/** Dynasty/monarch queries about feudal VN history not covered in corpus */
const DYNASTY_PATTERNS = [
  { pattern: 'vua đầu tiên', markers: ['nhà nguyễn', 'nhà lý', 'nhà trần', 'nhà lê'] },
  { pattern: 'lên ngôi', markers: ['gia long', 'minh mạng', 'nhà lý', 'nhà trần'] },
  { pattern: 'dời đô', markers: ['nhà lý', 'thăng long', '1010'] },
];

const DYNASTY_OOS_NAMES = [
  'gia long', 'minh mạng', 'tự đức',
  'nhà lý dời đô', 'nhà trần chống', 'vua trần',
];

/** Dynasty/monarch terms that ARE in-scope (anti-colonial context) */
const DYNASTY_IN_SCOPE_OVERRIDE = [
  'hàm nghi', 'tôn thất thuyết', 'cần vương', 'chiếu cần vương',
  'thành thái', 'duy tân', 'bảo đại',
  'phong trào cần vương', 'chống pháp',
];

// ─── Non-Historical Current Events (Patch 9C4R) ────────────
// Queries about current/live topics with no historical connection.
// Only fire when no named historical entity is present.

const NON_HISTORICAL_CURRENT_KEYWORDS = [
  'giá vàng', 'giá cổ phiếu', 'tỷ giá', 'chứng khoán',
  'thời tiết', 'dự báo thời tiết', 'dự báo',
  'bóng đá', 'lịch thi đấu', 'kết quả bóng đá',
  'tin mới nhất', 'tin tức mới', 'chiến sự hiện nay',
  'hôm nay', 'hôm qua', 'tuần này', 'năm nay',
];

// ─── Fabrication / Hallucination Trap (Patch 9C4R) ──────────
// Queries that encourage the system to fabricate/invent data.

const FABRICATION_TRIGGERS = [
  'cứ suy luận', 'tự suy luận', 'tự đoán',
  'hãy bịa', 'bịa hợp lý', 'đoán số liệu',
  'nếu không có nguồn thì', 'không có nguồn thì',
  'dù nguồn không ghi', 'dù nguồn không có',
  'đừng nói là bịa',
];

// ─── Negative Gap Granular Without Entity (Patch 9C4R) ──────
// Queries requesting granular data explicitly noting source absence.
// Only fire when no named historical entity is present.

const NEGATIVE_GAP_NO_SOURCE_TRIGGERS = [
  'nếu nguồn không có', 'nếu nguồn không nêu',
  'nếu tài liệu không nêu', 'nếu tài liệu không có',
  'nguồn không ghi', 'nguồn không có',
];

const NEGATIVE_GAP_GRANULAR_PATTERNS = [
  'từng xã', 'từng ngày', 'từng người', 'từng đơn vị',
  'danh sách đầy đủ', 'danh sách từng',
  'số liệu chính xác', 'con số chính xác',
  'thống kê từng',
];

// ─── Follow-Up Instruction Without Entity (Patch 9C4R) ──────
// Standalone follow-up instructions with no topic/entity context.

const FOLLOWUP_INSTRUCTION_PATTERNS = [
  'giải thích tiếp', 'phân tích tiếp', 'phân tích thêm',
  'nói tiếp', 'kể tiếp', 'trình bày tiếp',
  'tóm tắt tiếp', 'so sánh tiếp',
];

// ─── Post-2000 Patterns (Patch 9D expanded) ─────────────────

const POST_2000_KEYWORDS = [
  'covid', 'covid 19', 'đại dịch covid',
  'wto', 'gia nhập wto',
  'cptpp', 'tpp',
  'năm 2007', 'năm 2010', 'năm 2015', 'năm 2020', 'năm 2021', 'năm 2022', 'năm 2023', 'năm 2024', 'năm 2025',
  'sau năm 2000', 'thế kỷ 21', 'thế kỷ xxi',
  'mạng xã hội', 'internet hiện đại', 'smartphone',
];

// ─── Foreign History Patterns (Patch 9D expanded) ───────────

/** Non-Vietnam / global-only topics with no VN link */
const FOREIGN_HISTORY_PATTERNS = [
  // European history
  'châu âu', 'nội chiến mỹ', 'cách mạng pháp', 'napoleon', 'waterloo',
  'la mã', 'đế chế la mã', 'chiến tranh thế giới thứ nhất ảnh hưởng gì đến châu',
  'hy lạp cổ đại', 'trung cổ châu âu',
  // Chinese history (not VN-related)
  'trung quốc cổ đại', 'nhà hán', 'nhà đường', 'nhà tống', 'mao trạch đông',
  // Korean war (general)
  'chiến tranh triều tiên', 'hàn quốc', 'bắc triều tiên', 'triều tiên',
  // Japanese history (not VN-related)
  'nhật bản phong kiến', 'samurai', 'shogun', 'meiji',
  // Other
  'ấn độ giành độc lập', 'mahatma gandhi', 'nelson mandela',
];

/** Vietnam-related terms that exempt a query from foreign blocking */
const VN_CONTEXT_TERMS = [
  'việt nam', 'đông dương', 'sài gòn', 'hà nội', 'huế',
  'nguyễn ái quốc', 'hồ chí minh', 'pháp xâm lược',
  'cần vương', 'đông kinh nghĩa thục', 'versailles',
  'nhật đảo chính', 'liên hợp quốc', 'asean', 'apec',
  'phong trào yêu nước', 'lính đông dương',
  'miền bắc việt nam', 'miền nam việt nam',
  'bắc việt', 'nam việt nam', 'vnch', 'vndcch',
  'quân mỹ ở việt nam', 'pháp ở đông dương',
];

// ─── Ambiguity Patterns ─────────────────────────────────────

/** Vague demonstrative pronouns that lack a referent */
const VAGUE_REFERENTS = [
  'nó', 'đó', 'này', 'ấy', 'cái đó',
  'sự kiện đó', 'hiệp định đó', 'chiến dịch đó',
  'sự kiện này', 'hiệp định này', 'chiến dịch này',
  'trận đó', 'cuộc chiến đó',
];

/** Named entities that make a query specific enough even with demonstratives */
const NAMED_ENTITIES = [
  'điện biên phủ', 'genève', 'paris', 'tháng tám', 'hồ chí minh',
  'cách mạng', 'đổi mới', 'asean', 'apec', 'biên giới',
  'mỹ rút quân', 'ký hiệp định', 'tuyên ngôn độc lập',
  'pháp', 'nhật', 'mỹ', 'liên xô', 'trung quốc',
  'xô viết nghệ tĩnh', 'cần vương', 'việt minh',
  'ngô đình diệm', 'võ nguyên giáp', 'nguyễn ái quốc',
  '1858', '1945', '1946', '1954', '1964', '1968', '1973', '1975',
  '1986', '1995', '1930', '1941',
];

/** Ultra-short queries that are inherently ambiguous */
const ULTRA_SHORT_QUERIES = [
  'khi nào', 'ở đâu', 'ai', 'ai thắng', 'ai thua', 'cái gì',
  'là gì', 'thế nào', 'có ý nghĩa gì',
];

// ─── Main Guard ─────────────────────────────────────────────

/**
 * Evaluate scope and ambiguity of a query before retrieval.
 * Deterministic — no API calls.
 *
 * Patch 9D: Order of checks:
 * 0. In-scope allowlist (prevent false positives)
 * 1. Lexical collision detection
 * 2. Pre-1858 keywords
 * 3. Pre-1858 years
 * 4. Post-2000 keywords
 * 5. Post-2000 years
 * 6. Foreign history
 * 7. Dynasty/monarch
 * 7.5 Non-historical current events (Patch 9C4R)
 * 7.6 Fabrication/hallucination trap (Patch 9C4R)
 * 7.7 Negative gap granular without entity (Patch 9C4R)
 * 7.8 Follow-up instruction without entity (Patch 9C4R)
 * 8. Ambiguity / vague
 * 9. Default in_scope
 */
export function evaluateScopeAndAmbiguity(query: string): ScopeGuardResult {
  const q = norm(query);

  // ── 0. In-scope allowlist check (Patch 9D / 9D-R) ──
  // If query contains a strong in-scope marker, likely in-scope —
  // BUT post-2000 years or foreign-only context can override this.
  const matchedAllowlist = IN_SCOPE_ALLOWLIST.find(term => q.includes(term));
  if (matchedAllowlist) {
    // 9D-R: Check for post-2000 year override — "ASEAN năm 2024" is OOS
    const postYearMatch = q.match(/\b(20[0-9]{2})\b/);
    if (postYearMatch) {
      const yr = parseInt(postYearMatch[1], 10);
      if (yr > 2000) {
        return {
          decision: 'out_of_scope',
          reason: `Chủ đề "${matchedAllowlist}" kết hợp với năm ${yr} nằm sau phạm vi dữ liệu (1858–2000).`,
          confidence: 'high',
          matched_patterns: [matchedAllowlist, postYearMatch[1]],
        };
      }
    }

    // 9D-R: Check for post-2000 keyword override — "COVID + Đổi Mới" is OOS
    const postKwMatch = POST_2000_KEYWORDS.find(kw => q.includes(kw));
    if (postKwMatch) {
      return {
        decision: 'out_of_scope',
        reason: `Chủ đề "${postKwMatch}" nằm sau phạm vi dữ liệu (1858–2000).`,
        confidence: 'high',
        matched_patterns: [postKwMatch, matchedAllowlist],
      };
    }

    // 9D-R: Check for foreign-only context override
    // "Hiệp định Paris 1783" has "hiệp định paris" allowlist BUT 1783 is pre-1858
    const preYearMatch = q.match(/\b(1[0-7]\d{2}|180[0-7]|18[0-4]\d|185[0-7])\b/);
    if (preYearMatch) {
      const yr = parseInt(preYearMatch[1], 10);
      if (yr < 1858 && yr > 0) {
        // Check if query also has a valid in-scope year
        const hasInScopeYear = q.match(/\b(18[5-9]\d|19\d{2}|2000)\b/);
        if (!hasInScopeYear) {
          return {
            decision: 'out_of_scope',
            reason: `Năm ${yr} nằm trước phạm vi dữ liệu (1858–2000).`,
            confidence: 'medium',
            matched_patterns: [preYearMatch[1], matchedAllowlist],
          };
        }
      }
    }

    // 9D-R: Check for foreign-only patterns even with allowlist
    // "ASEAN và Liên minh châu Âu khác nhau" — allowlist "asean" but topic is comparative with foreign
    const foreignMatch = FOREIGN_HISTORY_PATTERNS.find(p => q.includes(p));
    if (foreignMatch) {
      // If query has VN context BEYOND just the allowlist keyword, keep in-scope
      // Exclude the matched allowlist term itself to prevent self-reference
      const hasStrongVN = VN_CONTEXT_TERMS.some(v => v !== matchedAllowlist && q.includes(v));
      if (!hasStrongVN) {
        return {
          decision: 'out_of_scope',
          reason: `Câu hỏi về "${foreignMatch}" không liên quan trực tiếp đến lịch sử Việt Nam trong dữ liệu.`,
          confidence: 'medium',
          matched_patterns: [foreignMatch, matchedAllowlist],
        };
      }
    }

    // 9C4R: Check for exhaustive enumeration override even with allowlist
    // "Danh sách từng người tham gia Cách mạng Tháng Tám" has allowlist match
    // but requests per-person list that corpus cannot provide
    const ALLOWLIST_EXHAUSTIVE = ['danh sách từng', 'từng người', 'từng xã', 'từng đơn vị', 'thống kê từng'];
    const allowlistExhaustMatch = ALLOWLIST_EXHAUSTIVE.find(t => q.includes(t));
    if (allowlistExhaustMatch) {
      return {
        decision: 'needs_clarification',
        reason: `Dữ liệu hiện có không có danh sách chi tiết theo "${allowlistExhaustMatch}". Hệ thống có thể cung cấp thông tin tổng quan về sự kiện.`,
        confidence: 'high',
        matched_patterns: [allowlistExhaustMatch, matchedAllowlist],
      };
    }

    // 9C4R: Check for fabrication triggers even with allowlist
    const allowlistFabMatch = FABRICATION_TRIGGERS.find(t => q.includes(t));
    if (allowlistFabMatch) {
      return {
        decision: 'needs_clarification',
        reason: `Hệ thống không thể suy luận hoặc bịa số liệu khi nguồn không ghi nhận. Vui lòng hỏi câu hỏi dựa trên dữ liệu có sẵn.`,
        confidence: 'high',
        matched_patterns: [allowlistFabMatch, matchedAllowlist],
      };
    }

    // 9C4R: Check for no-source triggers even with allowlist
    const allowlistNegSrcMatch = NEGATIVE_GAP_NO_SOURCE_TRIGGERS.find(t => q.includes(t));
    if (allowlistNegSrcMatch) {
      return {
        decision: 'needs_clarification',
        reason: `Câu hỏi yêu cầu dữ liệu mà nguồn hiện có không ghi nhận. Vui lòng hỏi câu hỏi dựa trên dữ liệu có sẵn.`,
        confidence: 'high',
        matched_patterns: [allowlistNegSrcMatch, matchedAllowlist],
      };
    }

    return {
      decision: 'in_scope',
      confidence: 'high',
      matched_patterns: [matchedAllowlist],
    };
  }

  // ── 1. Lexical collision detection (Patch 9D) ──
  for (const rule of COLLISION_RULES) {
    const hasTrigger = rule.triggers.some(t => q.includes(t));
    if (!hasTrigger) continue;

    const hasInScope = rule.inScopeMarkers.some(m => q.includes(m));
    if (hasInScope) continue; // Modern entity → in-scope

    const hasOutOfScope = rule.outOfScopeMarkers.some(m => q.includes(m));
    if (hasOutOfScope) {
      return {
        decision: 'out_of_scope',
        reason: rule.reason,
        confidence: 'high',
        matched_patterns: rule.triggers,
      };
    }
    // Trigger present but no clear markers → fall through to other checks
  }

  // ── 2. Pre-1858 keyword check ──
  for (const kw of PRE_1858_KEYWORDS) {
    if (q.includes(kw)) {
      // Exception: if query also mentions VN modern context
      if (VN_CONTEXT_TERMS.some(v => q.includes(v) && v !== 'việt nam')) continue;
      // Exception: in-scope override (e.g., Hàm Nghi in Cần Vương context)
      if (DYNASTY_IN_SCOPE_OVERRIDE.some(o => q.includes(o))) continue;
      return {
        decision: 'out_of_scope',
        reason: `Chủ đề "${kw}" thuộc giai đoạn trước 1858, nằm ngoài phạm vi dữ liệu.`,
        confidence: 'high',
        matched_patterns: [kw],
      };
    }
  }

  // ── 3. Pre-1858 explicit years ──
  for (const yr of PRE_1858_YEARS) {
    const yrStr = yr.toString();
    if (q.includes(`năm ${yrStr}`) || q.includes(yrStr)) {
      if (VN_CONTEXT_TERMS.some(v => q.includes(v) && v !== 'việt nam')) continue;
      if (DYNASTY_IN_SCOPE_OVERRIDE.some(o => q.includes(o))) continue;
      return {
        decision: 'out_of_scope',
        reason: `Năm ${yr} nằm trước phạm vi dữ liệu (1858–2000).`,
        confidence: 'high',
        matched_patterns: [yrStr],
      };
    }
  }

  // ── 4. Post-2000 keywords ──
  for (const kw of POST_2000_KEYWORDS) {
    if (q.includes(kw)) {
      return {
        decision: 'out_of_scope',
        reason: `Chủ đề "${kw}" nằm sau phạm vi dữ liệu (1858–2000).`,
        confidence: 'high',
        matched_patterns: [kw],
      };
    }
  }

  // ── 5. Post-2000 years from query ──
  const yearMatches = q.matchAll(/\b(\d{4})\b/g);
  for (const match of yearMatches) {
    const year = parseInt(match[1], 10);
    if (year > 2000 && year < 2100) {
      return {
        decision: 'out_of_scope',
        reason: `Năm ${year} nằm sau phạm vi dữ liệu (1858–2000).`,
        confidence: 'high',
        matched_patterns: [match[1]],
      };
    }
    // Pre-1858 year with clear intent
    if (year > 0 && year < 1858) {
      if (q.includes(`năm ${year}`) || q.includes(`vào ${year}`) || q.includes(`thành lập ${year}`)) {
        if (VN_CONTEXT_TERMS.some(v => q.includes(v) && v !== 'việt nam')) continue;
        if (DYNASTY_IN_SCOPE_OVERRIDE.some(o => q.includes(o))) continue;
        return {
          decision: 'out_of_scope',
          reason: `Năm ${year} nằm trước phạm vi dữ liệu (1858–2000).`,
          confidence: 'medium',
          matched_patterns: [match[1]],
        };
      }
    }
  }

  // ── 6. Foreign history (Patch 9D expanded) ──
  for (const pattern of FOREIGN_HISTORY_PATTERNS) {
    if (q.includes(pattern)) {
      const hasVNContext = VN_CONTEXT_TERMS.some(v => q.includes(v));
      if (!hasVNContext) {
        return {
          decision: 'out_of_scope',
          reason: `Câu hỏi về "${pattern}" không liên quan trực tiếp đến lịch sử Việt Nam trong dữ liệu.`,
          confidence: 'medium',
          matched_patterns: [pattern],
        };
      }
    }
  }

  // ── 7. Dynasty/monarch detection (Patch 9D) ──
  // Check if query asks about dynasty/monarch in generic feudal context
  if (DYNASTY_IN_SCOPE_OVERRIDE.some(o => q.includes(o))) {
    // Has an in-scope dynasty term → let through
  } else {
    for (const dp of DYNASTY_PATTERNS) {
      if (q.includes(dp.pattern)) {
        const matchedMarker = dp.markers.find(m => q.includes(m));
        if (matchedMarker) {
          return {
            decision: 'out_of_scope',
            reason: `Câu hỏi về "${matchedMarker}" trong ngữ cảnh triều đại phong kiến nằm ngoài phạm vi dữ liệu hiện có (1858–2000).`,
            confidence: 'medium',
            matched_patterns: [dp.pattern, matchedMarker],
          };
        }
      }
    }
    // Check specific dynasty names
    for (const dn of DYNASTY_OOS_NAMES) {
      if (q.includes(dn)) {
        if (VN_CONTEXT_TERMS.some(v => q.includes(v) && v !== 'việt nam')) continue;
        return {
          decision: 'out_of_scope',
          reason: `Chủ đề "${dn}" thuộc lịch sử triều đại phong kiến, nằm ngoài trọng tâm dữ liệu hiện có (1858–2000).`,
          confidence: 'medium',
          matched_patterns: [dn],
        };
      }
    }
  }

  // ── 7.5 Non-historical current events (Patch 9C4R) ──
  // Queries about live/current topics with no historical entity → out_of_scope
  const hasNamedEntityEarly = NAMED_ENTITIES.some(e => q.includes(e));
  if (!hasNamedEntityEarly) {
    const currentMatch = NON_HISTORICAL_CURRENT_KEYWORDS.find(kw => q.includes(kw));
    if (currentMatch) {
      // Check that query has no historical year (1858-2000)
      const hasHistoricalYear = /\b(18[5-9]\d|19\d{2}|2000)\b/.test(q);
      if (!hasHistoricalYear) {
        return {
          decision: 'out_of_scope',
          reason: `Câu hỏi về "${currentMatch}" không thuộc phạm vi lịch sử Việt Nam (1858–2000).`,
          confidence: 'high',
          matched_patterns: [currentMatch],
        };
      }
    }
  }

  // ── 7.6 Fabrication/hallucination trap (Patch 9C4R) ──
  // Queries encouraging the system to fabricate → needs_clarification (refusal)
  {
    const fabMatch = FABRICATION_TRIGGERS.find(t => q.includes(t));
    if (fabMatch) {
      return {
        decision: 'needs_clarification',
        reason: `Hệ thống không thể suy luận hoặc bịa số liệu khi nguồn không ghi nhận. Vui lòng hỏi câu hỏi dựa trên dữ liệu có sẵn.`,
        confidence: 'high',
        matched_patterns: [fabMatch],
      };
    }
  }

  // ── 7.7 Negative gap granular (Patch 9C4R / 9C4R-fix2) ──
  // Two sub-checks:
  // 7.7a: No-source triggers → block regardless of entity (user explicitly says source lacks data)
  // 7.7b: Granular enumeration ("danh sách từng", "từng người") → block even with entity
  //        because corpus never has per-person/per-village exhaustive lists
  {
    const negSrcMatch = NEGATIVE_GAP_NO_SOURCE_TRIGGERS.find(t => q.includes(t));
    if (negSrcMatch) {
      return {
        decision: 'needs_clarification',
        reason: `Câu hỏi yêu cầu dữ liệu mà nguồn hiện có không ghi nhận. Vui lòng hỏi câu hỏi dựa trên dữ liệu có sẵn.`,
        confidence: 'high',
        matched_patterns: [negSrcMatch],
      };
    }
    // Granular enumeration patterns: block even with entity
    const EXHAUSTIVE_ENUMERATION = ['danh sách từng', 'từng người', 'từng xã', 'từng đơn vị', 'thống kê từng'];
    const exhaustMatch = EXHAUSTIVE_ENUMERATION.find(t => q.includes(t));
    if (exhaustMatch) {
      return {
        decision: 'needs_clarification',
        reason: `Dữ liệu hiện có không có danh sách chi tiết theo "${exhaustMatch}". Hệ thống có thể cung cấp thông tin tổng quan về sự kiện.`,
        confidence: 'high',
        matched_patterns: [exhaustMatch],
      };
    }
    // Other granular patterns without entity
    if (!hasNamedEntityEarly) {
      const negGranMatch = NEGATIVE_GAP_GRANULAR_PATTERNS.find(t => q.includes(t));
      if (negGranMatch) {
        const hasAnchor = IN_SCOPE_ALLOWLIST.some(a => q.includes(a));
        if (!hasAnchor) {
          return {
            decision: 'needs_clarification',
            reason: `Câu hỏi yêu cầu dữ liệu chi tiết mà nguồn hiện có không ghi nhận. Vui lòng nêu sự kiện/chủ đề cụ thể.`,
            confidence: 'high',
            matched_patterns: [negGranMatch],
          };
        }
      }
    }
  }

  // ── 7.8 Follow-up instruction without entity (Patch 9C4R) ──
  // Standalone follow-up commands with no topic context
  if (!hasNamedEntityEarly) {
    const fuMatch = FOLLOWUP_INSTRUCTION_PATTERNS.find(p => q.includes(p));
    if (fuMatch) {
      // Only block if query is short (≤8 tokens) and has no historical anchor
      const tokens = q.split(/\s+/).filter(t => t.length > 0);
      const hasAnchor = IN_SCOPE_ALLOWLIST.some(a => q.includes(a));
      if (tokens.length <= 8 && !hasAnchor) {
        return {
          decision: 'needs_clarification',
          reason: `Câu hỏi "${query}" là yêu cầu tiếp tục mà không nêu chủ đề cụ thể. Vui lòng cho biết sự kiện/chủ đề bạn muốn tìm hiểu.`,
          confidence: 'high',
          matched_patterns: [fuMatch],
        };
      }
    }
  }

  // ── 8. Ambiguity / vague query detection ──
  const hasNamedEntity = NAMED_ENTITIES.some(e => q.includes(e));

  // 8a. Ultra-short query without entity
  if (!hasNamedEntity) {
    const tokens = q.split(/\s+/).filter(t => t.length > 0);
    if (tokens.length <= 4) {
      for (const us of ULTRA_SHORT_QUERIES) {
        if (q.includes(us) || q === us) {
          return {
            decision: 'needs_clarification',
            reason: `Câu hỏi "${query}" quá ngắn và thiếu ngữ cảnh cụ thể.`,
            confidence: 'high',
            matched_patterns: [us],
          };
        }
      }
    }
  }

  // 8b. Vague referent without named entity
  // Patch 9C3R: Short Vietnamese words like "nó" must use whitespace/punctuation
  // boundaries instead of simple includes(), to prevent matching inside longer
  // words like "nói" (to say), "đóng" (to close), "này" inside "những này".
  if (!hasNamedEntity) {
    for (const ref of VAGUE_REFERENTS) {
      // Multi-word refs ("sự kiện đó") are unambiguous with includes()
      // Single/short refs need proper Vietnamese word boundary check
      const isShortRef = !ref.includes(' ');
      const matched = isShortRef
        ? new RegExp(`(?:^|\\s)${ref}(?:\\s|[,;.?!]|$)`, 'i').test(q)
        : q.includes(ref);
      if (matched) {
        return {
          decision: 'needs_clarification',
          reason: `Câu hỏi chứa đại từ "${ref}" mà không kèm tên sự kiện/hiệp định/chiến dịch cụ thể.`,
          confidence: 'high',
          matched_patterns: [ref],
        };
      }
    }
  }

  // ── 9. In scope ──
  return {
    decision: 'in_scope',
    confidence: 'high',
  };
}
