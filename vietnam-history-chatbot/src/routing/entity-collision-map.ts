/**
 * Entity Collision Map (Patch 9E)
 *
 * Generic entity alias + collision registry.
 * Used by query-frame-builder, evidence-selector, answer-verifier
 * to distinguish colliding entities (e.g., ĐBP 1954 vs ĐBP trên không 1972).
 *
 * Design:
 * - ZERO doc_id hard-coding.
 * - ZERO question-id hard-coding.
 * - All matching via normalized text patterns.
 * - Each profile has aliases (positive) + forbidden_aliases (negative).
 */

import { normalizeVietnameseText } from '../indexing/semantic-taxonomy.js';

// ─── Types ────────────────────────────────────────────────────

export interface EntityProfile {
  /** Unique profile ID for logging */
  id: string;
  /** Display name */
  canonical_name: string;
  /** Normalized aliases that confirm this entity */
  aliases: string[];
  /** Normalized phrases that indicate a DIFFERENT (colliding) entity */
  forbidden_aliases: string[];
  /** Expected year for date_lookup validation */
  expected_year?: number;
  /** Year range this entity belongs to */
  expected_year_range?: [number, number];
  /** Actor hints — for actor_lookup on this entity */
  actor_hints?: string[];
  /** Query expansion terms to boost retrieval */
  expansion_terms?: string[];
}

// ─── Registry ─────────────────────────────────────────────────

const ENTITY_PROFILES: EntityProfile[] = [
  // ── ĐBP trên không (1972) vs ĐBP 1954 ──
  {
    id: 'dbp_tren_khong_1972',
    canonical_name: 'Điện Biên Phủ trên không',
    aliases: [
      'dien bien phu tren khong',
      'linebacker ii', 'linebacker 2',
      'ha noi 12 ngay dem', '12 ngay dem',
      'b-52', 'b52',
      'phao dai bay',
    ],
    forbidden_aliases: [
      'doi a1', 'tran doi a1',
      '56 ngay dem',
      '13-3-1954', '7-5-1954',
      '13/3/1954', '7/5/1954',
    ],
    expected_year: 1972,
    expected_year_range: [1972, 1972],
    expansion_terms: [
      'Linebacker II', 'B-52', 'Hà Nội 12 ngày đêm',
      'ném bom rải thảm', '1972', 'Điện Biên Phủ trên không',
    ],
  },
  {
    id: 'dbp_1954',
    canonical_name: 'Chiến dịch Điện Biên Phủ 1954',
    aliases: [
      'chien dich dien bien phu',
      'chien thang dien bien phu',
      'dien bien phu 1954',
      'doi a1', 'tran doi a1',
      '56 ngay dem',
    ],
    forbidden_aliases: [
      'tren khong',
      'linebacker', 'b-52', 'b52',
      '12 ngay dem',
      'phao dai bay',
      'nem bom rai tham',
    ],
    expected_year: 1954,
    expected_year_range: [1953, 1954],
  },

  // ── Chiếu Cần Vương ──
  {
    id: 'chieu_can_vuong',
    canonical_name: 'Chiếu Cần Vương',
    aliases: [
      'chieu can vuong',
      'hich can vuong',
    ],
    forbidden_aliases: [],
    expected_year: 1885,
    actor_hints: ['ham nghi', 'ton that thuyet'],
  },

  // ── Bình thường hóa Việt-Mỹ ──
  {
    id: 'binh_thuong_hoa_viet_my',
    canonical_name: 'Bình thường hóa quan hệ Việt-Mỹ',
    aliases: [
      'binh thuong hoa quan he viet my',
      'binh thuong hoa quan he viet-my',
      'binh thuong hoa viet my',
      'binh thuong hoa',
    ],
    forbidden_aliases: [],
    expected_year: 1995,
  },

  // ── Gia nhập ASEAN (Patch 9E-S: expanded) ──
  {
    id: 'gia_nhap_asean',
    canonical_name: 'Việt Nam gia nhập ASEAN',
    aliases: [
      'gia nhap asean',
      'thanh vien asean',
      'asean 1995',
      'viet nam vao asean',
      'viet nam tro thanh thanh vien asean',
      'hoi nhap khu vuc dong nam a',
      '28-7-1995',
    ],
    forbidden_aliases: [
      'bta', 'wto',
      'hiep dinh thuong mai viet my',
      'mo duong gia nhap wto',
      'apec',
    ],
    expected_year: 1995,
    expected_year_range: [1995, 1995],
    expansion_terms: [
      'ASEAN', 'gia nhập ASEAN', '1995', '28-7-1995',
      'hội nhập khu vực', 'Đông Nam Á', 'thành viên ASEAN',
    ],
  },

  // ── Việt Nam hóa chiến tranh ──
  {
    id: 'viet_nam_hoa_chien_tranh',
    canonical_name: 'Việt Nam hóa chiến tranh',
    aliases: [
      'viet nam hoa chien tranh',
      'hoc thuyet nixon',
      'nixon doctrine',
      'rut quan my',
    ],
    forbidden_aliases: [
      'chien tranh trieu tien',
      'han quoc',
    ],
    expected_year: 1969,
    expected_year_range: [1969, 1973],
  },

  // ── CCRD Việt Nam ──
  {
    id: 'ccrd_viet_nam',
    canonical_name: 'Cải cách ruộng đất ở Việt Nam',
    aliases: [
      'cai cach ruong dat o viet nam',
      'cai cach ruong dat o mien bac',
      'cai cach ruong dat mien bac',
      'cai cach ruong dat viet nam',
    ],
    forbidden_aliases: [
      'trung quoc',
      'lien xo',
    ],
    expected_year: 1953,
    expected_year_range: [1953, 1956],
  },

  // ── Patch 9G: Việt Minh founding ──
  {
    id: 'viet_minh_founding',
    canonical_name: 'Mặt trận Việt Minh',
    aliases: [
      'viet minh',
      'mat tran viet minh',
      'viet nam doc lap dong minh hoi',
      'pac bo',
      'hoi nghi trung uong viii',
      'hoi nghi trung uong 8',
      'thanh lap viet minh',
    ],
    forbidden_aliases: [
      'lien viet',
      'mat tran lien viet',
      'mat tran to quoc',
      'hoi lien hiep quoc dan',
    ],
    expected_year: 1941,
    expected_year_range: [1941, 1941],
    expansion_terms: [
      'Việt Minh', 'Mặt trận Việt Minh', 'Pắc Bó', 'Hội nghị Trung ương VIII',
      'Nguyễn Ái Quốc', '1941', '19-5-1941',
    ],
  },

  // ── Patch 9G: ĐCSVN formation ──
  {
    id: 'dang_csvn_formation',
    canonical_name: 'Đảng Cộng sản Việt Nam thành lập',
    aliases: [
      'dang cong san viet nam',
      'dang cong san',
      'thanh lap dang',
      'hinh thanh dang',
      'hoi nghi hop nhat',
      'hong kong',
      '3-2-1930',
      'dong duong cong san dang',
      'an nam cong san dang',
    ],
    forbidden_aliases: [
      'harmand',
      'patenotre',
      'bao ho',
      'hiep uoc 1883',
      'hiep uoc 1884',
    ],
    expected_year: 1930,
    expected_year_range: [1929, 1930],
    expansion_terms: [
      'Đảng Cộng sản', 'Hội nghị hợp nhất', 'Hồng Kông', 'Nguyễn Ái Quốc',
      '3-2-1930', 'Đông Dương Cộng sản Đảng', 'An Nam Cộng sản Đảng',
    ],
  },

  // ── Patch 9G: ĐBP commander role ──
  {
    id: 'dbp_commander',
    canonical_name: 'Chỉ huy chiến dịch Điện Biên Phủ',
    aliases: [
      'tong tu lenh',
      'chi huy chien dich dien bien phu',
      'chi huy dien bien phu',
      'tuong dien bien',
      'dai tuong dien bien',
    ],
    forbidden_aliases: [],
    expected_year: 1954,
    expected_year_range: [1953, 1954],
    actor_hints: ['vo nguyen giap', 'dai tuong vo nguyen giap', 'tuong giap'],
  },
];

// ─── Matching Functions ───────────────────────────────────────

/**
 * Detect which entity profile(s) a query matches.
 * Returns the best matching profile, or undefined.
 */
export function detectEntityProfile(query: string): EntityProfile | undefined {
  const normQ = normalizeVietnameseText(query);

  let bestProfile: EntityProfile | undefined;
  let bestScore = 0;

  for (const profile of ENTITY_PROFILES) {
    // Count alias matches
    const aliasMatches = profile.aliases.filter(a => normQ.includes(a)).length;
    if (aliasMatches === 0) continue;

    // Check if any forbidden alias also matches — strong disambiguation signal
    const forbiddenMatches = profile.forbidden_aliases.filter(f => normQ.includes(f)).length;
    if (forbiddenMatches > 0) continue; // This profile's forbidden aliases are IN the query → skip

    // Score = alias matches (higher = better fit)
    const score = aliasMatches;
    if (score > bestScore) {
      bestScore = score;
      bestProfile = profile;
    }
  }

  return bestProfile;
}

/**
 * Check if a document title/text matches forbidden aliases for a given profile.
 * Returns penalty score (0 = no collision, 0.4 = strong collision).
 */
export function scoreEntityCollisionPenalty(
  docTitle: string,
  docText: string,
  docYear: number | undefined,
  profile: EntityProfile
): number {
  const normTitle = normalizeVietnameseText(docTitle);
  const normText = normalizeVietnameseText(docText);

  let penalty = 0;

  // Check forbidden aliases in title (strongest signal)
  for (const forbidden of profile.forbidden_aliases) {
    if (normTitle.includes(forbidden)) {
      penalty += 0.3;
    }
  }

  // Check year mismatch
  if (profile.expected_year && docYear && docYear !== profile.expected_year) {
    if (profile.expected_year_range) {
      if (docYear < profile.expected_year_range[0] || docYear > profile.expected_year_range[1]) {
        penalty += 0.2;
      }
    } else {
      penalty += 0.2;
    }
  }

  return Math.min(penalty, 0.5);
}

/**
 * Check if a document matches entity aliases positively.
 * Returns boost score (0 = no match, 0.3 = strong match).
 */
export function scoreEntityAliasBoost(
  docTitle: string,
  docText: string,
  docYear: number | undefined,
  profile: EntityProfile
): number {
  const normTitle = normalizeVietnameseText(docTitle);
  const normText = normalizeVietnameseText(docText);

  let boost = 0;

  // Check aliases in title
  for (const alias of profile.aliases) {
    if (normTitle.includes(alias)) {
      boost += 0.2;
    }
  }

  // Check aliases in text (weaker signal)
  if (boost === 0) {
    for (const alias of profile.aliases) {
      if (normText.includes(alias)) {
        boost += 0.1;
        break; // Only count once for text
      }
    }
  }

  // Year match boost
  if (profile.expected_year && docYear === profile.expected_year) {
    boost += 0.1;
  }

  return Math.min(boost, 0.4);
}

/**
 * Extract actor from document text for actor_lookup.
 * Uses entity profile actor_hints if available, otherwise searches text.
 */
export function extractActorFromDoc(
  docSummary: string,
  docText: string,
  profile?: EntityProfile
): string | undefined {
  // Patch 9E fix: use normalizeVietnameseText to strip diacritics for matching
  const searchText = normalizeVietnameseText(`${docSummary} ${docText}`);

  // Try profile-specific actor hints first
  if (profile?.actor_hints) {
    for (const hint of profile.actor_hints) {
      if (searchText.includes(hint)) {
        // Find the actor name in original text (preserve diacritics)
        const originalText = `${docSummary} ${docText}`;
        const patterns = ACTOR_EXTRACTION_PATTERNS[hint];
        if (patterns) {
          for (const pattern of patterns) {
            const match = originalText.match(pattern);
            if (match) return match[1] || match[0];
          }
        }
        // Fallback: return the canonical form
        return ACTOR_CANONICAL_NAMES[hint] ?? hint;
      }
    }
  }

  // Generic actor extraction from text patterns
  const originalText = `${docSummary} ${docText}`;
  for (const [normName, canonical] of Object.entries(ACTOR_CANONICAL_NAMES)) {
    if (searchText.includes(normName)) {
      // Check if actor is the subject/agent in the text
      const agentPatterns = [
        new RegExp(`${canonical}[^.]*?(?:ban|lãnh đạo|thành lập|ký|chỉ huy|ra lệnh|kêu gọi|tuyên bố|xuống chiếu|nhân danh)`, 'i'),
        new RegExp(`(?:vua|chủ tịch|tổng thống|tướng|đại tướng)\\s+${canonical}`, 'i'),
      ];
      for (const p of agentPatterns) {
        if (p.test(originalText)) return canonical;
      }
    }
  }

  return undefined;
}

// ─── Actor Canonical Names ────────────────────────────────────

const ACTOR_CANONICAL_NAMES: Record<string, string> = {
  'ham nghi': 'Hàm Nghi',
  'ton that thuyet': 'Tôn Thất Thuyết',
  'ho chi minh': 'Hồ Chí Minh',
  'vo nguyen giap': 'Võ Nguyên Giáp',
  'ngo dinh diem': 'Ngô Đình Diệm',
  'phan boi chau': 'Phan Bội Châu',
  'phan chau trinh': 'Phan Châu Trinh',
  'nguyen ai quoc': 'Nguyễn Ái Quốc',
  'tran hung dao': 'Trần Hưng Đạo',
  'le duan': 'Lê Duẩn',
  'truong chinh': 'Trường Chinh',
};

const ACTOR_EXTRACTION_PATTERNS: Record<string, RegExp[]> = {
  'ham nghi': [
    /vua\s+(Hàm Nghi)/i,
    /(Hàm Nghi)\s+ban/i,
    /(Hàm Nghi)/i,
  ],
  'ton that thuyet': [
    /(Tôn Thất Thuyết)/i,
  ],
  'ho chi minh': [
    /[Cc]hủ tịch\s+(Hồ Chí Minh)/i,
    /(Hồ Chí Minh)/i,
    /(Nguyễn Ái Quốc)/i,
  ],
  'vo nguyen giap': [
    /[Đđ]ại tướng\s+(Võ Nguyên Giáp)/i,
    /[Tt]ướng\s+(Võ Nguyên Giáp)/i,
    /(Võ Nguyên Giáp)/i,
  ],
};

/**
 * Get all entity profiles for diagnostic/testing.
 */
export function getAllEntityProfiles(): EntityProfile[] {
  return [...ENTITY_PROFILES];
}
