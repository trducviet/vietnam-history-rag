/**
 * Semantic Taxonomy Configuration (Patch 7C-0C)
 *
 * Design rules:
 * - NO regex with Vietnamese diacritics as core matching logic.
 * - NO \b word boundaries.
 * - Use normalized (no-diacritics, lowercase) phrase lists only.
 * - NO evidence_role / is_hard_negative at document level (query-dependent).
 * - evidence_quality here = source/verification quality only.
 */

// ─── Core Domain Types ────────────────────────────────────────────────────────

export type SemanticDomain =
  | 'military'
  | 'politics'
  | 'diplomacy'
  | 'administration'
  | 'economy'
  | 'society';

// ─── Semantic Actions (30+) ───────────────────────────────────────────────────

export type SemanticAction =
  // Military
  | 'invasion_or_occupation'
  | 'resistance'
  | 'uprising'
  | 'movement'
  | 'campaign'
  | 'campaign_start'
  | 'battle'
  | 'victory_or_end'
  | 'withdrawal_or_evacuation'
  // Diplomacy
  | 'treaty_related'
  | 'treaty_signing'
  | 'treaty_clause'
  | 'conference'
  | 'negotiation'
  | 'normalization'
  | 'accession'
  // Politics & Administration
  | 'institution_founding'
  | 'organization_founding'
  | 'state_founding'
  | 'independence_declaration'
  | 'law_constitution'
  | 'election_referendum'
  | 'leadership_change'
  | 'coup_or_assassination'
  | 'boundary_or_division'
  | 'territorial_change'
  // Economy & Society
  | 'reform'
  | 'economic_program'
  | 'crisis'
  | 'policy'
  // Synthesis
  | 'timeline_synthesis'
  | 'comparison_synthesis'
  | 'person_profile'
  | 'period_overview';

// ─── Answer Affordances (22) ──────────────────────────────────────────────────

export type AnswerAffordance =
  | 'fact_lookup'
  | 'date_lookup'
  | 'actor_lookup'
  | 'actor_date_lookup'
  | 'location_lookup'
  | 'organization_lookup'
  | 'treaty_lookup'
  | 'clause_lookup'
  | 'conference_lookup'
  | 'significance_lookup'
  | 'campaign_lookup'
  | 'sub_event_lookup'
  | 'movement_lookup'
  | 'explanation'
  | 'cause_effect'
  | 'comparison'
  | 'timeline'
  | 'disambiguation'
  | 'misconception_check'
  | 'entity_profile'
  | 'multi_hop'
  | 'out_of_scope';

// ─── Semantic Features Interface ──────────────────────────────────────────────

/**
 * Document-level semantic features.
 * MUST NOT contain query-dependent logic (evidence roles, hard-negative flags).
 * evidence_quality here = source/verification quality only.
 */
export interface SemanticFeatures {
  domains: SemanticDomain[];
  actions: SemanticAction[];
  answer_affordances: AnswerAffordance[];
  /** Key topic strings for BM25 boosting, extracted from title + keywords */
  topics: string[];
  /** Aliases and variant names for this document */
  aliases: string[];
  treaty_names?: string[];
  campaign_names?: string[];
  movement_names?: string[];
  time: {
    year_min: number | null;
    year_max: number | null;
    period_label: string | null;
  };
  /** Source-level quality signals — NOT evidence role for a specific query */
  evidence_quality: {
    verification_status?: 'verified' | 'reviewed' | 'unverified';
    significance_level?: string;
    has_sources?: boolean;
  };
  flags: {
    is_planned_not_executed: boolean;
    is_treaty_related: boolean;
    is_treaty_signing: boolean;
    is_treaty_clause: boolean;
    is_withdrawal_or_evacuation: boolean;
    is_foundation: boolean;
    is_accession: boolean;
    is_normalization: boolean;
    is_boundary_or_division: boolean;
    is_campaign_start: boolean;
    is_victory_or_end: boolean;
  };
}

// ─── Normalization Utilities ──────────────────────────────────────────────────

/**
 * Normalize Vietnamese text:
 * 1. NFD decompose
 * 2. Remove combining diacritical marks
 * 3. Replace đ/Đ → d/D
 * 4. Lowercase
 * 5. Collapse whitespace
 * 6. Trim
 *
 * Safe for all Vietnamese diacritics. No \b dependency.
 */
export function normalizeVietnameseText(input: string): string {
  return input
    .normalize('NFD')
    // Remove combining diacritical marks
    .replace(/[\u0300-\u036f]/g, '')
    // Replace đ/Đ before lowercasing
    .replace(/đ/g, 'd')
    .replace(/Đ/g, 'D')
    .toLowerCase()
    // Replace any non-alphanumeric character (punctuation, special chars) with a space
    // This ensures "ky hiep dinh," still matches phrase "ky hiep dinh"
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

/**
 * Normalize an array of phrases in-place.
 * Returns a new readonly array of normalized strings.
 */
export function normalizePhraseList(
  phrases: readonly string[]
): readonly string[] {
  return phrases.map(normalizeVietnameseText);
}

/**
 * Check if a normalized text contains a normalized phrase as a whole word/token.
 * Uses space/punctuation boundaries instead of \b.
 */
export function hasPhrase(
  normalizedText: string,
  normalizedPhrase: string
): boolean {
  // Pad with spaces to allow simple includes with boundary check
  const padded = ` ${normalizedText} `;
  const target = ` ${normalizedPhrase} `;
  return padded.includes(target);
}

/**
 * Check if a normalized text contains ANY phrase from the normalized list.
 */
export function hasAnyPhrase(
  normalizedText: string,
  normalizedPhrases: readonly string[]
): boolean {
  return normalizedPhrases.some((p) => hasPhrase(normalizedText, p));
}

// ─── Controlled Phrase Lists ──────────────────────────────────────────────────

/**
 * All phrase values are pre-normalized (no diacritics, lowercase).
 * These are used by the extractor at index-time, not at query-time.
 *
 * IMPORTANT DISTINCTIONS:
 * - "hiep dinh" alone = treaty_related topic only (not treaty_signing)
 * - "ky hiep dinh", "ky ket hiep dinh", etc. = treaty_signing action
 * - "my rut quan", "rut toan bo quan" = withdrawal_or_evacuation (NOT treaty_signing)
 * - "gia nhap" alone ≠ accession unless combined with org context
 */
export const CONTROLLED_PHRASES = {
  action: {
    treatySigning: normalizePhraseList([
      'ky hiep dinh',
      'ky ket hiep dinh',
      'duoc ky',
      'da duoc ky',
      'phe chuan hiep dinh',
      'chinh thuc ky',
      'le ky ket',
    ]),
    treatyClause: normalizePhraseList([
      'dieu khoan',
      'khoan',
      'dieu 2',
      'dieu 3',
      'dieu 4',
      'khoan b',
      'muc',
      'phu luc',
    ]),
    withdrawalOrEvacuation: normalizePhraseList([
      'rut quan',
      'rut khoi',
      'rut toan bo',
      'rut het',
      'quan my rut',
      'hoa ky rut',
      'my rut quan',
      'di tan',
      'tan cu',
      'chuyen quan',
      'bang lieu',
    ]),
    uprising: normalizePhraseList([
      'khoi nghia',
      'noi day',
      'khoi binh',
      'bieu tinh vu trang',
      'phan loan',
    ]),
    normalization: normalizePhraseList([
      'binh thuong hoa',
      'thiet lap quan he',
      'noi lai quan he',
      'khoi phuc quan he',
    ]),
    accession: normalizePhraseList([
      'gia nhap asean',
      'gia nhap apec',
      'gia nhap wto',
      'gia nhap lhq',
      'tro thanh thanh vien',
      'chinh thuc gia nhap',
      'ket nap',           // fixed typo from kep nap
    ]),
    campaignStart: normalizePhraseList([
      'mo man',
      'bat dau chien dich',
      'khai hoa',
      'phat dong chien dich',
      'tan cong lan dau',
      'mo dau',
    ]),
    victoryOrEnd: normalizePhraseList([
      'chien thang',
      'dau hang',
      'giai phong',
      'ket thuc chien dich',
      'toan thang',
      'tan thu',
    ]),
    foundingState: normalizePhraseList([
      'thanh lap nuoc',
      'tuyen ngon doc lap',
      'khai sinh nuoc',
      'nen cong hoa',
      'nuoc viet nam dan chu',
    ]),
  },
  topic: {
    treatyRelated: normalizePhraseList([
      'hiep dinh',
      'hiep uoc',
      'hoa uoc',
      'geneve',
      'paris',
      'ky kết',
      'van kien',
      'thoa uoc',
    ]),
    militaryRelated: normalizePhraseList([
      'quan su',
      'chien tranh',
      'chien dich',
      'tran danh',
      'tiep quan',
      'quan doi',
    ]),
    politicsRelated: normalizePhraseList([
      'chinh tri',
      'chinh phu',
      'chinh quyen',
      'dang cong san',
      'dang csv',
      'bo chinh tri',
    ]),
    parisAgreement: normalizePhraseList([
      'hiep dinh paris',
      'hiep uoc paris',
      'hoa dam paris',
    ]),
    genevaAccords: normalizePhraseList([
      'hiep dinh geneve',
      'hoi nghi geneve',
      'hiep uoc geneve',
    ]),
    usWithdrawal: normalizePhraseList([
      'my rut quan',
      'rut quan my',
      'quan my roi viet nam',
      'my cham dut tham chien',
    ]),
    doiMoi: normalizePhraseList([
      'doi moi',
      'doi moi kinh te',
      'chinh sach doi moi',
    ]),
  },
} as const;

// ─── Static Taxonomy Mappings ─────────────────────────────────────────────────

/** Map from doc_type / doc_kind → SemanticDomain[] */
export const DOC_TYPE_TO_DOMAIN_MAP: Record<string, SemanticDomain[]> = {
  campaign: ['military'],
  battle: ['military'],
  uprising: ['military', 'politics'],
  movement: ['politics', 'society'],
  process: ['politics', 'society'],
  reform: ['politics', 'economy', 'society'],
  program: ['economy', 'administration'],
  treaty: ['diplomacy'],
  conference: ['diplomacy', 'politics'],
  institution_founding: ['politics', 'administration'],
  organization_founding: ['politics', 'administration'],
  law_constitution: ['politics'],
  policy: ['administration', 'politics'],
  crisis: ['politics', 'military', 'economy'],
  withdrawal_evacuation: ['military', 'diplomacy'],
  accession: ['diplomacy'],
  normalization: ['diplomacy'],
  event: ['politics'], // generic fallback
};

/** Map from doc_kind (synthesis) → AnswerAffordance[] */
export const DOC_KIND_TO_AFFORDANCE_MAP: Record<string, AnswerAffordance[]> = {
  period_overview: ['timeline', 'explanation', 'cause_effect'],
  timeline_summary: ['timeline', 'date_lookup', 'fact_lookup'],
  comparison: ['comparison', 'explanation'],
  person_profile: ['entity_profile', 'actor_lookup', 'fact_lookup'],
  treaty_explainer: ['treaty_lookup', 'explanation', 'clause_lookup'],
  cause_effect: ['cause_effect', 'explanation'],
  significance_note: ['significance_lookup', 'explanation'],
};
