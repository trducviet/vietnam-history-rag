/**
 * Retrieval Reranker — Stage 7E1 post-retrieval reranking for BM25-only mode.
 *
 * Applies deterministic, no-API boosts to improve:
 * 1. Teacher-style / paraphrase query retrieval
 * 2. Cross-period confusion pair handling
 * 3. Rule/comparison doc promotion
 * 4. Period-aware top-1 correction
 *
 * SAFETY: No API calls. No embedding. No data modification.
 * All boosts are additive score adjustments on already-retrieved candidates.
 */

import type { HybridSearchResult, RoutingResult } from '../shared/types.js';

// ─── Teacher-style Intent Detection ──────────────────────────

const TEACHER_STYLE_TRIGGERS = [
  'phân tích', 'chứng minh', 'nhận xét', 'vì sao có thể nói',
  'làm rõ', 'mối quan hệ', 'theo tiêu chí', 'giải thích',
  'trình bày ngắn gọn nhưng đủ ý', 'nếu học sinh', 'cần giải thích',
  'so sánh theo', 'đánh giá', 'bình luận',
];

const TEACHER_EXPANSION_TERMS = [
  'ý nghĩa', 'nguyên nhân', 'kết quả', 'vai trò', 'bối cảnh',
  'diễn biến', 'tác động', 'so sánh', 'khác nhau', 'phân biệt',
];

// ─── Comparison/Rule Intent Detection ────────────────────────

const COMPARISON_TRIGGERS = [
  'khác nhau', 'so sánh', 'phân biệt', 'có phải', 'nhầm',
  'không đồng nghĩa', ' vs ', 'khác gì',
];

// ─── Cross-Period Pair Detection ─────────────────────────────

interface CrossPeriodPair {
  keywords_a: string[];
  keywords_b: string[];
  label: string;
}

const CROSS_PERIOD_PAIRS: CrossPeriodPair[] = [
  { keywords_a: ['genève', 'giơ-ne-vơ', '1954'], keywords_b: ['paris', 'pa-ri', '1973'], label: 'geneve_vs_paris' },
  { keywords_a: ['điện biên phủ', '1954'], keywords_b: ['điện biên phủ trên không', '1972'], label: 'dbp_vs_dbp_air' },
  { keywords_a: ['30/4/1975', '30 4 1975'], keywords_b: ['thống nhất nhà nước', '1976'], label: '30apr_vs_unification' },
  { keywords_a: ['cách mạng tháng tám'], keywords_b: ['quốc khánh', '2/9/1945', '2 9 1945', 'tuyên ngôn độc lập'], label: 'cmtt_vs_qk' },
  { keywords_a: ['cương lĩnh'], keywords_b: ['luận cương'], label: 'cuong_linh_vs_luan_cuong' },
  { keywords_a: ['mậu thân', '1968'], keywords_b: ['tổng tiến công mùa xuân', '1975'], label: 'mauthan_vs_spring75' },
  { keywords_a: ['chiến dịch hồ chí minh'], keywords_b: ['tổng tiến công mùa xuân 1975'], label: 'hcm_vs_spring75' },
  { keywords_a: ['chiến tranh đặc biệt'], keywords_b: ['chiến tranh cục bộ'], label: 'special_vs_limited_war' },
  { keywords_a: ['hội nghị trung ương 6'], keywords_b: ['hội nghị trung ương 7', 'hội nghị trung ương 8'], label: 'htw_678' },
  { keywords_a: ['hậu phương miền bắc', 'miền bắc hậu phương', 'hậu phương'], keywords_b: ['tiền tuyến miền nam', 'miền nam tiền tuyến', 'tiền tuyến'], label: 'north_rear_vs_south_front' },
];

// ─── Year Extraction ─────────────────────────────────────────

function extractYears(text: string): number[] {
  const matches = text.match(/\b(18[5-9]\d|19\d{2}|20[0-2]\d)\b/g);
  return matches ? matches.map(Number) : [];
}

function yearMatchesPeriod(year: number, periodLabel: string | null | undefined): boolean {
  if (!periodLabel) return false;
  const pl = periodLabel.toLowerCase();
  if (year >= 1930 && year <= 1945 && (pl.includes('1930') || pl.includes('august revolution'))) return true;
  if (year >= 1945 && year <= 1954 && (pl.includes('1945') || pl.includes('first indochina') || pl.includes('august revolution'))) return true;
  if (year >= 1954 && year <= 1975 && (pl.includes('1954') || pl.includes('vietnam war') || pl.includes('partition'))) return true;
  return false;
}

// ─── Main Reranker ───────────────────────────────────────────

export interface RerankerDebugInfo {
  teacher_style_detected: boolean;
  comparison_intent_detected: boolean;
  cross_period_pair_detected: string | null;
  query_years: number[];
  expansion_terms_used: string[];
  rule_boost_applied: number;
  period_corrections: number;
  total_score_adjustments: number;
}

/**
 * Apply post-retrieval reranking to improve weak query types.
 * Returns reranked results and debug info for tracing.
 */
export function rerankResults(
  query: string,
  results: HybridSearchResult[],
  routing: RoutingResult,
  topK: number
): { results: HybridSearchResult[]; debug: RerankerDebugInfo } {
  const queryLower = query.toLowerCase().normalize('NFKC');
  const debug: RerankerDebugInfo = {
    teacher_style_detected: false,
    comparison_intent_detected: false,
    cross_period_pair_detected: null,
    query_years: extractYears(query),
    expansion_terms_used: [],
    rule_boost_applied: 0,
    period_corrections: 0,
    total_score_adjustments: 0,
  };

  // Clone results for safe mutation
  const reranked = results.map(r => ({ ...r }));

  // ── A. Teacher-style detection ──────────────────────────
  const isTeacherStyle = TEACHER_STYLE_TRIGGERS.some(t => queryLower.includes(t));
  debug.teacher_style_detected = isTeacherStyle;

  // ── B. Comparison/rule intent detection ──────────────────
  const isComparison = COMPARISON_TRIGGERS.some(t => queryLower.includes(t)) ||
    routing.intent === 'comparison' ||
    routing.query_frame?.intent === 'disambiguation' ||
    routing.query_frame?.intent === 'misconception_check';
  debug.comparison_intent_detected = isComparison;

  // ── C. Cross-period pair detection ──────────────────────
  let detectedPair: CrossPeriodPair | null = null;
  for (const pair of CROSS_PERIOD_PAIRS) {
    const hasA = pair.keywords_a.some(k => queryLower.includes(k));
    const hasB = pair.keywords_b.some(k => queryLower.includes(k));
    if (hasA && hasB) {
      detectedPair = pair;
      debug.cross_period_pair_detected = pair.label;
      break;
    }
  }

  // ── E0. Phrase-match boost for specific topic queries ───
  // Detects rare multi-word phrases in query and boosts docs containing them.
  // This rescues queries like Q014 "giặc đói, giặc dốt" where docs have
  // these phrases in text but not necessarily in title.
  const QUERY_PHRASE_GROUPS = [
    ['giặc đói', 'giặc dốt', 'giặc ngoại xâm', 'nạn đói', 'bình dân học vụ', 'cứu đói'],
    ['ngàn cân treo sợi tóc', 'khó khăn sau', 'thù trong giặc ngoài'],
  ];
  let phraseMatchGroup: string[] | null = null;
  for (const group of QUERY_PHRASE_GROUPS) {
    const matched = group.filter(p => queryLower.includes(p));
    if (matched.length >= 2) { phraseMatchGroup = group; break; }
  }

  // ── Apply boosts ────────────────────────────────────────

  // Get the max score for proportional boosting
  const maxScore = reranked.length > 0 ? reranked[0].combined_score : 1;
  const boostUnit = maxScore * 0.08; // 8% of max score

  for (const r of reranked) {
    let adjustment = 0;
    const titleLower = (r.metadata.title || '').toLowerCase();
    const docType = r.metadata.doc_type || '';

    // B1. Rule/comparison boost when comparison intent detected
    if (isComparison && (docType === 'disambiguation_rule' || docType === 'comparison_note')) {
      // Check relevance: title should contain at least one query keyword
      const queryTokens = queryLower.split(/\s+/).filter(t => t.length > 2);
      const titleTokens = titleLower.split(/\s+/);
      const overlap = queryTokens.filter(qt => titleTokens.some(tt => tt.includes(qt) || qt.includes(tt)));
      if (overlap.length >= 1) {
        adjustment += boostUnit * 2.5; // Strong boost for relevant rules
        debug.rule_boost_applied++;
      }
    }

    // B2. Cross-period pair: boost docs matching both sides
    if (detectedPair) {
      const matchesA = detectedPair.keywords_a.some(k => titleLower.includes(k));
      const matchesB = detectedPair.keywords_b.some(k => titleLower.includes(k));
      if (matchesA && matchesB) {
        adjustment += boostUnit * 3; // Very strong boost for docs with both sides
      } else if (matchesA || matchesB) {
        adjustment += boostUnit * 0.5; // Mild boost for one-sided
      }
    }

    // A. Teacher-style: boost synthesis/explanation docs with expansion term overlap
    if (isTeacherStyle) {
      const expansionOverlap = TEACHER_EXPANSION_TERMS.filter(t => titleLower.includes(t));
      if (expansionOverlap.length > 0) {
        adjustment += boostUnit * 0.5 * expansionOverlap.length;
        for (const term of expansionOverlap) {
          if (!debug.expansion_terms_used.includes(term)) {
            debug.expansion_terms_used.push(term);
          }
        }
      }
      // Also mildly boost synthesis docs for teacher queries
      if (r.metadata.doc_source === 'synthesis') {
        adjustment += boostUnit * 0.3;
      }
    }

    // E0. Phrase-match boost: if query contains 2+ phrases from a group,
    // boost docs whose title or doc_type relate to those phrases
    if (phraseMatchGroup) {
      const phraseOverlap = phraseMatchGroup.filter(p => titleLower.includes(p));
      if (phraseOverlap.length >= 1) {
        adjustment += boostUnit * 1.5 * phraseOverlap.length;
      }
    }

    // D. Period-aware: very mild penalty for clearly wrong-period results
    // Only penalize if query has explicit year AND doc year is very far away
    if (debug.query_years.length > 0 && !detectedPair) {
      const docYear = r.metadata.year;
      const periodLabel = r.metadata.period_label;

      // Only penalize if doc year is far from ALL query years (>40 years)
      if (docYear && debug.query_years.every(qy => Math.abs(docYear - qy) > 40)) {
        adjustment -= boostUnit * 0.3; // Mild penalty only
        debug.period_corrections++;
      }

      // Boost docs whose period matches query years
      if (periodLabel && debug.query_years.some(y => yearMatchesPeriod(y, periodLabel))) {
        adjustment += boostUnit * 0.3;
      }
    }

    r.combined_score += adjustment;
    debug.total_score_adjustments += Math.abs(adjustment) > 0.001 ? 1 : 0;
  }

  // Re-sort after adjustments
  reranked.sort((a, b) => b.combined_score - a.combined_score);

  // ── E. Rule lane minimum guarantee ──────────────────────
  // If comparison intent and no rule in top5, promote best rule from top10
  if (isComparison) {
    const top5 = reranked.slice(0, Math.min(5, topK));
    const hasRuleInTop5 = top5.some(r =>
      r.metadata.doc_type === 'disambiguation_rule' ||
      r.metadata.doc_type === 'comparison_note'
    );
    if (!hasRuleInTop5) {
      const ruleInTop10 = reranked.slice(5, 10).find(r =>
        r.metadata.doc_type === 'disambiguation_rule' ||
        r.metadata.doc_type === 'comparison_note'
      );
      if (ruleInTop10) {
        // Promote: swap with position 4 (5th slot)
        const idx = reranked.indexOf(ruleInTop10);
        if (idx > 4 && idx < 10) {
          const temp = reranked[4];
          reranked[4] = ruleInTop10;
          reranked[idx] = temp;
          debug.rule_boost_applied++;
        }
      }
    }
  }

  return {
    results: reranked.slice(0, topK),
    debug,
  };
}

/**
 * Expand a teacher-style query with additional retrieval terms.
 * Returns expanded query string for BM25 search.
 */
export function expandTeacherQuery(query: string): string {
  const queryLower = query.toLowerCase().normalize('NFKC');
  const isTeacher = TEACHER_STYLE_TRIGGERS.some(t => queryLower.includes(t));
  if (!isTeacher) return query;

  // Add selective expansion terms based on query content
  const additions: string[] = [];

  // If query asks about comparison/difference
  if (queryLower.includes('so sánh') || queryLower.includes('khác') || queryLower.includes('phân biệt')) {
    additions.push('khác nhau', 'so sánh');
  }

  // If query asks about meaning/significance
  if (queryLower.includes('ý nghĩa') || queryLower.includes('vai trò') || queryLower.includes('tác động')) {
    additions.push('ý nghĩa', 'tác động', 'kết quả');
  }

  // If query asks about relationship
  if (queryLower.includes('mối quan hệ') || queryLower.includes('quan hệ')) {
    additions.push('vai trò', 'tác động', 'hỗ trợ', 'chi viện');
  }

  // If query asks to prove/demonstrate
  if (queryLower.includes('chứng minh') || queryLower.includes('nhận xét')) {
    additions.push('ý nghĩa', 'nguyên nhân', 'kết quả', 'diễn biến');
  }

  // Deduplicate and append
  const uniqueAdditions = [...new Set(additions)].filter(a =>
    !queryLower.includes(a)
  );

  if (uniqueAdditions.length === 0) return query;
  return `${query} ${uniqueAdditions.join(' ')}`;
}
