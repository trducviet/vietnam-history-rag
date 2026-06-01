/**
 * Vietnamese Text Normalization Utilities — Stage 12C
 *
 * Pure deterministic functions — no network, no LLM, no imports beyond stdlib.
 *
 * Covers:
 * - Lowercasing & whitespace collapse
 * - Accent removal (đ/d + full vowel decomposition)
 * - Repeated-character normalization (dienn→dien, phuu→phu)
 * - Date preservation (30/4, 2/9)
 * - Token variant generation for fuzzy matching
 */

// ─── Accent Tables ────────────────────────────────────────────

const ACCENT_MAP: Record<string, string> = {
  // a
  à: 'a', á: 'a', ạ: 'a', ả: 'a', ã: 'a',
  â: 'a', ầ: 'a', ấ: 'a', ậ: 'a', ẩ: 'a', ẫ: 'a',
  ă: 'a', ằ: 'a', ắ: 'a', ặ: 'a', ẳ: 'a', ẵ: 'a',
  // e
  è: 'e', é: 'e', ẹ: 'e', ẻ: 'e', ẽ: 'e',
  ê: 'e', ề: 'e', ế: 'e', ệ: 'e', ể: 'e', ễ: 'e',
  // i
  ì: 'i', í: 'i', ị: 'i', ỉ: 'i', ĩ: 'i',
  // o
  ò: 'o', ó: 'o', ọ: 'o', ỏ: 'o', õ: 'o',
  ô: 'o', ồ: 'o', ố: 'o', ộ: 'o', ổ: 'o', ỗ: 'o',
  ơ: 'o', ờ: 'o', ớ: 'o', ợ: 'o', ở: 'o', ỡ: 'o',
  // u
  ù: 'u', ú: 'u', ụ: 'u', ủ: 'u', ũ: 'u',
  ư: 'u', ừ: 'u', ứ: 'u', ự: 'u', ử: 'u', ữ: 'u',
  // y
  ỳ: 'y', ý: 'y', ỵ: 'y', ỷ: 'y', ỹ: 'y',
  // d
  đ: 'd',
};

/**
 * Remove Vietnamese diacritics including đ→d.
 * Result is always ASCII-safe lowercase.
 */
export function removeVietnameseAccents(input: string): string {
  return input
    .toLowerCase()
    .split('')
    .map(ch => ACCENT_MAP[ch] ?? ch)
    .join('');
}

/**
 * Basic normalisation: lowercase, collapse whitespace, trim.
 * Preserves date patterns like 30/4, 2/9, 7/5/1954.
 */
export function normalizeText(input: string): string {
  return input
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Collapse duplicated consonants common in VN typos.
 * Conservative — only repeats of ≥3 identical chars or known patterns.
 */
export function collapseRepeatedChars(input: string): string {
  // Collapse 3+ same chars down to 2, then known double→single
  let s = input.replace(/(.)\1{2,}/g, '$1$1');
  // Known double-char patterns that should be single in VN
  const doubles: [RegExp, string][] = [
    [/nn/g, 'n'],
    [/pp(?!h)/g, 'p'],
    [/tt/g, 't'],
    [/mm/g, 'm'],
    [/cc/g, 'c'],
    [/ll/g, 'l'],
    [/kk/g, 'k'],
    [/gg/g, 'g'],
    [/bb/g, 'b'],
    [/dd(?!a|e|i|o|u)/g, 'd'],  // keep dd in "ddd"→"d" but not "dd" before vowel
    [/uu/g, 'u'],
    [/ii/g, 'i'],
    [/hh/g, 'h'],
  ];
  for (const [pat, rep] of doubles) {
    s = s.replace(pat, rep);
  }
  return s;
}

/**
 * Generate all query variant forms for matching:
 * 1. Normalised (lowercase + collapsed spaces)
 * 2. No-accent
 * 3. Repeat-collapsed of (2)
 * 4. All three with trailing 's' stripped (optional)
 */
export function generateQueryVariants(input: string): string[] {
  const norm = normalizeText(input);
  const noAccent = removeVietnameseAccents(norm);
  const collapsed = collapseRepeatedChars(noAccent);

  const variants = Array.from(new Set([norm, noAccent, collapsed]));
  return variants;
}

/**
 * Simple token-level Jaccard similarity (bag of tokens).
 */
export function tokenJaccard(a: string, b: string): number {
  const ta = new Set(a.split(/\s+/).filter(Boolean));
  const tb = new Set(b.split(/\s+/).filter(Boolean));
  if (ta.size === 0 && tb.size === 0) return 1;
  const intersection = [...ta].filter(x => tb.has(x)).length;
  const union = new Set([...ta, ...tb]).size;
  return intersection / union;
}

/**
 * Character-level edit distance (Levenshtein), capped at maxDist.
 * Returns Infinity if too far to bother.
 */
export function editDistance(a: string, b: string, maxDist = 4): number {
  if (Math.abs(a.length - b.length) > maxDist) return Infinity;
  const m = a.length, n = b.length;
  const dp: number[] = Array.from({ length: n + 1 }, (_, i) => i);
  for (let i = 1; i <= m; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const tmp = dp[j];
      dp[j] = a[i - 1] === b[j - 1]
        ? prev
        : 1 + Math.min(prev, dp[j], dp[j - 1]);
      prev = tmp;
    }
    if (Math.min(...dp) > maxDist) return Infinity;
  }
  return dp[n];
}

/**
 * Compute a fuzzy similarity score [0..1] between a query fragment
 * and a candidate string, using combined token Jaccard + edit distance.
 */
export function fuzzyScore(query: string, candidate: string): number {
  if (query === candidate) return 1.0;
  const jac = tokenJaccard(query, candidate);
  // Edit distance similarity (short-circuit for long strings)
  const maxLen = Math.max(query.length, candidate.length);
  if (maxLen === 0) return 1;
  const ed = editDistance(query, candidate, Math.ceil(maxLen * 0.4));
  const edSim = ed === Infinity ? 0 : 1 - ed / maxLen;
  return Math.max(jac, edSim);
}
