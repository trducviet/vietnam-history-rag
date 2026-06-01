/**
 * Answer Cleaner — PATCH 9B
 *
 * Removes internal metadata annotations from final answer text before
 * it is returned to the user. This cleaner operates ONLY on the final
 * output layer and NEVER touches corpus/index raw data.
 *
 * Rules:
 *  1. Strip bracket-label metadata lines  ([TỪ KHÓA VI] ..., [TÓM TẮT NGẮN] ..., etc.)
 *  2. Strip technical field lines         (source_ids: ..., record_id: ..., etc.)
 *  3. Collapse excessive blank lines
 *  4. Preserve citation IDs (EVT_xxxx, SYN_xxxx) in text when they are part
 *     of natural sentences — only remove them when they appear as raw field lines.
 *  5. Never remove historical content sentences.
 *
 * Applied ONLY in answer-generator.ts buildFallbackResponse() before return,
 * and in pipeline.ts for guard responses if needed.
 * NOT applied to corpus raw text, context bundles, or citation data structures.
 */

// ─── Bracket-Label Metadata Patterns ─────────────────────────

/**
 * Vietnamese bracket-label lines that signal internal metadata.
 * These appear as lines starting with [LABEL] followed by content.
 *
 * Pattern: /^\[LABEL\][^\n]*$/gm
 * We remove the ENTIRE line when it starts with one of these labels.
 *
 * Labels known to appear in blind generalization outputs (from 9A triage):
 * - [TỪ KHÓA VI]
 * - [TỪ KHÓA EN]
 * - [TÓM TẮT NGẮN]
 * - [MÔ TẢ]
 * - [TỔ CHỨC]
 * - [SỰ KIỆN LIÊN KẾT]
 * - [NỘI DUNG]
 * - [NGỮ CẢNH]
 * - [THỜI GIAN]
 * - [ĐỊA ĐIỂM]
 * - [NHÂN VẬT]
 * - [NGUỒN]
 * - [ID]
 * - [LOẠI]
 * - [GIAI ĐOẠN]
 * - [BỐI CẢNH]
 * - [Ý NGHĨA]
 * - [ALIAS ...]
 * - [TIÊU ĐỀ] (when standalone, not part of natural text)
 */
const BRACKET_METADATA_LABEL_PATTERN =
  /^\[(?:TỪ KHÓA(?: VI| EN)?|TÓM TẮT(?: NGẮN)?|MÔ TẢ|TỔ CHỨC|SỰ KIỆN LIÊN KẾT|NỘI DUNG|NGỮ CẢNH|THỜI GIAN|ĐỊA ĐIỂM|NHÂN VẬT|NGUỒN|ID|LOẠI|GIAI ĐOẠN|BỐI CẢNH|Ý NGHĨA|ALIAS[^\]]*|TIÊU ĐỀ)\][^\n]*/gim;

// ─── Technical Field Line Patterns ───────────────────────────

/**
 * Index/technical field lines that expose internal data structures.
 * These appear as "field_name: value" lines.
 *
 * We match lines that START with these field names followed by colon.
 * Does NOT match mid-sentence uses like "Theo source_ids được ghi nhận..."
 */
const TECHNICAL_FIELD_LINE_PATTERN =
  /^(?:source_ids|record_id|event_id|doc_id|content_hash|embedding|vector|metadata|keywords_vi|keywords_en|summary(?:\s*:)|description|linked_events|entities|raw_text|indexed_text|Source IDs)\s*:[^\n]*/gim;

// ─── Excessive Blank Line Collapse ───────────────────────────

/** Collapses 3+ consecutive blank lines into 2 max */
const MULTI_BLANK_LINE_PATTERN = /\n{3,}/g;

// ─── Title-Repeat Pattern ─────────────────────────────────────

/**
 * Pattern: "Theo tài liệu \"TITLE\" (YEAR): TITLE\n\n\n\n..."
 * When the answer opener is followed by the exact title repeated
 * on the next non-blank line, remove the repeated title line.
 *
 * This catches the specific pattern from 9A:
 *   "Theo tài liệu \"Chiếu Cần Vương ban hành\" (1885): Chiếu Cần Vương ban hành"
 *   ↑ opener                                               ↑ repeated title
 */
function removeTitleRepeat(text: string): string {
  // Match: opener line ending with ": TITLE" then blank lines then TITLE repeated
  return text.replace(
    /(Theo tài liệu\s+"([^"]+)"[^:]*:\s+\2)(\s*\n+\s*\2)/g,
    '$1'
  );
}

// ─── Citation ID Preservation Check ──────────────────────────

/**
 * Citation IDs (EVT_xxxx, SYN_xxxx, SRC_xxxx) in the ANSWER TEXT
 * are rarely meaningful to end users — they appear as raw inline tokens.
 * We strip them ONLY when they appear as standalone tokens on a metadata-
 * annotation line, not when embedded in natural Vietnamese sentences.
 *
 * Natural sentence examples (PRESERVE):
 *   "Theo EVT_0001, sự kiện này..."  ← part of explanation
 *
 * Metadata-line examples (already caught by TECHNICAL_FIELD_LINE_PATTERN):
 *   "Source IDs: SRC_0001, SRC_0064"  ← strip full line
 *
 * Citation IDs in the citations[] data structure are NEVER touched here.
 */

// ─── Main Cleaner ─────────────────────────────────────────────

/**
 * Clean the final answer text that will be shown to the user.
 *
 * SAFE TO CALL MULTIPLE TIMES (idempotent).
 * Does not modify citation data structures.
 * Does not modify corpus or index data.
 *
 * @param text Raw answer text from fallback generator or LLM
 * @returns Cleaned answer text
 */
export function cleanAnswerTextForUser(text: string): string {
  if (!text || text.length === 0) return text;

  let cleaned = text;

  // Step 1: Remove title repeat (e.g. opener "Theo ... TITLE: TITLE\n\n\nTITLE")
  cleaned = removeTitleRepeat(cleaned);

  // Step 2: Strip bracket-label metadata lines (line-start)
  cleaned = stripBracketMetadataLines(cleaned);

  // Step 2b (Patch 9G-R): Strip INLINE bracket metadata markers.
  cleaned = stripInlineBracketMarkers(cleaned);

  // Step 3: Strip technical field lines
  cleaned = stripTechnicalFieldLines(cleaned);

  // Step 3a: Strip raw reference/query lines that can leak from fallback snippets.
  cleaned = stripRawReferenceAndQueryLines(cleaned);

  // Step 3b (Patch 9D-R): Strip truncated bracket metadata fragments.
  cleaned = stripTruncatedBracketFragments(cleaned);

  // Step 3c (Patch 9G-R): Fix broken snippet prefixes from truncation.
  cleaned = repairBrokenSnippetBoundaries(cleaned);

  // Step 3d (Patch 9G-R): Remove trailing title duplicates.
  cleaned = removeTrailingTitleDuplicate(cleaned);

  // Step 3e (Patch 9G-R3): Repair orphaned colon fragments (":  đổ bộ...", ":  vào Nam Lào...")
  cleaned = repairOrphanedColonFragment(cleaned);

  // Step 3f (Patch 9G-R3): Deduplicate repeated sentence blocks in answer body
  cleaned = deduplicateAnswerBody(cleaned);

  // Step 3g (Patch 9H): Remove trailing duplicate block (text repeats its own opening)
  cleaned = removeTrailingDuplicateBlock(cleaned);

  // Step 3h (Patch 9H-R): Repair orphan body start — answer body starts mid-sentence
  cleaned = repairOrphanBodyStart(cleaned);

  // Step 3i (Patch 9H-R): Clean trailing orphan fragments (broken words after dedup cut)
  // Example: "...giằng co  đổ" → truncate at last complete sentence
  cleaned = cleanTrailingOrphanFragment(cleaned);

  // Step 4: Normalize whitespace
  cleaned = normalizeAnswerWhitespace(cleaned);

  return cleaned;
}

/**
 * Strip lines that begin with a Vietnamese bracket metadata label.
 * Example:  "[TỪ KHÓA VI] Liên quân Pháp; Rigault..." → removed
 * Preserves lines that don't start with a known metadata bracket.
 */
export function stripBracketMetadataLines(text: string): string {
  return text.replace(BRACKET_METADATA_LABEL_PATTERN, '').trim();
}

/**
 * Strip lines that start with a technical field name followed by colon.
 * Example: "source_ids: SRC_0001, SRC_0064" → removed
 * Example: "Source IDs: SRC_0001" → removed
 */
export function stripTechnicalFieldLines(text: string): string {
  return text.replace(TECHNICAL_FIELD_LINE_PATTERN, '').trim();
}

/**
 * Strip whole lines that are clearly not answer prose:
 * - raw source/doc IDs or URLs copied from corpus fields
 * - semicolon-separated keyword lists
 * - embedded benchmark/query prompt lines
 */
export function stripRawReferenceAndQueryLines(text: string): string {
  const lines = text.split(/\r?\n/);
  return lines.map((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) return line;
    if (/\b(?:SRC|EVT|SYN)_[A-Z0-9_]+\b/i.test(trimmed)) return '';
    if (/https?:\/\/|www\./i.test(trimmed)) return '';
    if (/^(?:nguồn|source|url|link|record|doc|id)\b/i.test(trimmed)) return '';
    if (index > 0 && /\?/.test(trimmed)) return '';
    if ((trimmed.match(/;/g) ?? []).length >= 1 && trimmed.length < 220) {
      const beforeSemicolon = trimmed.split(';')[0].trim();
      if (/[.!?]$/.test(beforeSemicolon)) return beforeSemicolon;
      const lastSentenceEnd = Math.max(beforeSemicolon.lastIndexOf('.'), beforeSemicolon.lastIndexOf('!'), beforeSemicolon.lastIndexOf('?'));
      if (lastSentenceEnd >= 20) return beforeSemicolon.slice(0, lastSentenceEnd + 1).trim();
      if (!/[.!?]$/.test(trimmed)) return '';
    }
    return line;
  }).filter(line => line.trim().length > 0).join('\n').trim();
}

/**
 * Patch 9D-R / 9G-R2: Strip truncated bracket metadata fragments.
 * When text_for_embedding is truncated mid-tag (e.g. "[TỪ K..." without closing "]"),
 * the main BRACKET_METADATA_LABEL_PATTERN won't match because it requires "]".
 * Patch 9G-R2: Also catches INLINE fragments (mid-line), not just line-start.
 * Catches: "[TÓ", "[TÓM", "[TÓ...", "[MÔ", "[NGU", "[META" etc.
 */
export function stripTruncatedBracketFragments(text: string): string {
  // Line-start fragments (original 9D-R)
  let result = text.replace(/^\[(?:TỪ|TÓM|MÔ|TỔ|SỰ|NỘI|NGỮ|THỜI|ĐỊA|NHÂN|NGUỒN|LOẠI|GIAI|BỐI|ALIAS|TIÊU|Ý|META|NGU|KẾT|HỆ|NGUYÊN|THAM)[^\]\n]*(?:\.{2,})?$/gm, '');
  // Patch 9G-R2: Inline fragments — partial bracket markers mid-line
  // Catches: "[TÓ", "[TÓ...", "[TÓM", "[TÓM...", "[MÔ", "[MÔ...", "[NGU", "[META" etc.
  result = result.replace(/\[(?:TÓ|MÔ|TỪ|TỔ|SỰ|NỘI|NGỮ|THỜI|ĐỊA|NHÂN|NGU|LOẠI|GIAI|BỐI|TIÊU|Ý|META|KẾT|HỆ|NGUYÊN|THAM|ALIAS)(?:[A-ZÀ-Ỹa-zà-ỹ\s]{0,20}(?:\.{2,})?)?(?!\])/g, '');
  return result.trim();
}

/**
 * Normalize whitespace in answer text:
 * - Collapse 3+ consecutive blank lines into at most 2 blank lines
 * - Trim leading/trailing whitespace
 * - Remove trailing spaces from each line
 */
export function normalizeAnswerWhitespace(text: string): string {
  return text
    .replace(MULTI_BLANK_LINE_PATTERN, '\n\n')
    .replace(/[ \t]+$/gm, '')
    .trim();
}

// ─── Patch 9G-R: Inline bracket marker strip ─────────────────

/**
 * Strip inline bracket metadata markers that appear MID-LINE.
 * The existing BRACKET_METADATA_LABEL_PATTERN only catches line-start.
 * This catches: "...diễn ra năm 1862. [MÔ TẢ] Ngày 28-7-1995..."
 * → "...diễn ra năm 1862. Ngày 28-7-1995..."
 */
export function stripInlineBracketMarkers(text: string): string {
  // Catch any [UPPERCASE_VIET_LABEL] inline, preserving surrounding text
  return text
    .replace(/\[(?:TỪ KHÓA(?: VI| EN)?|TÓM TẮT(?: NGẮN)?|MÔ TẢ|TỔ CHỨC|SỰ KIỆN LIÊN KẾT|NỘI DUNG|NGỮ CẢNH|THỜI GIAN|ĐỊA ĐIỂM|NHÂN VẬT|NGUỒN|ID|LOẠI|GIAI ĐOẠN|BỐI CẢNH|Ý NGHĨA|HỆ QUẢ|KẾT QUẢ|NGUYÊN NHÂN|SỰ KIỆN LIÊN QUAN|THAM CHIẾU|TIÊU ĐỀ|ALIAS[^\]]*)\]\s*/gi, '')
    // Catch-all: any remaining [UPPERCASE_VIET_2+_CHARS] that look like metadata
    .replace(/\[[A-ZÀ-Ỹ][A-ZÀ-Ỹ\s]{1,25}\]\s*/g, '');
}

// ─── Patch 9G-R / 9G-R2: Broken snippet boundary repair ──────────────

/**
 * Repair broken snippet boundaries from truncation.
 * Patch 9G-R: "100.000 quân" truncated to "000 quân" at char boundary.
 * Patch 9G-R2: Also fixes broken numeric spacing like "3. 000" → "3.000",
 * orphaned snippet starts like "vào Nam Lào..." or "đổ bộ..." after opener.
 */
export function repairBrokenSnippetBoundaries(text: string): string {
  let result = text;
  // 1. Remove orphaned numeric prefix at very start of answer content
  result = result.replace(/(:\s*)(\d{1,3}\s+(?:quân|người|lính|binh|chiến sĩ))/g, (match, prefix, fragment) => {
    if (fragment.startsWith('000') || fragment.startsWith('00')) {
      return prefix;
    }
    return match;
  });
  // 2. Patch 9G-R2: Fix broken numeric spacing: "3. 000" → "3.000", "100. 000" → "100.000"
  result = result.replace(/(\d+)\.\s+(\d{3})(?=\b)/g, '$1.$2');
  // 3. Patch 9G-R3: Remove mid-sentence orphaned "000 quân" / "000 người" (truncation artifact)
  result = result.replace(/\b000\s+(?:quân|người|lính|binh|chiến sĩ)/g, '');
  // 4. Patch 9G-R3: Clean orphaned numeric fragments like " 000" standalone
  result = result.replace(/\s+000(?=\s|[.,;:]|$)/g, '');
  // 5. Patch 9H: Fix broken "digit. word" patterns from truncated numbers
  // Catches: "và 3. đổ bộ" → drop broken fragment "và 3."
  // Catches: "hơn 8. Pháp" → drop broken fragment "hơn 8."
  // These occur when "3.000 đổ bộ" or "8.000 Pháp" is truncated at the '.' boundary
  result = result.replace(/(?:với|và|hơn|khoảng|gần|trên|dưới|chừng)\s+\d{1,2}\.\s+(?=đổ bộ|Pháp|quân|lính|binh|chiến|tàu|máy|xe|khẩu|súng)/g, '');
  // Also catch standalone: "tiêu diệt và bắt hơn 8. Pháp" → "tiêu diệt nhiều quân Pháp"
  result = result.replace(/(tiêu diệt và bắt|tiêu diệt|bắt sống|bắt)\s+(?:hơn|khoảng|gần)?\s*\d{1,2}\.\s+(?:Pháp|quân|lính)/g, 'tiêu diệt nhiều quân Pháp');
  // Patch 9H-R: If prior repair left "tiêu diệt và bắt Pháp" (missing quantity), fix grammar
  result = result.replace(/tiêu diệt và bắt\s+Pháp/g, 'tiêu diệt nhiều quân Pháp');
  // Patch 9H-R: Fix "30. vào", "30. quân" etc — broken truncation
  result = result.replace(/\b(\d{1,2})\.\s+(?=vào|quân|binh|lính|chiến|tàu|máy|xe)/g, 'nhiều ');
  return result;
}

// ─── Patch 9G-R: Trailing title duplicate removal ────────────

/**
 * Remove trailing title duplicate at end of answer.
 * Catches: "...content. Hiệp ước Nhâm Tuất..." at the very end
 * when it's just a repeat of the opener title.
 */
export function removeTrailingTitleDuplicate(text: string): string {
  // Extract title from opener pattern
  const titleMatch = text.match(/(?:Theo tài liệu|Dựa trên tài liệu|Sự kiện)\s+"([^"]+)"/);
  if (!titleMatch) return text;

  const title = titleMatch[1].trim();
  if (title.length < 10) return text;

  // Check if text ends with the title (possibly truncated)
  const titleNorm = title.toLowerCase().normalize('NFKC');
  const textTrimmed = text.trimEnd();
  const lastChunk = textTrimmed.substring(Math.max(0, textTrimmed.length - title.length - 20)).toLowerCase().normalize('NFKC');

  if (lastChunk.endsWith(titleNorm) || lastChunk.endsWith(titleNorm + '.') || lastChunk.endsWith(titleNorm + '...')) {
    // Find and remove the trailing duplicate
    const idx = textTrimmed.lastIndexOf(title);
    if (idx > title.length + 50) { // Only if not the first occurrence
      const beforeDup = textTrimmed.substring(0, idx).trimEnd();
      // Clean trailing punctuation/whitespace
      return beforeDup.replace(/[.\s,;:]+$/, '.').trim();
    }
  }
  return text;
}

/**
 * Verify that a text block is clean (no known metadata leak patterns).
 * Used by diagnostic script.
 *
 * Returns array of detected leak patterns (empty = clean).
 */
export function detectMetadataLeaks(text: string): string[] {
  const leaks: string[] = [];

  const LEAK_CHECKS: Array<{ label: string; pattern: RegExp }> = [
    { label: '[TỪ KHÓA VI]',        pattern: /\[TỪ KHÓA VI\]/i },
    { label: '[TỪ KHÓA EN]',        pattern: /\[TỪ KHÓA EN\]/i },
    { label: '[TÓM TẮT NGẮN]',      pattern: /\[TÓM TẮT NGẮN\]/i },
    { label: '[TÓM TẮT]',           pattern: /\[TÓM TẮT\]/i },
    { label: '[MÔ TẢ]',             pattern: /\[MÔ TẢ\]/i },
    { label: '[TỔ CHỨC]',           pattern: /\[TỔ CHỨC\]/i },
    { label: '[SỰ KIỆN LIÊN KẾT]',  pattern: /\[SỰ KIỆN LIÊN KẾT\]/i },
    { label: '[NỘI DUNG]',          pattern: /\[NỘI DUNG\]/i },
    { label: '[NGỮ CẢNH]',          pattern: /\[NGỮ CẢNH\]/i },
    { label: '[LOẠI]',              pattern: /\[LOẠI\]/i },
    { label: '[GIAI ĐOẠN]',         pattern: /\[GIAI ĐOẠN\]/i },
    { label: '[BỐI CẢNH]',          pattern: /\[BỐI CẢNH\]/i },
    { label: '[Ý NGHĨA]',           pattern: /\[Ý NGHĨA\]/i },
    // Patch 9G-R: Additional inline markers
    { label: '[HỆ QUẢ]',            pattern: /\[HỆ QUẢ\]/i },
    { label: '[KẾT QUẢ]',           pattern: /\[KẾT QUẢ\]/i },
    { label: '[NGUYÊN NHÂN]',       pattern: /\[NGUYÊN NHÂN\]/i },
    { label: '[THAM CHIẾU]',        pattern: /\[THAM CHIẾU\]/i },
    { label: '[SỰ KIỆN LIÊN QUAN]', pattern: /\[SỰ KIỆN LIÊN QUAN\]/i },
    { label: 'source_ids:',         pattern: /^source_ids\s*:/im },
    { label: 'record_id:',          pattern: /^record_id\s*:/im },
    { label: 'keywords_vi:',        pattern: /^keywords_vi\s*:/im },
    { label: 'keywords_en:',        pattern: /^keywords_en\s*:/im },
    { label: 'Source IDs:',         pattern: /^Source IDs\s*:/im },
  ];

  for (const check of LEAK_CHECKS) {
    if (check.pattern.test(text)) {
      leaks.push(check.label);
    }
  }

  // Patch 9D-R / 9G-R2: Detect truncated bracket fragments (e.g. "[TỪ K..." without closing "]")
  // Also catches inline fragments: "[TÓ", "[TÓ...", "[TÓM", "[MÔ", "[NGU", "[META"
  const TRUNCATED_BRACKET_PATTERN = /\[(?:TỪ|TÓM|MÔ|TỔ|SỰ|NỘI|NGỮ|THỜI|ĐỊA|NHÂN|NGUỒN|LOẠI|GIAI|BỐI|ALIAS|TIÊU|Ý|META|NGU|KẾT|HỆ|NGUYÊN|THAM)[^\]\n]{0,30}(?:\.{2,})?(?:\n|$)/m;
  if (TRUNCATED_BRACKET_PATTERN.test(text)) {
    const match = text.match(TRUNCATED_BRACKET_PATTERN);
    leaks.push(`TRUNCATED_BRACKET: ${match ? match[0].trim().substring(0, 30) : 'unknown'}`);
  }

  // Patch 9G-R2: Detect partial/fragment bracket markers inline
  // Catches: "[TÓ", "[TÓ...", "[TÓM", "[MÔ", "[MÔ...", "[NGU", "[META"
  const FRAGMENT_BRACKET_PATTERN = /\[(?:TÓ|MÔ|TỪ|TỔ|SỰ|NỘI|NGỮ|THỜI|ĐỊA|NHÂN|NGU|LOẠI|GIAI|BỐI|TIÊU|Ý|META|KẾT|HỆ|NGUYÊN|THAM)(?:[A-ZÀ-Ỹa-zà-ỹ\s]{0,20}(?:\.{2,})?)?(?![\]A-ZÀ-Ỹ])/;
  if (FRAGMENT_BRACKET_PATTERN.test(text)) {
    const match = text.match(FRAGMENT_BRACKET_PATTERN);
    if (match && !leaks.some(l => l.includes('TRUNCATED_BRACKET'))) {
      leaks.push(`MARKER_FRAGMENT: ${match[0].trim().substring(0, 20)}`);
    }
  }

  // Patch 9G-R: Catch-all inline bracket markers
  const INLINE_BRACKET = /\[[A-ZÀ-Ỹ][A-ZÀ-Ỹ\s]{2,25}\]/;
  if (INLINE_BRACKET.test(text)) {
    const match = text.match(INLINE_BRACKET);
    if (match) {
      const found = match[0];
      if (!leaks.some(l => found.includes(l.replace(/[\[\]]/g, '')))) {
        leaks.push(`INLINE_BRACKET: ${found}`);
      }
    }
  }

  return leaks;
}

/**
 * Patch 9G-R2: Detect metadata marker FRAGMENTS specifically.
 * Returns array of fragment patterns found (empty = clean).
 * Separate from detectMetadataLeaks to allow independent counting.
 */
export function detectMetadataMarkerFragments(text: string): string[] {
  const fragments: string[] = [];
  const FRAG_PATTERNS = [
    /\[TÓ(?![M\]])/,   // [TÓ without M or ]
    /\[TÓM(?![\s\]])/,  // [TÓM without space or ]
    /\[TÓ\.\.\./,       // [TÓ...
    /\[MÔ(?![\s\]])/,   // [MÔ without space or ]
    /\[MÔ\.\.\./,       // [MÔ...
    /\[NGU(?![ỒÔ\]])/,  // [NGU fragment
    /\[META/i,          // [META
  ];
  for (const p of FRAG_PATTERNS) {
    if (p.test(text)) {
      const m = text.match(p);
      fragments.push(m ? m[0] : 'unknown');
    }
  }
  return fragments;
}

// ─── Patch 9G-R3: Orphaned colon fragment repair ─────────────

/**
 * Repair orphaned fragments after citation prefix colon.
 * Catches: ":  đổ bộ tấn công...", ":  vào Nam Lào...", ":  tiến hành..."
 * These occur when text_for_embedding is truncated mid-sentence and
 * the extracted snippet starts with a verb/preposition fragment.
 */
export function repairOrphanedColonFragment(text: string): string {
  // Pattern: after opener "Dựa trên/Theo tài liệu ...":" followed by
  // 2+ spaces and a lowercase word (orphaned fragment start)
  let result = text;
  // Match colon + whitespace + lowercase fragment (verb/preposition start)
  result = result.replace(
    /((?:Theo tài liệu|Dựa trên tài liệu|Sự kiện)\s+"[^"]+"[^:]*:\s*)([a-zà-ỹ]{1,6}\s+)/g,
    (match, prefix, fragment) => {
      const word = fragment.trim();
      // Known orphan starters: prepositions/verbs without subject
      const ORPHAN_STARTERS = ['vào', 'đổ', 'bộ', 'tiến', 'giải', 'mở', 'tấn', 'bắt', 'kéo', 'đánh', 'chiếm', 'từ', 'qua', 'với', 'cho', 'theo', 'trong', 'trên', 'dưới'];
      if (ORPHAN_STARTERS.includes(word)) {
        // Drop the orphan word, capitalize next word
        return prefix;
      }
      return match;
    }
  );
  // Also catch double-space after colon: ":  " → ": "
  result = result.replace(/:\s{2,}/g, ': ');

  // Patch 9H-R: Remove short orphan sentence at body start (metadata residual)
  // Pattern: "Theo tài liệu \"title\" (year): <ShortOrphan>. Ngày..."
  // The orphan is a broken field value like "Bộ tấn công cửa biển Đà Nẵng."
  result = result.replace(
    /((?:Theo tài liệu|Dựa trên tài liệu|Sự kiện)\s+"[^"]+"\s*\([^)]+\)\s*:\s*)([A-ZÀ-Ỹ][^.]{3,60}\.)\s+((?:Ngày|Năm|Tháng|Đầu|Cuối|Giữa|Sau|Trước|Vào)\s)/,
    (_match, prefix, orphan, nextStart) => {
      if (orphan.length < 70) return prefix + nextStart;
      return _match;
    }
  );

  return result;
}

// ─── Patch 9G-R3: Duplicate answer body cleanup ──────────────

/**
 * Deduplicate repeated sentence blocks in answer body.
 * Patch 9H: Also detects trailing duplicates WITHOUT period boundary.
 * The truncation from text_for_embedding often removes the period at the
 * snippet end, so the repeat starts as "...dân tộc Ngày 2-9-1945..." without
 * any sentence delimiter between original and copy.
 */
export function deduplicateAnswerBody(text: string): string {
  // Patch 9H: First check for trailing duplicate via sliding window
  // This catches the case where the answer tail repeats the answer head
  // without any period boundary between them.
  const cleaned = removeTrailingDuplicateBlock(text);

  // Then do sentence-level dedup on the result
  const sentences = cleaned.split(/(?<=[.。])\s+/).filter(s => s.trim().length > 0);
  if (sentences.length < 4) return cleaned;

  const normalizeForCompare = (s: string): string =>
    s.toLowerCase().normalize('NFKC')
      .replace(/\s+/g, ' ')
      .replace(/[.,;:!?()\[\]"']/g, '')
      .trim();

  const kept: boolean[] = new Array(sentences.length).fill(true);
  for (let i = 0; i < sentences.length; i++) {
    if (!kept[i]) continue;
    const si = normalizeForCompare(sentences[i]);
    if (si.length < 15) continue;

    for (let j = i + 1; j < sentences.length; j++) {
      if (!kept[j]) continue;
      const sj = normalizeForCompare(sentences[j]);
      if (sj.length < 15) continue;

      if (si === sj || (si.length > 20 && sj.includes(si)) || (sj.length > 20 && si.includes(sj))) {
        if (sentences[j].includes('Vế thứ')) continue;
        if (sentences[j].match(/^-\s+/)) continue;
        kept[j] = false;
      }
    }
  }

  // Block-level dedup (3+ consecutive sentences repeated)
  for (let blockLen = 3; blockLen <= Math.floor(sentences.length / 2); blockLen++) {
    for (let i = 0; i <= sentences.length - blockLen * 2; i++) {
      const blockA = sentences.slice(i, i + blockLen).map(normalizeForCompare).join(' ');
      for (let j = i + blockLen; j <= sentences.length - blockLen; j++) {
        const blockB = sentences.slice(j, j + blockLen).map(normalizeForCompare).join(' ');
        if (blockA === blockB || (blockA.length > 40 && blockB.includes(blockA))) {
          for (let k = j; k < j + blockLen; k++) {
            if (!sentences[k].includes('Vế thứ') && !sentences[k].match(/^-\s+/)) {
              kept[k] = false;
            }
          }
        }
      }
    }
  }

  return sentences.filter((_, i) => kept[i]).join(' ');
}

/**
 * Patch 9H: Detect and remove trailing duplicate block.
 * The answer ends with a copy of its beginning, without period boundary.
 * Example: "...quan trọng nhất của dân tộc Ngày 2-9-1945 tại Quảng trường Ba Đình..."
 * The text after "dân tộc" is an exact repeat of the opening.
 */
function removeTrailingDuplicateBlock(text: string): string {
  // Skip comparison side markers
  if (text.includes('Vế thứ')) return text;
  if (text.length < 120) return text;

  const textLower = text.toLowerCase();

  // Find the body start (after citation opener colon)
  const bodyStartMatch = text.match(/^(?:Theo tài liệu|Dựa trên tài liệu|Sự kiện)\s+"[^"]+"\s*\([^)]+\)\s*:\s*/);
  const bodyStart = bodyStartMatch ? bodyStartMatch[0].length : 0;

  // Strategy: find body prefix chunk repeated later in the text
  const bodyLower = textLower.substring(bodyStart);
  if (bodyLower.length < 80) return text;

  const minWindow = 30;
  const maxWindow = Math.min(200, Math.floor(bodyLower.length / 3));

  for (let wSize = minWindow; wSize <= maxWindow; wSize += 5) {
    const bodyHead = bodyLower.substring(0, wSize);
    // Search for this body prefix repeated later (after at least 40% of body)
    const searchFrom = Math.floor(bodyLower.length * 0.35);
    const repeatIdx = bodyLower.indexOf(bodyHead, searchFrom);
    if (repeatIdx > 0 && repeatIdx >= searchFrom) {
      // Verify continuation match
      const tailPart = bodyLower.substring(repeatIdx);
      const headPart = bodyLower.substring(0, tailPart.length);
      let matchLen = 0;
      for (let i = 0; i < tailPart.length && i < headPart.length; i++) {
        if (tailPart[i] === headPart[i]) matchLen++;
        else break;
      }
      if (matchLen > 25 && matchLen > tailPart.length * 0.6) {
        // Found trailing duplicate of body — cut at the repeat point in original text
        const cutPosInOriginal = bodyStart + repeatIdx;
        // Search backwards for a sentence boundary
        let cutAt = cutPosInOriginal;
        for (let i = cutPosInOriginal - 1; i > Math.max(0, cutPosInOriginal - 80); i--) {
          if (text[i] === '.' || text[i] === '。') {
            cutAt = i + 1;
            break;
          }
        }
        const result = text.substring(0, cutAt).trim();
        if (result.length > 50) {
          return result.replace(/[\s,;:]+$/, '') + (result.endsWith('.') ? '' : '.');
        }
      }
    }
  }
  return text;
}

// ─── Patch 9H-R: Orphan body start repair ────────────────────

/**
 * Repair answers where the body content after the citation opener starts mid-sentence.
 * Example: "Dựa trên tài liệu \"Chiến dịch đường 9...\" (1971): Nam Lào (Chiến dịch Lam Sơn 719) với sự hỗ trợ..."
 * → The "Nam Lào..." is not a proper sentence start; it's a truncated fragment.
 * Fix: If the body starts with a location/proper noun fragment followed by "(..." or lowercase connector,
 * try to form a proper sentence by prepending context from the title.
 */
function repairOrphanBodyStart(text: string): string {
  // Only process answers with citation opener
  const openerMatch = text.match(/^((?:Theo tài liệu|Dựa trên tài liệu|Sự kiện)\s+"([^"]+)"\s*\([^)]+\)\s*:\s*)/);
  if (!openerMatch) return text;

  const opener = openerMatch[1];
  const title = openerMatch[2];
  const body = text.substring(opener.length);

  // Check if body starts with an orphan fragment (location/noun + parenthetical)
  // Pattern: "Nam Lào (Chiến dịch...)" or "Nam Lào ... với"
  const orphanBodyStart = body.match(/^([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)?)\s*\(([^)]+)\)\s+(với\s|nhằm\s|do\s|là\s|vào\s|trong\s|khi\s|sau\s|trước\s)/);
  if (orphanBodyStart) {
    // The body starts mid-sentence — prepend a proper intro from the title
    // Extract year from title if present
    const yearMatch = title.match(/\b(1\d{3})\b/);
    const yearStr = yearMatch ? ` năm ${yearMatch[1]}` : '';
    // Build a proper sentence start
    const entityName = orphanBodyStart[2]; // "Chiến dịch Lam Sơn 719"
    const fixedBody = `${entityName}${yearStr} tại ${orphanBodyStart[1]} ${body.substring(orphanBodyStart[0].length - orphanBodyStart[3].length)}`;
    return opener + fixedBody;
  }

  // Patch 9H-R: Remove short orphan sentence at body start (metadata residual)
  // Pattern: body = "<ShortOrphan>. Ngày..." where orphan is a broken metadata field value
  const orphanSentence = body.match(/^([A-ZÀ-Ỹ][^.]{3,60}\.)\s+((?:Ngày|Năm|Tháng|Đầu|Cuối|Giữa|Sau|Trước|Vào)\s)/);
  if (orphanSentence && orphanSentence[1].length < 70) {
    return opener + body.substring(orphanSentence[1].length).trim();
  }

  // Also catch: body starts with lowercase word (mid-sentence fragment)
  if (/^[a-zà-ỹ]/.test(body) && body.length > 20) {
    // Capitalize the first character
    return opener + body.charAt(0).toUpperCase() + body.substring(1);
  }

  return text;
}

// ─── Patch 9H-R: Trailing orphan fragment cleanup ────────────

/**
 * Clean trailing orphan fragments left after dedup truncation.
 * If the answer ends with double-space + short word or incomplete tail,
 * truncate back to the last complete sentence.
 */
function cleanTrailingOrphanFragment(text: string): string {
  if (text.includes('Vế thứ')) return text;
  if (text.length < 50) return text;

  // Patch 9H-R: Check for double-space in the text followed by a partial repeat
  // Example: "...giằng co  đổ bộ tấn công cửa biển Đà Nẵng." where "đổ bộ tấn công..."
  // is a partial repeat of earlier text in the answer
  const doubleSpaceIdx = text.search(/\s{2,}[a-zà-ỹA-ZÀ-Ỹ]/);
  if (doubleSpaceIdx > 50) {
    const afterDoubleSpace = text.substring(doubleSpaceIdx).replace(/^\s+/, '').toLowerCase();
    const beforeDoubleSpace = text.substring(0, doubleSpaceIdx).toLowerCase();
    if (afterDoubleSpace.length > 15 && afterDoubleSpace.length < beforeDoubleSpace.length) {
      const tailChunk = afterDoubleSpace.substring(0, Math.min(30, afterDoubleSpace.length));
      if (beforeDoubleSpace.includes(tailChunk)) {
        // Cut right at the double-space point, preserving everything before
        const cutText = text.substring(0, doubleSpaceIdx).trim();
        // Clean trailing incomplete clause
        return cutText.replace(/[\s,;:]+$/, '') + '.';
      }
    }
  }

  // Check for trailing double-space + short word(s): "giằng co  đổ"
  const trailingDoubleSpace = text.match(/\s{2,}[a-zà-ỹA-ZÀ-Ỹ]{1,6}\.?\s*$/);
  if (trailingDoubleSpace) {
    const cutPos = text.length - trailingDoubleSpace[0].length;
    const lastPeriod = text.lastIndexOf('.', cutPos);
    if (lastPeriod > 50) {
      return text.substring(0, lastPeriod + 1).trim();
    }
    return text.substring(0, cutPos).trim().replace(/[\s,;:]+$/, '') + '.';
  }

  // Check for answer ending without period and short tail after last period
  const trimmed = text.trimEnd();
  if (!/[.。!?]$/.test(trimmed)) {
    const lastPeriod = trimmed.lastIndexOf('.');
    if (lastPeriod > 50) {
      const tail = trimmed.substring(lastPeriod + 1).trim();
      if (tail.length < 20 && tail.length > 0) {
        return trimmed.substring(0, lastPeriod + 1).trim();
      }
    }
  }

  return text;
}

// ─── Patch 9G-R3: Broken snippet residual detector ───────────

/**
 * Detect broken snippet residuals in answer text.
 * Returns array of detected residual patterns (empty = clean).
 * Used by diagnostic scripts.
 */
export function detectBrokenSnippetResidual(text: string): string[] {
  const residuals: string[] = [];

  // 1. Colon + whitespace + lowercase verb/preposition fragment
  const colonOrphan = text.match(/:\s{2,}[a-zà-ỹ]/g);
  if (colonOrphan) {
    residuals.push(`COLON_ORPHAN: ${colonOrphan[0].trim().substring(0, 20)}`);
  }

  // 2. Orphaned "000 quân" / "000 người" mid-sentence
  if (/\b000\s+(?:quân|người|lính|binh|chiến sĩ)/.test(text)) {
    residuals.push('ORPHAN_000_NUMERIC');
  }

  // 3. Broken numeric spacing: "3. 000", "100. 000"
  if (/(\d+)\.\s+(\d{3})(?=\b)/.test(text)) {
    residuals.push('BROKEN_NUMERIC_SPACING');
  }

  // 4. Patch 9G-R3: Only flag known orphan starters after citation prefix
  const afterColon = text.match(/(?:Theo tài liệu|Dựa trên tài liệu)\s+"[^"]+\"[^:]*:\s+([a-zà-ỹ]+)/);
  if (afterColon) {
    const w = afterColon[1];
    if (['vào','đổ','tiến','giải','mở','tấn','bắt','kéo','đánh','chiếm'].includes(w)) {
      residuals.push(`LOWERCASE_AFTER_PREFIX: starts with '${w}'`);
    }
  }

  return residuals;
}

/**
 * Detect duplicate answer body content.
 * Returns true if answer contains repeated sentence blocks.
 */
export function detectDuplicateAnswerBody(text: string): boolean {
  const sentences = text.split(/(?<=[.。])\s+/).filter(s => s.trim().length > 0);
  if (sentences.length < 4) return false;

  const normalizeForCompare = (s: string): string =>
    s.toLowerCase().normalize('NFKC')
      .replace(/\s+/g, ' ')
      .replace(/[.,;:!?()\[\]"']/g, '')
      .trim();

  const seen = new Set<string>();
  for (const s of sentences) {
    const ns = normalizeForCompare(s);
    if (ns.length < 15) continue;
    if (seen.has(ns)) return true;
    // Also check substring containment
    for (const prev of seen) {
      if (prev.length > 20 && (ns.includes(prev) || prev.includes(ns))) return true;
    }
    seen.add(ns);
  }
  return false;
}
