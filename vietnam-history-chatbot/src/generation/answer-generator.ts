/**
 * Answer Generator — uses gpt-5.4 to generate grounded historical answers
 * from curated context bundles. Never receives raw top-k results.
 *
 * Prompt strategy:
 *  1. Trả lời trực tiếp trước
 *  2. Giải thích sau
 *  3. Nêu rõ ngày tháng khi có
 *  4. Phân biệt planned_not_executed vs actual events
 *  5. Hiển thị citations (record IDs + titles)
 *  6. Confidence level
 *  7. Nói rõ uncertainty khi bằng chứng yếu
 *
 * PATCH 3: Citation enrichment with provenance, updated prompt with
 *          citation plan and disambiguation instructions.
 * PATCH 7G: Answer verifier integration for intent-aware repair.
 */

import OpenAI from 'openai';
import { config } from '../shared/config.js';
import { buildChatCompletionParams, shouldCallLLM } from '../llm/openai-chat-compat.js';
import { computeConfidence, clampConfidenceByAnswerQuality } from './confidence-scorer.js';
import { enrichCitations, buildFallbackCitationsFromContext } from './citation-enricher.js';
import { verifyAndRepairAnswer, validateSideRelevance } from './answer-verifier.js';
import { detectFocusProfile, extractQueryFocus, scoreDocumentFocus } from '../evidence/focus-precision.js';
// PATCH 9B: Output cleanliness — strip metadata leaks from final answers
import { cleanAnswerTextForUser } from '../utils/answer-cleaner.js';
// Patch 9C-R Final: side-aware doc selection for comparison/disambiguation
import { expandComparisonSideTerms, normalizeVietnamesePhrase } from '../routing/query-frame-builder.js';
import type {
  ContextBundle,
  HybridSearchResult,
  LoadedDataset,
  ChatResponse,
  Citation,
  ConfidenceLevel,
  ConfidenceSignals,
  QueryFrame,
} from '../shared/types.js';

// ─── Generation Prompt ───────────────────────────────────────

const GENERATION_SYSTEM_PROMPT = `Bạn là một chuyên gia lịch sử Việt Nam (1858–2000), trả lời câu hỏi dựa HOÀN TOÀN trên tài liệu được cung cấp.

## QUY TẮC BẮT BUỘC:

1. **Trả lời trực tiếp trước**, rồi mới giải thích chi tiết sau.
2. **Nêu rõ ngày tháng, năm** khi tài liệu cung cấp.
3. **Phân biệt rõ ràng** giữa sự kiện ĐÃ XẢY RA (actual) và sự kiện KẾ HOẠCH CHƯA THỰC HIỆN (planned_not_executed). Nếu tài liệu đánh dấu "planned_not_executed", bạn PHẢI nói rõ.
4. **Chỉ dùng thông tin từ tài liệu** được cung cấp. Không bịa đặt.
5. Nếu tài liệu không đủ để trả lời, hãy **nói rõ sự không chắc chắn**.
6. Nếu có nhiều nguồn mâu thuẫn, hãy **nêu rõ sự khác biệt**.
7. **Ưu tiên citation từ PRIMARY evidence**. SUPPORTING evidence chỉ dùng để giải thích/bối cảnh.
8. **Không dùng DISAMBIGUATION NOTES làm bằng chứng chính**. Chúng chỉ giúp tránh nhầm lẫn.
9. Nếu có CITATION PLAN, tuân theo citation_plan khi ghi citations.
10. Nếu câu hỏi là fact/date/actor/location, câu trả lời phải ngắn, trực tiếp, rồi mới giải thích.

## FORMAT TRẢ LỜI:

Trả về JSON object với các trường:
- "answer": câu trả lời trực tiếp, ngắn gọn (1-3 câu)
- "explanation": giải thích chi tiết hơn với bối cảnh lịch sử (2-5 câu)
- "citations": array các object {"record_id", "title", "relevance"} — tài liệu đã sử dụng
- "related_events": array các object {"record_id", "title"} — sự kiện liên quan đáng chú ý

Trả về ĐÚNG JSON object, không kèm text khác.`;

// ─── Answer Generation ───────────────────────────────────────

/**
 * Generate a grounded answer from a curated context bundle.
 * Uses gpt-5.4 for high-quality Vietnamese historical responses.
 *
 * Patch 7G: accepts optional queryFrame for verifier integration.
 */
export async function generateAnswer(
  query: string,
  contextBundle: ContextBundle,
  retrievalResults: HybridSearchResult[],
  dataset: LoadedDataset,
  queryFrame?: QueryFrame
): Promise<ChatResponse> {
  // Compute confidence BEFORE generation
  const { level: confidence, signals: confidenceDetails } = computeConfidence(
    contextBundle,
    retrievalResults,
    query,
    dataset
  );

  try {
    const llmResponse = await generateWithLLM(query, contextBundle, confidence);

    // Enrich citations with provenance from sources.jsonl
    const enrichedCitations = enrichCitations(
      llmResponse.citations,
      contextBundle,
      dataset
    );

    // Patch 9F: Clamp confidence based on answer quality
    const clampedConfidence = clampConfidenceByAnswerQuality(
      confidence, llmResponse.answer, enrichedCitations.length
    );

    return {
      ...llmResponse,
      citations: enrichedCitations,
      confidence: clampedConfidence,
      confidence_details: confidenceDetails,
    };
  } catch (error) {
    console.warn('⚠️  LLM generation failed, using fallback:', (error as Error).message);
    return buildFallbackResponse(query, contextBundle, confidence, confidenceDetails, dataset, queryFrame);
  }
}

/** Generate answer via LLM */
async function generateWithLLM(
  query: string,
  contextBundle: ContextBundle,
  confidence: ConfidenceLevel
): Promise<Omit<ChatResponse, 'confidence' | 'confidence_details'>> {
  if (!shouldCallLLM(config.openaiApiKey)) {
    throw new Error('LLM disabled or no API key — falling back to template');
  }

  const openai = new OpenAI({ apiKey: config.openaiApiKey });

  // Build user message with context
  const confidenceNote = confidence === 'low'
    ? '\n\n⚠️ Mức độ tin cậy THẤP: bằng chứng yếu hoặc có risk nhầm lẫn. Hãy nêu rõ sự không chắc chắn.'
    : confidence === 'medium'
    ? '\n\nℹ️ Mức độ tin cậy TRUNG BÌNH: có thể có sự mơ hồ. Hãy cẩn thận khi kết luận.'
    : '';

  const userMessage = `Câu hỏi: ${query}\n\n${contextBundle.context_text}${confidenceNote}`;

  const params = buildChatCompletionParams({
    model: config.generationModel,
    messages: [
      { role: 'system', content: GENERATION_SYSTEM_PROMPT },
      { role: 'user', content: userMessage },
    ],
    maxTokens: 1000,
    temperature: 0.1,
    responseFormat: { type: 'json_object' },
    purpose: 'generation',
  });

  const response = await openai.chat.completions.create(params as any);

  const content = response.choices[0]?.message?.content;
  if (!content) throw new Error('Empty response from generation model');

  const parsed = JSON.parse(content) as Record<string, unknown>;

  // PATCH 9B: Clean LLM output in case prompt leaked metadata into answer
  const rawAnswer = typeof parsed.answer === 'string' ? parsed.answer : 'Không thể tạo câu trả lời.';
  const rawExplanation = typeof parsed.explanation === 'string' ? parsed.explanation : '';

  return {
    answer: cleanAnswerTextForUser(rawAnswer),
    explanation: cleanAnswerTextForUser(rawExplanation),
    citations: Array.isArray(parsed.citations)
      ? parsed.citations.map((c: Record<string, unknown>) => ({
          record_id: String(c.record_id ?? ''),
          title: String(c.title ?? ''),
          relevance: String(c.relevance ?? ''),
        }))
      : extractBasicCitationsFromBundle(contextBundle),
    related_events: Array.isArray(parsed.related_events)
      ? parsed.related_events.map((e: Record<string, unknown>) => ({
          record_id: String(e.record_id ?? ''),
          title: String(e.title ?? ''),
        }))
      : [],
  };
}

// ─── Fallback Response ───────────────────────────────────────

/**
 * Build a response from context bundle when LLM is unavailable.
 *
 * Patch 7G: Uses answer verifier for intent-aware repair.
 */
function buildFallbackResponse(
  query: string,
  bundle: ContextBundle,
  confidence: ConfidenceLevel,
  confidenceDetails: ConfidenceSignals,
  dataset: LoadedDataset,
  queryFrame?: QueryFrame
): ChatResponse {
  const primaryDoc = bundle.primary_docs[0];
  const secondaryDoc = bundle.primary_docs[1];

  // Build draft answer from primary doc content
  let draftAnswer: string;
  if (!primaryDoc) {
    draftAnswer = 'Không tìm thấy tài liệu phù hợp để trả lời câu hỏi này.';
  } else if (queryFrame?.comparison_sides && (() => {
    // Patch 8D: Allow two-sided format except for primary-focused disambiguation
    const isDisambig = ['disambiguation', 'misconception_check'].includes(queryFrame.intent);
    if (!isDisambig) return true; // Not disambiguation → use two-sided
    // Disambiguation with 'có phải' → use two-sided (identity check)
    const marker = queryFrame.comparison_sides?.marker ?? '';
    return ['có phải', 'có phải cùng', 'có giống nhau'].includes(marker);
  })()) {
    // Patch 7K: Two-sided comparison format
    // Patch 9C-R Final: Side-aware doc selection — find best dedicated doc for each side
    const sides = queryFrame.comparison_sides;
    const allDocs = [...bundle.primary_docs, ...bundle.supporting_docs];

    // Helper: find best doc for a given side, excluding person/timeline when possible
    // Patch 9E-S: Integrates validateSideRelevance to reject foreign false-positive docs
    const findBestSideDoc = (sideLabel: string): typeof primaryDoc | undefined => {
      const sideTerms = expandComparisonSideTerms(sideLabel);
      const multiWordTerms = sideTerms.filter(t => t.length > 5 && t.includes(' '));

      const matchesSide = (doc: typeof primaryDoc): boolean => {
        const text = normalizeVietnamesePhrase(`${doc.title} ${doc.summary}`);
        if (multiWordTerms.some(t => text.includes(t))) return true;
        const longTokens = sideTerms.filter(t => t.length > 3);
        return longTokens.filter(t => text.includes(t)).length >= 2;
      };

      const isPersonOrTimeline = (doc: typeof primaryDoc): boolean => {
        const titleNorm = doc.title.toLowerCase();
        return titleNorm.startsWith('hồ sơ nhân vật') || doc.doc_id.startsWith('SYN_PERSON_') || doc.doc_id.startsWith('SYN_TIMELINE_');
      };

      // Patch 9E-S: Filter candidates through foreign side validator
      const validCandidates = allDocs.filter(d => validateSideRelevance(sideLabel, d));

      // Priority 1: dedicated event/synthesis (not person/timeline)
      const dedicated = validCandidates.filter(d => matchesSide(d) && !isPersonOrTimeline(d));
      if (dedicated.length > 0) return dedicated[0];

      // Priority 2: any matching doc including person/timeline
      const anyMatch = validCandidates.filter(d => matchesSide(d));
      if (anyMatch.length > 0) return anyMatch[0];

      return undefined;
    };

    const sideADoc = findBestSideDoc(sides.side_a) ?? primaryDoc;
    const sideBDoc = findBestSideDoc(sides.side_b);

    // 7N-E: Strip metadata blocks from text_for_embedding before building draft
    // Patch 9H: Also strip SYN_COMPARE inline metadata (EVT IDs, "Ghi chú so sánh", "So sánh..." prefixes)
    const stripSynMetadata = (s: string): string => s
      .replace(/EVT_\d{3,4}/g, '').replace(/SYN_[A-Z_]+\d*/g, '')
      .replace(/Ghi chú\s+so sánh[^.]*\./gi, '')
      .replace(/^So sánh\s+[^:]+:\s*/gm, '')
      .replace(/;\s*;/g, ';').replace(/\s{2,}/g, ' ').trim();
    const docAContent = stripSynMetadata(
      (sideADoc.summary || sideADoc.text_for_embedding)
        .replace(/\[TIÊU ĐỀ\]\s*/g, '').replace(/\[LOẠI\][^\n]*/g, '')
        .replace(/\[THỜI GIAN\][^\n]*/g, '').replace(/\[GIAI ĐOẠN\][^\n]*/g, '')
        .replace(/\[NHÂN VẬT\][^\n]*/g, '').replace(/\[ĐỊA ĐIỂM\][^\n]*/g, '')
        .replace(/\[TỪ KHÓA[^\]]*\][^\n]*/g, '').replace(/\[TỔ CHỨC\][^\n]*/g, '')
        .replace(/\[TÓM TẮT\]\s*/g, '').replace(/\[BỐI CẢNH\][^\n]*/g, '')
        .replace(/\[Ý NGHĨA\][^\n]*/g, '').replace(/\[MÔ TẢ\]\s*/g, '')
        .replace(/\[[A-ZÀ-Ỹ][A-ZÀ-Ỹ\s]{1,20}\](?:\s*)/g, '').trim()
    );
    const docASummary = docAContent.split(/[.。]\s*/).filter(s => s.length > 5).slice(0, 2).join('. ').trim();
    const docATrunc = docASummary.length > 250 ? docASummary.substring(0, 250) + '...' : docASummary;

    let sideAPart = `- Vế thứ nhất (${sides.side_a}): ${docATrunc}`;
    let sideBPart: string;

    // Patch 9E-S: Use shared validateSideRelevance for both sides (already filtered in findBestSideDoc)
    // But sideADoc fallback to primaryDoc may bypass filter — revalidate here
    const validSideADoc = sideADoc && validateSideRelevance(sides.side_a, sideADoc) ? sideADoc : undefined;
    const validSideBDoc = sideBDoc && validateSideRelevance(sides.side_b, sideBDoc) ? sideBDoc : undefined;

    // Rebuild sideAPart if sideA doc is invalid (foreign)
    if (!validSideADoc) {
      sideAPart = `- Vế thứ nhất (${sides.side_a}): Ngữ cảnh hiện có chưa có đủ bằng chứng trực tiếp về ${sides.side_a} để so sánh đầy đủ.`;
    }

    if (validSideBDoc) {
      const docBContent = stripSynMetadata(
        (validSideBDoc.summary || validSideBDoc.text_for_embedding)
          .replace(/\[TIÊU ĐỀ\]\s*/g, '').replace(/\[LOẠI\][^\n]*/g, '')
          .replace(/\[THỜI GIAN\][^\n]*/g, '').replace(/\[GIAI ĐOẠN\][^\n]*/g, '')
          .replace(/\[NHÂN VẬT\][^\n]*/g, '').replace(/\[ĐỊA ĐIỂM\][^\n]*/g, '')
          .replace(/\[TỪ KHÓA[^\]]*\][^\n]*/g, '').replace(/\[TỔ CHỨC\][^\n]*/g, '')
          .replace(/\[TÓM TẮT\]\s*/g, '').replace(/\[BỐI CẢNH\][^\n]*/g, '')
          .replace(/\[Ý NGHĨA\][^\n]*/g, '').replace(/\[MÔ TẢ\]\s*/g, '')
          .replace(/\[[A-ZÀ-Ỹ][A-ZÀ-Ỹ\s]{1,20}\](?:\s*)/g, '').trim()
      );
      const docBSummary = docBContent.split(/[.。]\s*/).filter(s => s.length > 5).slice(0, 2).join('. ').trim();
      const docBTrunc = docBSummary.length > 250 ? docBSummary.substring(0, 250) + '...' : docBSummary;
      sideBPart = `- Vế thứ hai (${sides.side_b}): ${docBTrunc}`;
    } else {
      sideBPart = `- Vế thứ hai (${sides.side_b}): Ngữ cảnh hiện có chưa có đủ bằng chứng trực tiếp về ${sides.side_b} để so sánh đầy đủ.`;
    }

    // Determine which side (if any) is missing
    const missingSide = !validSideADoc ? sides.side_a : (!validSideBDoc ? sides.side_b : null);
    const hasBothSideEvidence = validSideADoc && validSideBDoc;

    if (hasBothSideEvidence) {
      draftAnswer = `Đây là hai chủ đề/sự kiện khác nhau.\n\n${sideAPart}\n\n${sideBPart}`;
    } else {
      draftAnswer = `Đây là hai chủ đề/sự kiện khác nhau, tuy nhiên ngữ cảnh hiện có chỉ đủ bằng chứng cho một vế.\n\n${sideAPart}\n\n${sideBPart}\n\nLưu ý: ngữ cảnh hiện có chưa có đủ bằng chứng trực tiếp về ${missingSide ?? 'một vế'}, nên câu trả lời chỉ mang tính kết luận một phần.`;
    }
  } else {
    // 7N-E: Strip metadata blocks from text_for_embedding
    const docContent = (primaryDoc.summary || primaryDoc.text_for_embedding)
      .replace(/\[TIÊU ĐỀ\]\s*/g, '').replace(/\[LOẠI\][^\n]*/g, '')
      .replace(/\[THỜI GIAN\][^\n]*/g, '').replace(/\[GIAI ĐOẠN\][^\n]*/g, '')
      .replace(/\[NHÂN VẬT\][^\n]*/g, '').replace(/\[ĐỊA ĐIỂM\][^\n]*/g, '')
      .replace(/\[TỪ KHÓA[^\]]*\][^\n]*/g, '').replace(/\[TỔ CHỨC\][^\n]*/g, '')
      .replace(/\[TÓM TẮT[^\]]*\][^\n]*/g, '').replace(/\[BỐI CẢNH\][^\n]*/g, '')
      .replace(/\[Ý NGHĨA\][^\n]*/g, '').replace(/\[ALIAS[^\]]*\][^\n]*/g, '')
      .replace(/\[MÔ TẢ\]\s*/g, '').replace(/\[[A-ZÀ-Ỹ][A-ZÀ-Ỹ\s]{1,20}\](?:\s*)/g, '')
      .trim();
    const yearStr = primaryDoc.year ? ` (${primaryDoc.year})` : '';

    const isPlanned = primaryDoc.event_status === 'planned_not_executed';
    const plannedNote = isPlanned
      ? ' Lưu ý: Sự kiện này được DỰ KIẾN nhưng CHƯA THỰC HIỆN.'
      : '';

    const sentences = docContent.split(/[.。]\s*/).filter(s => s.length > 5);
    // Patch 9G-R2: Skip sentences that just repeat the title
    const titleNormGen = primaryDoc.title.toLowerCase().normalize('NFKC');
    const meaningfulGen = sentences.filter(s => {
      const sn = s.toLowerCase().normalize('NFKC');
      if (titleNormGen.length > 10 && sn.includes(titleNormGen)) return false;
      if (sn.length < titleNormGen.length * 1.2 && titleNormGen.includes(sn)) return false;
      return true;
    });
    const finalGen = meaningfulGen.length > 0 ? meaningfulGen : sentences;
    // Patch 9G-R2: Increased from 2/300 to 4/500 to reduce title-only
    const mainContent = finalGen.slice(0, 4).join('. ').trim();
    const truncated = mainContent.length > 500 ? mainContent.substring(0, 500) + '...' : mainContent;

    draftAnswer = `Dựa trên tài liệu "${primaryDoc.title}"${yearStr}: ${truncated}${plannedNote}`;

    // Patch 8E: Target-only disambiguation contrast phrase
    // For queries like "Sự kiện nào nói về A, khác với B?" — add a short contrast.
    // This is NOT a full two-sided comparison; it's primary-focused with a brief distinction.
    const disambigMarker = queryFrame?.comparison_sides?.marker ?? '';
    const isTargetOnlyDisambig = queryFrame?.comparison_sides
      && ['disambiguation', 'multi_hop'].includes(queryFrame.intent)
      && ['khác với', 'khác gì với', 'không phải', 'phân biệt với'].includes(disambigMarker);

    if (isTargetOnlyDisambig && queryFrame?.comparison_sides) {
      const contrastSide = queryFrame.comparison_sides.side_b;
      // Try to find a contrast doc in supporting_docs
      const contrastTerms = contrastSide.toLowerCase().normalize('NFKC').split(/\s+/).filter(t => t.length > 2);
      const contrastDoc = bundle.supporting_docs.find(d => {
        const text = `${d.title} ${d.summary}`.toLowerCase().normalize('NFKC');
        return contrastTerms.filter(t => text.includes(t)).length >= 2;
      });

      if (contrastDoc) {
        const contrastSummary = contrastDoc.summary.length > 150
          ? contrastDoc.summary.substring(0, 150) + '...'
          : contrastDoc.summary;
        draftAnswer += ` Cần phân biệt với ${contrastSide}: ${contrastDoc.title} — ${contrastSummary}`;
      } else {
        // No contrast doc found — still add a brief distinction phrase
        draftAnswer += ` Cần phân biệt: sự kiện này khác với ${contrastSide}.`;
      }
    }
  }

  // Planned docs warning
  const plannedDocs = bundle.planned_not_executed_docs ?? [];
  let plannedWarning = '';
  if (plannedDocs.length > 0) {
    const plannedNames = plannedDocs.map(d => `"${d.title}"`).join(', ');
    plannedWarning = ` Lưu ý: Các sự kiện sau được dự kiến nhưng chưa/không thực hiện: ${plannedNames}.`;
  }

  // Build explanation from supporting docs
  let draftExplanation = '';
  if (bundle.supporting_docs.length > 0) {
    const supportParts = bundle.supporting_docs.map(d => {
      const year = d.year ? ` (${d.year})` : '';
      const summary = d.summary.length > 100 ? d.summary.substring(0, 100) + '...' : d.summary;
      return `${d.title}${year}: ${summary}`;
    });
    draftExplanation = `Các tài liệu hỗ trợ: ${supportParts.join('; ')}.`;
  }

  // Add disambiguation notes to explanation
  if (bundle.disambiguation_notes && bundle.disambiguation_notes.length > 0) {
    const disambigParts = bundle.disambiguation_notes.map(
      n => `${n.title}: ${n.reason}`
    );
    draftExplanation += ` Phân biệt: ${disambigParts.join('; ')}.`;
  }

  if (plannedWarning) {
    draftExplanation += plannedWarning;
  }

  // ── Patch 7G: Apply answer verifier ──
  let finalAnswer = draftAnswer;
  let finalExplanation = draftExplanation;
  let insufficientEvidence = false;
  let citationPolicy: 'clear' | 'focus_positive_only' | undefined;
  let comparisonNoiseYears: string[] | undefined;

  if (primaryDoc) {
    const verification = verifyAndRepairAnswer({
      query,
      draftAnswer,
      draftExplanation,
      queryFrame,
      contextBundle: bundle,
    });

    if (verification.revised_answer) {
      finalAnswer = verification.revised_answer;
    }
    if (verification.revised_explanation) {
      finalExplanation = verification.revised_explanation;
    }
    // 7L-D: Capture insufficient evidence flag
    if (verification.insufficient_evidence) {
      insufficientEvidence = true;
    }
    if (verification.citation_policy) {
      citationPolicy = verification.citation_policy;
    }

    // 7N-C: Capture comparison noise years for citation filtering
    if (verification.comparison_noise_years && verification.comparison_noise_years.length > 0) {
      comparisonNoiseYears = verification.comparison_noise_years;
    }

    if (verification.issues.length > 0) {
      const issueSummary = verification.issues.map(i => `${i.code}(${i.severity})`).join(', ');
      console.log(`   🔧 Verifier: ${issueSummary}`);
    }
  }

  // Build enriched fallback citations
  let citations = buildFallbackCitationsFromContext(bundle, dataset);

  // 7L-D: Citation hygiene — apply citation policy from verifier
  if (citationPolicy === 'clear') {
    // Insufficient evidence: clear all citations
    citations = [];
    confidence = 'low';
    console.log(`   🧹 Citation hygiene: cleared citations (insufficient_evidence=${insufficientEvidence})`);
  } else if (citationPolicy === 'focus_positive_only') {
    // Only keep citations that match focus-positive terms
    const focusProfile = detectFocusProfile(query, queryFrame);
    if (focusProfile) {
      const queryFocus = extractQueryFocus({ query, queryFrame });
      citations = citations.filter(c => {
        const doc = dataset.allCanonicalDocs.find(d => d.doc_id === c.record_id);
        if (!doc) return false;
        const focusScore = scoreDocumentFocus(doc, queryFocus);
        return focusScore.matched_terms.some(t => t.startsWith('profile+:'));
      });
      if (citations.length === 0) {
        confidence = 'low';
      }
      console.log(`   🧹 Citation hygiene: filtered to ${citations.length} focus-positive citations`);
    }
  }

  // 7N-C: Comparison noise-year citation filter
  if (comparisonNoiseYears && comparisonNoiseYears.length > 0 && citations.length > 0) {
    const beforeCount = citations.length;
    citations = citations.filter(c => {
      const doc = dataset.allCanonicalDocs.find(d => d.doc_id === c.record_id);
      if (!doc) return true;
      const titleNorm = doc.title.toLowerCase().normalize('NFKC');
      // Check if title contains noise-year topic patterns
      for (const ny of comparisonNoiseYears!) {
        if (titleNorm.includes(`hiến pháp ${ny}`) || titleNorm.includes(`hiệp định ${ny}`)) {
          return false;
        }
      }
      return true;
    });
    if (citations.length < beforeCount) {
      console.log(`   🧹 Comparison noise-year filter: ${beforeCount} → ${citations.length} citations (removed years: [${comparisonNoiseYears.join(',')}])`);
    }
  }

  // Patch 9F: Clamp confidence based on answer quality
  const clampedConfidence = clampConfidenceByAnswerQuality(
    confidence, finalAnswer, citations.length
  );

  // PATCH 9B: Apply output cleaner as the LAST step before returning to caller.
  // This removes any metadata tags that leaked from text_for_embedding into the answer/explanation.
  // Citation data structures (citations[]) are NOT modified — only display text.
  return {
    answer: cleanAnswerTextForUser(finalAnswer),
    explanation: cleanAnswerTextForUser(finalExplanation),
    citations,
    confidence: clampedConfidence,
    confidence_details: confidenceDetails,
    related_events: bundle.supporting_docs.map(d => ({
      record_id: d.doc_id,
      title: d.title,
    })),
  };
}

/** Extract basic citations from context bundle (without enrichment — used by LLM parse).
 *  Patch 7F-1: prefer citation_plan when available to avoid citing filtered-out docs.
 */
function extractBasicCitationsFromBundle(
  bundle: ContextBundle
): Citation[] {
  if (bundle.citation_plan && bundle.citation_plan.length > 0) {
    return [...bundle.citation_plan]
      .filter(item => item.citation_role !== 'excluded')
      .sort((a, b) => (a.priority ?? 99) - (b.priority ?? 99))
      .map(item => ({
        record_id: item.doc_id,
        title: '',
        relevance: item.citation_role === 'primary' ? 'primary'
                 : item.citation_role === 'contrast' ? 'contrast'
                 : 'supporting',
      }));
  }

  return [
    ...bundle.primary_docs.map(d => ({
      record_id: d.doc_id,
      title: d.title,
      relevance: 'primary',
    })),
    ...bundle.supporting_docs.map(d => ({
      record_id: d.doc_id,
      title: d.title,
      relevance: 'supporting',
    })),
  ];
}
