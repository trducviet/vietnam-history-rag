/**
 * Reranker — uses gpt-5-mini / gpt-5.4-mini to rerank top-k retrieval
 * candidates based on relevance to the query. Falls back to pass-through
 * when LLM is unavailable.
 */

import OpenAI from 'openai';
import { config } from '../shared/config.js';
import { buildChatCompletionParams, shouldCallLLM } from '../llm/openai-chat-compat.js';
import { isCloudRerankerDisabled } from '../runtime/no-cloud-guard.js';
import type {
  HybridSearchResult,
  RerankedResult,
  LoadedDataset,
} from '../shared/types.js';

// ─── Reranking Prompt ────────────────────────────────────────

const RERANK_SYSTEM_PROMPT = `Bạn là một chuyên gia đánh giá mức độ liên quan giữa câu hỏi và tài liệu lịch sử Việt Nam.

Nhiệm vụ: cho một câu hỏi và danh sách tài liệu, đánh giá mức độ liên quan của từng tài liệu với câu hỏi.

Trả về JSON array, mỗi phần tử có:
- "doc_id": ID tài liệu
- "relevance_score": điểm từ 0.0 đến 1.0 (1.0 = rất liên quan)
- "reason": lý do ngắn gọn (1 câu)

Quy tắc đánh giá:
- Tài liệu trả lời trực tiếp câu hỏi → 0.8–1.0
- Tài liệu cung cấp context hữu ích → 0.5–0.7
- Tài liệu liên quan nhưng không trả lời trực tiếp → 0.2–0.4
- Tài liệu không liên quan → 0.0–0.1

Trả về ĐÚNG JSON array, không kèm text khác.`;

// ─── Reranker Implementation ─────────────────────────────────

/**
 * Rerank retrieval results using LLM relevance scoring.
 * Falls back to pass-through (original scores) when LLM is unavailable.
 */
export async function rerankResults(
  query: string,
  candidates: HybridSearchResult[],
  dataset: LoadedDataset,
  maxCandidates: number = 10
): Promise<RerankedResult[]> {
  const trimmed = candidates.slice(0, maxCandidates);

  if (isCloudRerankerDisabled()) {
    return passthroughRerank(trimmed);
  }

  try {
    return await rerankWithLLM(query, trimmed, dataset);
  } catch (error) {
    console.warn('⚠️  LLM reranking failed, using original scores:', (error as Error).message);
    return passthroughRerank(trimmed);
  }
}

/** LLM-based reranking */
async function rerankWithLLM(
  query: string,
  candidates: HybridSearchResult[],
  dataset: LoadedDataset
): Promise<RerankedResult[]> {
  if (!shouldCallLLM(config.openaiApiKey)) {
    throw new Error('LLM disabled or no API key — falling back to passthrough');
  }

  // Build document summaries for the LLM
  const docSummaries = candidates.map((c, idx) => {
    const doc = dataset.events.get(c.doc_id) ?? dataset.synthesis.get(c.doc_id);
    const title = doc?.title ?? c.metadata.title;
    const summary = doc?.summary ?? '';
    const year = c.metadata.year ? `(${c.metadata.year})` : '';
    const status = c.metadata.event_status !== 'actual' ? ` [${c.metadata.event_status}]` : '';

    return `[${idx + 1}] doc_id: ${c.doc_id}\n    Tiêu đề: ${title} ${year}${status}\n    Tóm tắt: ${summary}`;
  }).join('\n\n');

  const userPrompt = `Câu hỏi: ${query}\n\nDanh sách tài liệu:\n${docSummaries}`;

  const openai = new OpenAI({ apiKey: config.openaiApiKey });

  const params = buildChatCompletionParams({
    model: config.routerModel,
    messages: [
      { role: 'system', content: RERANK_SYSTEM_PROMPT },
      { role: 'user', content: userPrompt },
    ],
    maxTokens: 800,
    temperature: 0,
    responseFormat: { type: 'json_object' },
    purpose: 'reranking',
  });

  const response = await openai.chat.completions.create(params as any);

  const content = response.choices[0]?.message?.content;
  if (!content) throw new Error('Empty response from reranker');

  const parsed = JSON.parse(content);
  const scores: Array<{ doc_id: string; relevance_score: number }> =
    Array.isArray(parsed) ? parsed : (parsed.results ?? parsed.documents ?? []);

  // Map LLM scores back to results
  const scoreMap = new Map<string, number>();
  for (const s of scores) {
    if (s.doc_id && typeof s.relevance_score === 'number') {
      scoreMap.set(s.doc_id, s.relevance_score);
    }
  }

  const reranked: RerankedResult[] = candidates.map(c => ({
    doc_id: c.doc_id,
    original_score: c.combined_score,
    rerank_score: scoreMap.get(c.doc_id) ?? c.combined_score * 0.5,
    metadata: c.metadata,
  }));

  // Sort by rerank score descending
  reranked.sort((a, b) => b.rerank_score - a.rerank_score);
  return reranked;
}

/** Passthrough: use original combined scores as rerank scores */
function passthroughRerank(candidates: HybridSearchResult[]): RerankedResult[] {
  return candidates.map(c => ({
    doc_id: c.doc_id,
    original_score: c.combined_score,
    rerank_score: c.combined_score,
    metadata: c.metadata,
  }));
}
