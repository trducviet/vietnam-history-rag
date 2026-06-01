/**
 * LLM Answer Judge — optional module for LLM-based answer quality evaluation.
 *
 * Uses gpt-5-mini to judge faithfulness, completeness, and citation quality.
 * Only runs when OPENAI_API_KEY is available and --llm-judge flag is passed.
 *
 * PATCH 6: New file (skeleton + implementation).
 */

import OpenAI from 'openai';
import { config } from '../shared/config.js';
import { buildChatCompletionParams, shouldCallLLM } from '../llm/openai-chat-compat.js';

// ─── Types ───────────────────────────────────────────────────

export interface LLMJudgeResult {
  faithfulness: number;    // 1-5
  completeness: number;    // 1-5
  citation_quality: number; // 1-5
  notes: string;
}

// ─── Judge Implementation ────────────────────────────────────

const JUDGE_SYSTEM_PROMPT = `Bạn là một giám khảo đánh giá chất lượng câu trả lời của chatbot lịch sử Việt Nam.

Cho một câu hỏi, câu trả lời, và danh sách citations, hãy đánh giá:

1. **faithfulness** (1-5): Câu trả lời có trung thực với tài liệu được trích dẫn không?
   - 5: Hoàn toàn trung thực
   - 3: Có vài chi tiết không rõ nguồn
   - 1: Nhiều thông tin bịa đặt

2. **completeness** (1-5): Câu trả lời có đầy đủ không?
   - 5: Trả lời đầy đủ, rõ ràng
   - 3: Thiếu một số chi tiết quan trọng
   - 1: Rất sơ sài hoặc không trả lời

3. **citation_quality** (1-5): Citations có phù hợp và hữu ích không?
   - 5: Citations chính xác, đủ
   - 3: Citations có nhưng thiếu hoặc thừa
   - 1: Citations sai hoặc không có

Trả về JSON: {"faithfulness": N, "completeness": N, "citation_quality": N, "notes": "..."}`;

/**
 * Judge an answer using LLM. Returns null if API key is not available.
 */
export async function judgeAnswer(
  query: string,
  answer: string,
  citationTitles: string[]
): Promise<LLMJudgeResult | null> {
  if (!shouldCallLLM(config.openaiApiKey)) {
    return null;
  }

  try {
    const openai = new OpenAI({ apiKey: config.openaiApiKey });

    const userMessage = `Câu hỏi: ${query}

Câu trả lời: ${answer}

Citations: ${citationTitles.join('; ') || '(không có)'}`;

    const params = buildChatCompletionParams({
      model: config.routerModel,
      messages: [
        { role: 'system', content: JUDGE_SYSTEM_PROMPT },
        { role: 'user', content: userMessage },
      ],
      maxTokens: 200,
      temperature: 0,
      responseFormat: { type: 'json_object' },
      purpose: 'evaluation',
    });

    const response = await openai.chat.completions.create(params as any);

    const content = response.choices[0]?.message?.content;
    if (!content) return null;

    const parsed = JSON.parse(content) as Record<string, unknown>;

    return {
      faithfulness: clampScore(parsed.faithfulness),
      completeness: clampScore(parsed.completeness),
      citation_quality: clampScore(parsed.citation_quality),
      notes: typeof parsed.notes === 'string' ? parsed.notes : '',
    };
  } catch (err) {
    console.warn(`⚠️ LLM judge failed: ${(err as Error).message}`);
    return null;
  }
}

function clampScore(val: unknown): number {
  const n = typeof val === 'number' ? val : parseInt(String(val), 10);
  if (isNaN(n)) return 3;
  return Math.max(1, Math.min(5, Math.round(n)));
}
