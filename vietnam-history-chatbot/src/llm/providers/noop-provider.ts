import type { FinalAnswerInputPackage, LLMAnswerOutput, LLMHealthCheckResult, LLMProvider } from '../llm-types.js';

export class NoopLLMProvider implements LLMProvider {
  readonly name = 'noop';
  readonly mode = 'none' as const;

  async generate(input: FinalAnswerInputPackage): Promise<LLMAnswerOutput> {
    const safeMode = input.safe_mode;
    return {
      answer_text: safeMode === 'safe_clarification'
        ? 'Mình cần bạn nói rõ hơn câu hỏi muốn hỏi về sự kiện hoặc nhân vật nào.'
        : 'Chưa gọi mô hình ngôn ngữ. Đây là backend noop dùng để giữ chế độ không gọi LLM.',
      citations_used: [],
      answer_mode: safeMode === 'safe_clarification'
        ? 'safe_clarification'
        : safeMode === 'safe_out_of_scope'
          ? 'safe_out_of_scope'
          : safeMode === 'safe_refusal'
            ? 'safe_refusal'
            : 'safe_insufficient_data',
      unsupported_claims_self_check: [],
      confidence: 'low',
      needs_more_context: true,
    };
  }

  async healthCheck(): Promise<LLMHealthCheckResult> {
    return {
      checked: true,
      base_url: 'none',
      is_localhost: false,
      server_reachable: false,
      model_detected: false,
      model_name: 'none',
      local_test_generation_run: false,
      local_test_generation_passed: false,
      notes: ['Noop provider selected; no local server request was made.'],
    };
  }
}
