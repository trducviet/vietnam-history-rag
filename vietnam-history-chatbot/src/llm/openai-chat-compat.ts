/**
 * OpenAI Chat Completions Compatibility Layer — Patch 8X + 8Y
 *
 * Centralizes all OpenAI Chat Completions parameter building to ensure:
 * - max_completion_tokens is used instead of the deprecated max_tokens (8X)
 * - temperature is omitted for GPT-5 family models that reject custom values (8Y)
 *
 * Does NOT affect embedding API calls.
 */

import type OpenAI from 'openai';
import { isCloudLlmDisabled } from '../runtime/no-cloud-guard.js';

// ─── Types ───────────────────────────────────────────────────

/** Purpose label for logging/tracing */
export type ChatPurpose = 'routing' | 'reranking' | 'generation' | 'evaluation' | 'diagnostic';

export interface ChatCompletionParamsInput {
  model: string;
  messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[];
  maxTokens: number;
  temperature: number;
  responseFormat?: { type: 'json_object' | 'text' };
  purpose: ChatPurpose;
}

// ─── Token Parameter Strategy (Patch 8X) ─────────────────────

/**
 * Determine which token-limit parameter to use for a given model.
 *
 * Modern OpenAI Chat models (gpt-4o, gpt-4.1, gpt-5, etc.) require
 * `max_completion_tokens` instead of the deprecated `max_tokens`.
 *
 * Override with env RAG_CHAT_TOKEN_PARAM=max_tokens for legacy models.
 */
export function getTokenParamName(): 'max_completion_tokens' | 'max_tokens' {
  const override = process.env.RAG_CHAT_TOKEN_PARAM;
  if (override === 'max_tokens') return 'max_tokens';
  // Default: always use max_completion_tokens for modern models
  return 'max_completion_tokens';
}

// ─── Temperature Strategy (Patch 8Y) ─────────────────────────

export type TemperatureMode = 'auto' | 'omit' | 'send';

/**
 * Get the temperature handling mode from environment.
 * - 'auto': decide per-model (default — omit for GPT-5 family)
 * - 'omit': always omit temperature from requests
 * - 'send': always send temperature if caller provides it
 */
export function getTemperatureMode(): TemperatureMode {
  const mode = process.env.OPENAI_CHAT_TEMPERATURE_MODE?.toLowerCase();
  if (mode === 'omit') return 'omit';
  if (mode === 'send') return 'send';
  return 'auto';
}

/**
 * Whether a model supports custom temperature values.
 *
 * GPT-5 family models (gpt-5, gpt-5-mini, gpt-5.4, o1, o3, o4)
 * reject temperature != 1 with: "400 Unsupported value: 'temperature'
 * does not support 0 with this model."
 *
 * For these models, we omit the temperature parameter entirely
 * (API defaults to 1).
 */
export function supportsCustomTemperature(model: string): boolean {
  const m = model.toLowerCase();
  // GPT-5 family and reasoning models reject custom temperature
  if (m.startsWith('gpt-5') || m.startsWith('gpt-4.1') ||
      m.startsWith('o1') || m.startsWith('o3') || m.startsWith('o4')) {
    return false;
  }
  // Legacy/other models support custom temperature
  return true;
}

/**
 * Determine whether to include temperature in API request params.
 */
export function shouldSendTemperature(model: string): boolean {
  const mode = getTemperatureMode();
  if (mode === 'omit') return false;
  if (mode === 'send') return true;
  // auto: decide based on model
  return supportsCustomTemperature(model);
}

// ─── LLM Mode ────────────────────────────────────────────────

export type LLMMode = 'auto' | 'off';

/**
 * Get current LLM mode from environment.
 * - 'auto': call LLM if API key present; fall back on error
 * - 'off': never call LLM, always use deterministic fallback
 */
export function getLLMMode(): LLMMode {
  const mode = process.env.RAG_LLM_MODE?.toLowerCase();
  if (mode === 'off') return 'off';
  return 'auto';
}

/**
 * Whether LLM calls should be attempted.
 * Returns false if mode is 'off' or API key is missing.
 */
export function shouldCallLLM(apiKey: string): boolean {
  if (isCloudLlmDisabled()) return false;
  if (getLLMMode() === 'off') return false;
  return !!apiKey;
}

// ─── Parameter Builder ───────────────────────────────────────

/**
 * Build parameters for openai.chat.completions.create().
 *
 * Ensures:
 * - Never sends both max_tokens and max_completion_tokens (8X)
 * - Uses the correct token parameter for the current model strategy (8X)
 * - Omits temperature for GPT-5 family models (8Y)
 * - Never sends undefined/null values
 */
export function buildChatCompletionParams(
  input: ChatCompletionParamsInput
): Record<string, unknown> {
  const tokenParam = getTokenParamName();

  const params: Record<string, unknown> = {
    model: input.model,
    messages: input.messages,
    [tokenParam]: input.maxTokens,
  };

  // Patch 8Y: only include temperature if model supports it
  if (shouldSendTemperature(input.model)) {
    params.temperature = input.temperature;
  }

  if (input.responseFormat) {
    params.response_format = input.responseFormat;
  }

  return params;
}

// ─── Logging Helper ──────────────────────────────────────────

/** Log LLM configuration at startup (never logs API key) */
export function logLLMConfig(apiKey: string, model?: string): void {
  const keyStatus = apiKey ? 'present' : 'missing';
  const mode = getLLMMode();
  const tokenParam = getTokenParamName();
  const tempMode = getTemperatureMode();
  const tempSupported = model ? supportsCustomTemperature(model) : 'N/A';
  console.log(
    `🤖 LLM config: mode=${mode} | key=${keyStatus} | token_param=${tokenParam}` +
    ` | temp_mode=${tempMode} | temp_supported=${tempSupported}`
  );
}
