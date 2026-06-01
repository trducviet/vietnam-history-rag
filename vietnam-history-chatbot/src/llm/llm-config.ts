import type { LLMBackendMode } from './llm-types.js';

export interface LocalLLMConfig {
  backend: LLMBackendMode;
  provider: 'noop' | 'ollama_openai_compatible' | string;
  baseUrl: string;
  model: string;
  temperature: number;
  maxTokens: number;
  timeoutMs: number;
  strictContextOnly: boolean;
  apiKey: string;
}

export const DEFAULT_LOCAL_LLM_CONFIG: LocalLLMConfig = {
  backend: 'local',
  provider: 'ollama_openai_compatible',
  baseUrl: 'http://localhost:11434/v1',
  model: 'qwen2.5:3b-instruct',
  temperature: 0.1,
  maxTokens: 700,
  timeoutMs: 120000,
  strictContextOnly: true,
  apiKey: 'ollama',
};

export function loadLocalLLMConfig(env: NodeJS.ProcessEnv = process.env): LocalLLMConfig {
  return {
    backend: parseBackend(env.LLM_BACKEND ?? DEFAULT_LOCAL_LLM_CONFIG.backend),
    provider: env.LOCAL_LLM_PROVIDER ?? DEFAULT_LOCAL_LLM_CONFIG.provider,
    baseUrl: env.LOCAL_LLM_BASE_URL ?? DEFAULT_LOCAL_LLM_CONFIG.baseUrl,
    model: env.LOCAL_LLM_MODEL ?? DEFAULT_LOCAL_LLM_CONFIG.model,
    temperature: parseNumber(env.LOCAL_LLM_TEMPERATURE, DEFAULT_LOCAL_LLM_CONFIG.temperature),
    maxTokens: parseInteger(env.LOCAL_LLM_MAX_TOKENS, DEFAULT_LOCAL_LLM_CONFIG.maxTokens),
    timeoutMs: parseInteger(env.LOCAL_LLM_TIMEOUT_MS, DEFAULT_LOCAL_LLM_CONFIG.timeoutMs),
    strictContextOnly: parseBoolean(env.LOCAL_LLM_STRICT_CONTEXT_ONLY, DEFAULT_LOCAL_LLM_CONFIG.strictContextOnly),
    apiKey: env.LOCAL_LLM_API_KEY ?? DEFAULT_LOCAL_LLM_CONFIG.apiKey,
  };
}

export function isLocalhostUrl(rawUrl: string): boolean {
  try {
    const url = new URL(rawUrl);
    if (!['http:', 'https:'].includes(url.protocol)) return false;
    return ['localhost', '127.0.0.1', '[::1]', '::1'].includes(url.hostname);
  } catch {
    return false;
  }
}

export function assertLocalhostBaseUrl(rawUrl: string): void {
  if (!isLocalhostUrl(rawUrl)) {
    throw new Error(`Blocked non-local LLM base URL: ${rawUrl}`);
  }
}

function parseBackend(value: string): LLMBackendMode {
  const normalized = value.toLowerCase();
  if (normalized === 'none' || normalized === 'local' || normalized === 'cloud') return normalized;
  return 'local';
}

function parseNumber(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseInteger(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  return ['1', 'true', 'yes', 'on'].includes(value.toLowerCase());
}
