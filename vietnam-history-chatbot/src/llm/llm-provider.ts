import type { LLMProvider } from './llm-types.js';
import { loadLocalLLMConfig, type LocalLLMConfig } from './llm-config.js';
import { NoopLLMProvider } from './providers/noop-provider.js';
import { LocalOpenAICompatibleProvider } from './providers/local-openai-compatible-provider.js';

export function createLLMProvider(config: LocalLLMConfig = loadLocalLLMConfig()): LLMProvider {
  if (config.backend === 'none') return new NoopLLMProvider();
  if (config.backend === 'local') return new LocalOpenAICompatibleProvider(config);
  throw new Error('Cloud LLM provider is not enabled by the local-first backend contract.');
}

export type { LLMProvider } from './llm-types.js';
