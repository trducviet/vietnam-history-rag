/**
 * External API Guard — Patch 8F-C
 *
 * Centralized guard to prevent ANY external API calls in no-cost mode.
 * All code paths that could reach OpenAI (chat, embedding, vector rescue)
 * MUST check this guard before making requests.
 *
 * Policy (deny-by-default):
 * - RAG_LLM_MODE=off → block ALL external API (chat + embedding + vector)
 * - RAG_ALLOW_EXTERNAL_API !== "1" → block ALL external API
 * - Only when BOTH RAG_LLM_MODE=auto AND RAG_ALLOW_EXTERNAL_API=1 → allow
 */

import { isNoCloudMode as isRuntimeNoCloudMode } from '../runtime/no-cloud-guard.js';

// ─── Core Guard Functions ────────────────────────────────────

/**
 * Whether the system is in no-cost mode.
 * True when RAG_LLM_MODE is 'off'.
 */
export function isNoCostMode(): boolean {
  return (process.env.RAG_LLM_MODE?.toLowerCase() ?? 'auto') === 'off';
}

/**
 * Whether external API calls are allowed.
 * Returns true ONLY when:
 * - RAG_LLM_MODE is NOT 'off'
 * - RAG_ALLOW_EXTERNAL_API is explicitly '1'
 *
 * Default is DENY — API key presence alone does NOT grant permission.
 */
export function shouldAllowExternalApi(): boolean {
  if (isRuntimeNoCloudMode()) return false;
  if (isNoCostMode()) return false;
  return process.env.RAG_ALLOW_EXTERNAL_API === '1';
}

/**
 * Assert that external API calls are allowed.
 * Throws if not allowed — caller should catch and fall back.
 */
export function assertExternalApiAllowed(operationName: string): void {
  if (!shouldAllowExternalApi()) {
    throw new ExternalApiBlockedError(operationName);
  }
}

/**
 * Mask an API key for safe logging.
 * Never logs the actual key — only 'present', 'missing', or first 4 chars masked.
 */
export function maskApiKeyForLog(value: string | undefined): string {
  if (!value) return 'missing';
  if (value.length <= 8) return 'present (short)';
  return `${value.substring(0, 4)}${'*'.repeat(8)}`;
}

/**
 * Whether live diagnostic mode is enabled.
 */
export function isLiveDiagnosticEnabled(): boolean {
  return process.env.OPENAI_LIVE_DIAGNOSTIC === '1';
}

/**
 * Log guard status at startup (never logs API key content).
 */
export function logGuardStatus(): void {
  const mode = process.env.RAG_LLM_MODE ?? 'auto';
  const allowApi = process.env.RAG_ALLOW_EXTERNAL_API ?? '0';
  const liveDiag = process.env.OPENAI_LIVE_DIAGNOSTIC ?? '0';
  const keyStatus = process.env.OPENAI_API_KEY ? 'present' : 'missing';
  console.log(
    `🔒 API Guard: mode=${mode} | allow_api=${allowApi} | live_diag=${liveDiag}` +
    ` | key=${keyStatus} | external_api=${shouldAllowExternalApi() ? 'ALLOWED' : 'BLOCKED'}`
  );
}

// ─── Error Type ──────────────────────────────────────────────

export class ExternalApiBlockedError extends Error {
  constructor(operationName: string) {
    super(`External API blocked for "${operationName}": no-cost guard active (RAG_LLM_MODE=${process.env.RAG_LLM_MODE ?? 'auto'}, RAG_ALLOW_EXTERNAL_API=${process.env.RAG_ALLOW_EXTERNAL_API ?? '0'})`);
    this.name = 'ExternalApiBlockedError';
  }
}
