/**
 * Runtime no-cloud guard for local-first execution.
 *
 * This guard is intentionally environment driven because processChat can be
 * used by API, benchmark, and data-pack scripts. In no-cloud/local mode, cloud
 * helpers must be skipped before any provider request is attempted.
 */

export class NoCloudViolationError extends Error {
  constructor(context: string, detail?: string) {
    super(`No-cloud guard blocked "${context}"${detail ? `: ${detail}` : ''}`);
    this.name = 'NoCloudViolationError';
  }
}

function truthy(value: string | undefined): boolean {
  return value === '1' || value?.toLowerCase() === 'true' || value?.toLowerCase() === 'yes' || value?.toLowerCase() === 'on';
}

export function isNoCloudMode(env: NodeJS.ProcessEnv = process.env): boolean {
  return truthy(env.NO_CLOUD)
    || truthy(env.DISABLE_EXTERNAL_NETWORK)
    || env.LLM_BACKEND?.toLowerCase() === 'local'
    || env.LOCAL_LLM_PROVIDER?.toLowerCase().includes('ollama') === true;
}

export function isCloudRouterDisabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return isNoCloudMode(env) || truthy(env.DISABLE_CLOUD_ROUTER);
}

export function isCloudRerankerDisabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return isNoCloudMode(env) || truthy(env.DISABLE_CLOUD_RERANKER);
}

export function isCloudEmbeddingDisabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return isNoCloudMode(env) || truthy(env.DISABLE_CLOUD_EMBEDDING);
}

export function isCloudLlmDisabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return isNoCloudMode(env) || truthy(env.DISABLE_CLOUD_LLM);
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

export function assertNoCloudUrl(rawUrl: string, context: string): void {
  if (isNoCloudMode() && !isLocalhostUrl(rawUrl)) {
    throw new NoCloudViolationError(context, `non-local URL ${rawUrl}`);
  }
}

export function assertCloudProviderNotAllowed(context: string): void {
  if (isCloudLlmDisabled()) {
    throw new NoCloudViolationError(context, 'cloud LLM/provider disabled');
  }
}

export function noCloudEnvironmentSummary(env: NodeJS.ProcessEnv = process.env): Record<string, unknown> {
  return {
    no_cloud: isNoCloudMode(env),
    llm_backend: env.LLM_BACKEND ?? null,
    local_llm_provider: env.LOCAL_LLM_PROVIDER ?? null,
    disable_cloud_llm: truthy(env.DISABLE_CLOUD_LLM),
    disable_cloud_router: truthy(env.DISABLE_CLOUD_ROUTER),
    disable_cloud_reranker: truthy(env.DISABLE_CLOUD_RERANKER),
    disable_cloud_embedding: truthy(env.DISABLE_CLOUD_EMBEDDING),
    disable_external_network: truthy(env.DISABLE_EXTERNAL_NETWORK),
    allow_localhost_only: truthy(env.ALLOW_LOCALHOST_ONLY),
  };
}
