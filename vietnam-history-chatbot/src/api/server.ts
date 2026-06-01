/**
 * Express API Server — serves the RAG chatbot and benchmark endpoints.
 *
 * PATCH 1: Uses dual BM25 indexes (event + synthesis).
 * PATCH 2: Loads vector caches at startup if available. Health endpoint shows vector status.
 *
 * Endpoints:
 *   POST /api/chat     — process a user question
 *   GET  /api/health   — health check
 *   GET  /api/stats    — dataset statistics
 */

import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { config, validateConfig } from '../shared/config.js';
import { loadDataset } from '../ingestion/data-loader.js';
import { loadEmbeddingCacheToVectorStore } from '../ingestion/embedding-generator.js';
import { BM25Index, BaselineTokenizer } from '../retrieval/bm25-index.js';
import { InMemoryVectorStore } from '../shared/vector-store.js';
import { processChat, type ChatPipelineConfig } from '../chat/pipeline.js';
import type { SessionMemoryState } from '../chat/session-memory.js';
import type { LoadedDataset } from '../shared/types.js';

process.env.NO_CLOUD ??= '1';
process.env.DISABLE_CLOUD_LLM ??= '1';
process.env.DISABLE_CLOUD_EMBEDDING ??= '1';
process.env.DISABLE_CLOUD_ROUTER ??= '1';
process.env.DISABLE_CLOUD_RERANKER ??= '1';
process.env.RAG_ALLOW_EXTERNAL_API ??= '0';
process.env.RAG_LLM_MODE ??= 'off';

const __dirname = dirname(fileURLToPath(import.meta.url));
type WebChatSessionState = {
  lastQuery?: string;
  lastFocus?: string;
  sessionMemory?: SessionMemoryState;
};

const localHybridSessions = new Map<string, WebChatSessionState>();
const RAG_SERVICE_URL = (process.env.RAG_SERVICE_URL || 'http://127.0.0.1:31114').replace(/\/+$/, '');

function getWebChatSession(sessionId: string): WebChatSessionState {
  const existing = localHybridSessions.get(sessionId);
  if (existing) return existing;
  const created: WebChatSessionState = {};
  localHybridSessions.set(sessionId, created);
  return created;
}

function updateWebChatSession(sessionId: string, patch: Partial<WebChatSessionState>): WebChatSessionState {
  const current = getWebChatSession(sessionId);
  const next = { ...current, ...patch };
  localHybridSessions.set(sessionId, next);
  return next;
}

function extractSessionMemory(response: any): SessionMemoryState | undefined {
  const state = response?.metadata?.memory?.session_memory_state;
  if (!state || typeof state !== 'object') return undefined;
  if (typeof state.session_id !== 'string' || typeof state.turn_count !== 'number') return undefined;
  return state as SessionMemoryState;
}

function rememberProcessChatTurn(sessionId: string, message: string, response: any): void {
  const memoryState = extractSessionMemory(response);
  const effectiveQuery = response?.metadata?.memory?.effective_query
    || response?.metadata?.memory?.rewrite?.rewritten_query
    || message;
  updateWebChatSession(sessionId, {
    lastQuery: message,
    lastFocus: effectiveQuery,
    ...(memoryState ? { sessionMemory: memoryState } : {}),
  });
}

function ragServiceEnabled(): boolean {
  return (process.env.RAG_SERVICE_ENABLED || 'true').toLowerCase() !== 'false';
}

function ragCliFallbackAllowed(): boolean {
  return (process.env.ALLOW_RAG_CLI_FALLBACK || 'false').toLowerCase() === 'true';
}

async function callRagService(path: string, payload?: Record<string, unknown>, method = 'POST'): Promise<{ status: number; data: any }> {
  const timeoutMs = Number(process.env.RAG_SERVICE_TIMEOUT_MS || 90000);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${RAG_SERVICE_URL}${path}`, {
      method,
      headers: method === 'POST' ? { 'Content-Type': 'application/json' } : undefined,
      body: method === 'POST' ? JSON.stringify(payload || {}) : undefined,
      signal: controller.signal,
    });
    const text = await response.text();
    let data: any;
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { raw_text: text };
    }
    return { status: response.status, data };
  } finally {
    clearTimeout(timer);
  }
}

async function getRagServiceHealth(): Promise<any> {
  if (!ragServiceEnabled()) {
    return { enabled: false, status: 'disabled', url: RAG_SERVICE_URL };
  }
  try {
    const health = await callRagService('/health', undefined, 'GET');
    const ready = await callRagService('/ready', undefined, 'GET');
    return {
      enabled: true,
      url: RAG_SERVICE_URL,
      health_status: health.status,
      ready_status: ready.status,
      health: health.data,
      ready: ready.data,
    };
  } catch (error) {
    return {
      enabled: true,
      url: RAG_SERVICE_URL,
      status: 'unavailable',
      error: (error as Error).message,
    };
  }
}

function runLocalHybridWebChat(payload: Record<string, unknown>): Promise<any> {
  const scriptPath = resolve(__dirname, '../../../scripts/web-demo/local-hybrid-chat-cli.py');
  const workspaceRoot = resolve(__dirname, '../../..');
  const pythonBin = process.env.PYTHON || 'python';
  const env = {
    ...process.env,
    NO_CLOUD: 'true',
    DISABLE_EXTERNAL_NETWORK: 'true',
    ALLOW_LOCALHOST_ONLY: 'true',
    DISABLE_CLOUD_LLM: 'true',
    DISABLE_CLOUD_EMBEDDING: 'true',
    DISABLE_CLOUD_ROUTER: 'true',
    DISABLE_CLOUD_RERANKER: 'true',
    RAG_ALLOW_EXTERNAL_API: '0',
    RAG_LLM_MODE: 'off',
    LLM_BACKEND: 'local',
    LOCAL_LLM_PROVIDER: 'ollama_openai_compatible',
    LOCAL_LLM_BASE_URL: 'http://localhost:11434/v1',
    LOCAL_LLM_MODEL: 'qwen2.5:3b-instruct',
    TRANSFORMERS_OFFLINE: '1',
    HF_HUB_OFFLINE: '1',
  };

  return new Promise((resolvePromise, reject) => {
    const child = spawn(pythonBin, [scriptPath], {
      cwd: workspaceRoot,
      env,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error('local hybrid runtime timed out'));
    }, 210_000);

    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    child.on('error', err => {
      clearTimeout(timer);
      reject(err);
    });
    child.on('close', code => {
      clearTimeout(timer);
      try {
        const parsed = JSON.parse(stdout.trim());
        if (code !== 0 && parsed?.error) {
          reject(new Error(`${parsed.message || parsed.error}${stderr ? `\n${stderr}` : ''}`));
          return;
        }
        resolvePromise(parsed);
      } catch (err) {
        reject(new Error(`Failed to parse local hybrid runtime output: ${(err as Error).message}\n${stderr}`));
      }
    });
    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}

function runNineRouterCloudRetrieval(payload: Record<string, unknown>): Promise<any> {
  const scriptPath = resolve(__dirname, '../../../scripts/web-demo/9router-cloud-retrieval-cli.py');
  const workspaceRoot = resolve(__dirname, '../../..');
  const pythonBin = process.env.PYTHON || 'python';
  const env = {
    ...process.env,
    RAG_RUNTIME_MODE: 'api_9router_fast',
    RAG_ALLOW_EXTERNAL_API: '1',
    ALLOW_LOCALHOST_ONLY: 'true',
  };

  return new Promise((resolvePromise, reject) => {
    const child = spawn(pythonBin, [scriptPath], {
      cwd: workspaceRoot,
      env,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error('9Router cloud retrieval runtime timed out'));
    }, 210_000);

    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    child.on('error', err => {
      clearTimeout(timer);
      reject(err);
    });
    child.on('close', code => {
      clearTimeout(timer);
      try {
        const parsed = JSON.parse(stdout.trim());
        if (code !== 0 && parsed?.error) {
          reject(new Error(`${parsed.message || parsed.error}${stderr ? `\n${stderr}` : ''}`));
          return;
        }
        resolvePromise(parsed);
      } catch (err) {
        reject(new Error(`Failed to parse 9Router cloud retrieval output: ${(err as Error).message}\n${stderr}`));
      }
    });
    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}

/** Detect response mode from pipeline output */
function detectResponseMode(response: any): string {
  if (response._capabilityDecision?.policy === 'REFUSE_OOS') return 'oos';
  if (response._capabilityDecision?.policy === 'CLARIFY') return 'clarification';
  if (/ngoài phạm vi|nằm ngoài/i.test(response.answer || '')) return 'oos';
  if (/cụ thể hơn|chưa rõ|chứa đại từ/i.test(response.answer || '')) return 'clarification';
  return 'bm25_fallback';
}

/** Remove internal record IDs from user-facing answer text */
function sanitizeResponse(response: any): any {
  const sanitized = { ...response };
  if (sanitized.answer) {
    sanitized.answer = sanitized.answer
      .replace(/\bEVT_\d{4}\b/g, '')
      .replace(/\bSYN_\w+\b/g, '')
      .replace(/\bdoc_id:\s*\S+/gi, '')
      .replace(/\s{2,}/g, ' ')
      .trim();
  }
  return sanitized;
}

async function checkLocalOllama(): Promise<'ok' | 'missing' | 'unknown'> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 800);
  try {
    const response = await fetch('http://localhost:11434/api/tags', {
      signal: controller.signal,
    });
    return response.ok ? 'ok' : 'missing';
  } catch {
    return 'missing';
  } finally {
    clearTimeout(timeout);
  }
}

type NineRouterConfig = {
  baseUrl: string;
  apiKey: string;
  model: string;
  timeoutMs: number;
  maxTokens: number;
  temperature: number;
  configured: boolean;
  missing: string[];
};

type AnswerProviderResult = {
  answer: string;
  provider: '9router_api';
  model: string;
  latency_ms: number;
  api_cloud_calls: number;
  raw_provider_response_redacted: null;
};

const API_FAST_METADATA_LEAK_RE = /(synthesis\/[a-z_]+|timeline_summary|comparison_note|event\s*\|\s*actual|actual\s+\d{4}|Câu hỏi alias trỏ tới|fallback noted|bm25_fallback|query embedding cache|citation_aware_fallback|template_name)/i;

function getNineRouterConfig(): NineRouterConfig {
  const baseUrl = (process.env['9ROUTER_BASE_URL'] || 'http://localhost:20128/v1').replace(/\/+$/, '');
  const apiKey = process.env['9ROUTER_API_KEY'] || '';
  const model = process.env['9ROUTER_MODEL'] || '';
  const timeoutMs = Number(process.env['9ROUTER_TIMEOUT_MS'] || 60000);
  const maxTokens = Number(process.env['9ROUTER_MAX_TOKENS'] || 512);
  const temperature = Number(process.env['9ROUTER_TEMPERATURE'] || 0.1);
  const missing = [
    !apiKey ? '9ROUTER_API_KEY' : '',
    !model ? '9ROUTER_MODEL' : '',
  ].filter(Boolean);
  return {
    baseUrl,
    apiKey,
    model,
    timeoutMs,
    maxTokens,
    temperature,
    configured: missing.length === 0,
    missing,
  };
}

function extractCitationMarkers(citations: any[]): Set<string> {
  return new Set((citations || []).map(citation => String(citation?.marker || '')).filter(Boolean));
}

function findInvalidCitationMarkers(answer: string, citations: any[]): string[] {
  const available = extractCitationMarkers(citations);
  const used = Array.from(new Set((answer.match(/\[[0-9]+\]/g) || [])));
  return used.filter(marker => !available.has(marker));
}

function buildNineRouterContext(citations: any[]): string {
  return (citations || []).map((citation, index) => {
    const marker = citation?.marker || `[${index + 1}]`;
    const title = citation?.title || `Nguồn ${index + 1}`;
    const snippet = citation?.snippet || '';
    return `${marker} Title: ${title}\nSnippet: ${snippet}`;
  }).join('\n\n');
}

function buildNineRouterTask(message: string, retrievalPayload: any): string {
  const generationPayload = retrievalPayload?.generation_payload || {};
  const context = buildNineRouterContext(retrievalPayload?.citations || []);
  const intent = generationPayload.intent || retrievalPayload?.debug?.intent || 'fact';
  return [
    `QUESTION: ${message}`,
    '',
    `INTENT: ${intent}`,
    '',
    'CONTEXT:',
    context || 'Không có context đủ tin cậy.',
    '',
    'TASK:',
    'Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm.',
    'Chỉ dùng CONTEXT ở trên.',
    'Mỗi nhận định lịch sử cụ thể phải có citation marker hợp lệ như [1] ngay trong câu hoặc bullet.',
    'Không dùng marker không xuất hiện trong CONTEXT.',
    'Không tự thêm ngày, số liệu, nhân vật hoặc nguồn ngoài context.',
    'Nếu context không đủ, nói rõ là chưa đủ nguồn trong corpus.',
  ].join('\n');
}

async function generateWithNineRouter(message: string, retrievalPayload: any, config9Router: NineRouterConfig): Promise<AnswerProviderResult> {
  const started = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config9Router.timeoutMs);
  try {
    const response = await fetch(`${config9Router.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${config9Router.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: config9Router.model,
        messages: [
          {
            role: 'system',
            content: 'Bạn là trợ lý RAG lịch sử Việt Nam. Chỉ trả lời dựa trên CONTEXT được cung cấp. Không tự thêm nguồn. Không dùng kiến thức ngoài. Không tra web. Nếu context không đủ, nói không đủ nguồn. Trả lời tiếng Việt, ngắn gọn, có citation markers như [1], [2] đúng với context.',
          },
          {
            role: 'user',
            content: buildNineRouterTask(message, retrievalPayload),
          },
        ],
        temperature: config9Router.temperature,
        max_tokens: config9Router.maxTokens,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`9Router provider error ${response.status}: ${body.slice(0, 240)}`);
    }
    const data: any = await response.json();
    const answer = data?.choices?.[0]?.message?.content;
    if (!answer || typeof answer !== 'string') {
      throw new Error('9Router provider returned no message content');
    }
    return {
      answer: answer.trim(),
      provider: '9router_api',
      model: config9Router.model,
      latency_ms: Date.now() - started,
      api_cloud_calls: 1,
      raw_provider_response_redacted: null,
    };
  } finally {
    clearTimeout(timer);
  }
}

function postcheckApiAnswer(answer: string, citations: any[], answerable: boolean): string[] {
  const issues: string[] = [];
  if (API_FAST_METADATA_LEAK_RE.test(answer || '')) issues.push('metadata_leakage');
  const invalidMarkers = findInvalidCitationMarkers(answer || '', citations);
  if (invalidMarkers.length) issues.push(`fake_marker:${invalidMarkers.join(',')}`);
  if (answerable && citations.length > 0 && !(answer || '').match(/\[[0-9]+\]/)) issues.push('missing_inline_citation');
  if (/according to my knowledge|as an ai|theo kiến thức của tôi/i.test(answer || '')) issues.push('outside_knowledge_language');
  return issues;
}

function createEmptyDataset(): LoadedDataset {
  return {
    events: new Map(),
    synthesis: new Map(),
    disambiguationRules: new Map(),
    canonicalEvents: [],
    canonicalSynthesis: [],
    canonicalDisambiguationRules: [],
    allCanonicalDocs: [],
    eventEntityLinks: [],
    synthesisEventLinks: [],
    qaBenchmark: [],
    retrievalQueries: [],
    hardNegatives: [],
    sources: new Map(),
    runtimeLinks: [],
    linksByFromDocId: new Map(),
    linksByToSourceId: new Map(),
    entityToEvents: new Map(),
    synthesisToEvents: new Map(),
    eventToSynthesis: new Map(),
  };
}


async function main() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  Vietnamese Historical RAG Chatbot — Server');
  console.log('═══════════════════════════════════════════════════\n');

  validateConfig();

  // Initialize dataset and indexes
  console.log('📦 Initializing...\n');
  let dataset: LoadedDataset;
  try {
    dataset = loadDataset();
  } catch (error) {
    console.warn('⚠️  Legacy dataset pack not found. Starting Express in persistent-service-only mode.');
    console.warn(`   ${(error as Error).message}`);
    dataset = createEmptyDataset();
  }

  // Build SEPARATE BM25 indexes for Event and Synthesis lanes
  const eventBM25 = new BM25Index(new BaselineTokenizer());
  eventBM25.indexDocuments(dataset.canonicalEvents);

  const synthesisBM25 = new BM25Index(new BaselineTokenizer());
  synthesisBM25.indexDocuments(dataset.canonicalSynthesis);

  const rulesBM25 = new BM25Index(new BaselineTokenizer());
  rulesBM25.indexDocuments(dataset.canonicalDisambiguationRules);

  // Try loading vector caches (PATCH 2)
  let eventVectorStore: InMemoryVectorStore | null = null;
  let synthesisVectorStore: InMemoryVectorStore | null = null;

  console.log('\n🔄 Loading vector caches...');

  const eventVS = new InMemoryVectorStore();
  if (loadEmbeddingCacheToVectorStore(config.eventEmbeddingCachePath, eventVS)) {
    eventVectorStore = eventVS;
    console.log(`   📗 Event vectors: ${eventVS.size()} loaded`);
  } else {
    console.log('   📗 Event vectors: cache not found (BM25-only for event lane)');
  }

  const synthesisVS = new InMemoryVectorStore();
  if (loadEmbeddingCacheToVectorStore(config.synthesisEmbeddingCachePath, synthesisVS)) {
    synthesisVectorStore = synthesisVS;
    console.log(`   📘 Synthesis vectors: ${synthesisVS.size()} loaded`);
  } else {
    console.log('   📘 Synthesis vectors: cache not found (BM25-only for synthesis lane)');
  }

  const bothHybrid = eventVectorStore !== null && synthesisVectorStore !== null;
  const anyHybrid = eventVectorStore !== null || synthesisVectorStore !== null;
  const useHybrid = anyHybrid;
  const modeLabel = bothHybrid
    ? 'hybrid-dual-index'
    : anyHybrid
      ? 'hybrid-dual-index-partial'
      : 'bm25-only-dual-index';

  const pipelineConfig: ChatPipelineConfig = {
    dataset,
    eventBM25,
    synthesisBM25,
    rulesBM25,
    eventVectorStore,
    synthesisVectorStore,
    useHybrid,
  };

  // BM25-only config for /api/chat free-form endpoint
  // Hybrid for free-form requires runtime embeddings (future 10E-lite)
  // Using hybrid here would call embedQuery() → OpenAI API → cost violation
  const bm25OnlyConfig: ChatPipelineConfig = {
    dataset,
    eventBM25,
    synthesisBM25,
    rulesBM25,
    eventVectorStore: null,
    synthesisVectorStore: null,
    useHybrid: false,
  };

  // Create Express app
  const app = express();
  app.use(cors());
  app.use(express.json());

  // Serve static files (UI)
  app.use(express.static(resolve(__dirname, '../../public')));

  // Serve demo questions as static (for frontend to fetch)
  app.use('/demo', express.static(resolve(__dirname, '../../demo')));

  // ─── Load vector caches for hybrid demo (10B cache) ──────
  let demoCacheLoaded = false;
  let demoEventStore: InMemoryVectorStore | null = null;
  let demoSynthesisStore: InMemoryVectorStore | null = null;
  try {
    const { loadVectorCaches } = await import('../retrieval/vector-cache-loader.js');
    const vc = await loadVectorCaches();
    if (vc.cacheLoaded && vc.eventStore && vc.synthesisStore) {
      demoEventStore = vc.eventStore;
      demoSynthesisStore = vc.synthesisStore;
      demoCacheLoaded = true;
      console.log(`   🔀 Demo hybrid cache: event=${vc.eventSize} synthesis=${vc.synthesisSize}`);
    }
  } catch { console.log('   ⚠️ Demo hybrid cache: not loaded (hybrid demo unavailable)'); }

  // ─── API Routes ──────────────────────────────────────────

  /** Health check */
  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'ok',
      mode: modeLabel,
      indexed_docs: {
        events: eventBM25.size(),
        synthesis: synthesisBM25.size(),
        disambiguation_rules: rulesBM25.size(),
        total: eventBM25.size() + synthesisBM25.size() + rulesBM25.size(),
      },
      sources: dataset.sources.size,
      links_provenance: dataset.runtimeLinks.length,
      vector: {
        event_loaded: eventVectorStore !== null,
        synthesis_loaded: synthesisVectorStore !== null,
        event_vectors: eventVectorStore?.size() ?? 0,
        synthesis_vectors: synthesisVectorStore?.size() ?? 0,
        event_cache_path: config.eventEmbeddingCachePath,
        synthesis_cache_path: config.synthesisEmbeddingCachePath,
      },
      demo_hybrid_cache: demoCacheLoaded,
    });
  });

  /** Local web-demo health endpoint — no external network calls. */
  app.get('/api/web-demo-health', async (_req, res) => {
    const ragService = await getRagServiceHealth();
    const serviceHybridReady =
      ragService?.enabled === true &&
      ragService?.health?.ready === true &&
      ragService?.ready?.loaded?.stage20g2_candidate_profile_ready === true &&
      ragService?.ready?.loaded?.stage20g2_candidate_faiss === true;
    const serviceIndexedDocs = Number(ragService?.ready?.counts?.stage20g2_candidate_records || 0);
    res.json({
      backend: 'ok',
      ollama: await checkLocalOllama(),
      model: 'qwen2.5:3b-instruct',
      retrieval: serviceHybridReady || demoCacheLoaded ? 'ok' : 'unknown',
      faiss: serviceHybridReady || demoCacheLoaded ? 'ok' : 'unknown',
      bm25: 'ok',
      no_cloud: true,
      cloud_api_calls: 0,
      cloud_llm_calls: 0,
      cloud_embedding_calls: 0,
      external_network_calls: 0,
      mode: serviceHybridReady ? 'persistent-final-local-hybrid' : modeLabel,
      persistent_rag_service: ragService,
      demo_hybrid_cache: demoCacheLoaded,
      local_hybrid_endpoint: 'ok',
      api_fast_mode: {
        endpoint: '/api/9router-fast-chat',
        provider: '9Router',
        configured: getNineRouterConfig().configured,
        retrieval_provider: process.env.RAG_API_RETRIEVAL_PROVIDER || 'local',
        embedding_provider: 'local_sentence_transformer',
        vector_index_provider: `${process.env.RAG_API_DATA_PROFILE || 'cloud_primary_final'}_local_faiss`,
        cloud_embedding_default_disabled: true,
        cloud_embedding_experimental_available: Boolean(process.env['9ROUTER_EMBEDDING_MODEL']),
        no_cloud: false,
      },
      indexed_docs: serviceHybridReady && serviceIndexedDocs > 0
        ? serviceIndexedDocs
        : eventBM25.size() + synthesisBM25.size() + rulesBM25.size(),
    });
  });

  /** Local hybrid web chat — runtime query embedding + FAISS + RRF, no cloud. */
  app.post('/api/local-hybrid-chat', async (req, res) => {
    try {
      const message = req.body?.message || req.body?.question;
      if (!message || typeof message !== 'string') {
        res.status(400).json({ error: 'Missing "message" field' });
        return;
      }
      const sessionId = String(req.body?.session_id || 'web-demo');
      const session = getWebChatSession(sessionId);
      const dataProfile = String(req.body?.data_profile || process.env.RAG_DATA_PROFILE || 'cloud_primary_final');
      const servicePayload = {
        message,
        session_id: sessionId,
        demo_mode: req.body?.demo_mode ?? true,
        return_debug: req.body?.return_debug ?? true,
        profile_latency: req.body?.profile_latency ?? false,
        capture_answer: req.body?.capture_answer ?? false,
        data_profile: dataProfile,
        runtime_mode: 'local_no_cloud',
        force_local_hybrid: true,
        previous_query: session.lastQuery ?? null,
        previous_focus: session.lastFocus ?? null,
      };
      if (ragServiceEnabled()) {
        try {
          const serviceResponse = await callRagService('/local-hybrid-chat', servicePayload);
          const debug = serviceResponse.data?.debug || {};
          if (serviceResponse.data?.status?.answerable && debug.safety_mode === 'none') {
            updateWebChatSession(sessionId, {
              lastQuery: message,
              lastFocus: debug.normalized_query || debug.rewritten_query || message,
            });
          }
          res.status(serviceResponse.status).json(serviceResponse.data);
          return;
        } catch (error) {
          if (!ragCliFallbackAllowed()) {
            res.status(503).json({
              error: 'Persistent RAG runtime service unavailable',
              message: 'Persistent RAG runtime service is unavailable. Start service with npm run rag:service.',
              service_url: RAG_SERVICE_URL,
              fallback_allowed: false,
              detail: (error as Error).message,
              status: { answerable: false, safe: false, no_cloud: true, hybrid_complete: false },
              debug: {
                served_by: 'service_unavailable',
                fallback_used: false,
                cloud_api_calls: 0,
                cloud_embedding_calls: 0,
                cloud_llm_calls: 0,
              },
            });
            return;
          }
        }
      }
      if (
        dataProfile === 'stage15d_candidate' ||
        dataProfile === 'stage15g_candidate' ||
        dataProfile === 'stage17b_candidate' ||
        dataProfile === 'stage18b2_candidate' ||
        dataProfile === 'stage19b3_candidate' ||
        dataProfile === 'stage20b_candidate' ||
        dataProfile === 'stage20d_candidate' ||
        dataProfile === 'stage20d2_candidate' ||
        dataProfile === 'stage20d3_candidate' ||
        dataProfile === 'stage20f0_local_style_candidate' ||
        dataProfile === 'stage20f1_local_style_candidate' ||
        dataProfile === 'stage20g2_candidate' || dataProfile === 'cloud_primary_final'
      ) {
        res.status(503).json({
          error: 'Candidate data profile requires persistent service',
          message: `data_profile=${dataProfile} chỉ chạy qua persistent RAG service. Hãy chạy npm run rag:service rồi thử lại.`,
          service_url: RAG_SERVICE_URL,
          fallback_allowed: false,
          status: { answerable: false, safe: false, no_cloud: true, hybrid_complete: false },
          debug: {
            served_by: 'service_unavailable',
            data_profile: dataProfile,
            candidate_profile: true,
            fallback_used: false,
            cloud_api_calls: 0,
            cloud_embedding_calls: 0,
            cloud_llm_calls: 0,
          },
        });
        return;
      }
      const response = await runLocalHybridWebChat({
        message,
        session_id: sessionId,
        demo_mode: req.body?.demo_mode ?? true,
        return_debug: req.body?.return_debug ?? true,
        data_profile: dataProfile,
        force_local_hybrid: true,
        previous_query: session.lastQuery ?? null,
        previous_focus: session.lastFocus ?? null,
      });

      const debug = response?.debug || {};
      if (response?.status?.answerable && debug.safety_mode === 'none') {
        updateWebChatSession(sessionId, {
          lastQuery: message,
          lastFocus: debug.normalized_query || debug.rewritten_query || message,
        });
      }

      res.json(response);
    } catch (error) {
      console.error('❌ Local hybrid web chat error:', error);
      res.status(500).json({
        error: 'Local hybrid runtime error',
        message: (error as Error).message,
        status: { answerable: false, safe: false, no_cloud: true, hybrid_complete: false },
        debug: {
          retrieval_mode: 'runtime_error',
          bm25_fallback: false,
          fallback_reason: (error as Error).message,
          cloud_api_calls: 0,
          cloud_embedding_calls: 0,
          external_network_calls: 0,
        },
      });
    }
  });

  /** Optional 9Router API-fast chat — BM25 + cloud embedding/FAISS retrieval, API answer generation. */
  app.post('/api/9router-fast-chat', async (req, res) => {
    const totalStarted = Date.now();
    try {
      const message = req.body?.message || req.body?.question;
      if (!message || typeof message !== 'string') {
        res.status(400).json({ error: 'Missing "message" field' });
        return;
      }

      const sessionId = String(req.body?.session_id || 'web-demo-api-fast');
      const session = getWebChatSession(sessionId);
      const dataProfile = String(req.body?.data_profile || process.env.RAG_API_DATA_PROFILE || 'cloud_primary_final');
      const apiRetrievalProvider = String(req.body?.retrieval_provider || process.env.RAG_API_RETRIEVAL_PROVIDER || 'local');
      const servicePayload = {
        message,
        session_id: sessionId,
        demo_mode: req.body?.demo_mode ?? true,
        return_debug: req.body?.return_debug ?? true,
        profile_latency: req.body?.profile_latency ?? false,
        capture_answer: req.body?.capture_answer ?? false,
        data_profile: dataProfile,
        runtime_mode: 'api_9router_fast',
        retrieval_provider: apiRetrievalProvider,
        force_cloud_embedding: req.body?.force_cloud_embedding === true,
        force_cloud_llm_final: req.body?.force_cloud_llm_final === true,
        return_generation_payload: true,
        previous_query: session.lastQuery ?? null,
        previous_focus: session.lastFocus ?? null,
      };
      if (ragServiceEnabled()) {
        try {
          const serviceResponse = await callRagService('/9router-fast-chat', servicePayload);
          const debug = serviceResponse.data?.debug || {};
          if (serviceResponse.data?.status?.answerable && debug.safety_mode === 'none') {
            updateWebChatSession(sessionId, {
              lastQuery: message,
              lastFocus: debug.normalized_query || debug.rewritten_query || message,
            });
          }
          res.status(serviceResponse.status).json(serviceResponse.data);
          return;
        } catch (error) {
          if (!ragCliFallbackAllowed()) {
            res.status(503).json({
              error: 'Persistent RAG runtime service unavailable',
              message: 'Persistent RAG runtime service is unavailable. Start service with npm run rag:service.',
              service_url: RAG_SERVICE_URL,
              fallback_allowed: false,
              detail: (error as Error).message,
              answer: 'Persistent RAG runtime service chưa chạy. Hãy chạy npm run rag:service rồi thử lại.',
              citations: [],
              status: { answerable: false, safe: false, no_cloud: false, api_fast_mode: true },
              debug: {
                served_by: 'service_unavailable',
                fallback_used: false,
                runtime_mode: 'api_9router_fast',
                cloud_api_calls: 0,
                cloud_embedding_calls: 0,
                cloud_llm_calls: 0,
              },
            });
            return;
          }
        }
      }
      if (!['cloud', 'cloud_embedding', '9router_embedding'].includes(apiRetrievalProvider.toLowerCase())) {
        res.status(503).json({
          error: 'Persistent RAG runtime service required',
          message: 'API-fast mode with local retrieval requires the persistent RAG runtime service. Start service with npm run rag:service.',
          service_url: RAG_SERVICE_URL,
          fallback_allowed: false,
          answer: 'API-fast local-retrieval mode cần persistent RAG service. Local no-cloud endpoint vẫn khả dụng nếu service đang chạy.',
          citations: [],
          status: { answerable: false, safe: false, no_cloud: false, api_fast_mode: true, retrieval_local: true },
          debug: {
            served_by: 'service_unavailable',
            fallback_used: false,
            runtime_mode: 'api_9router_fast',
            retrieval_provider: apiRetrievalProvider,
            embedding_provider: 'local_sentence_transformer',
            vector_index_provider: `${process.env.RAG_API_DATA_PROFILE || 'cloud_primary_final'}_local_faiss`,
            cloud_api_calls: 0,
            cloud_embedding_calls: 0,
            cloud_llm_calls: 0,
          },
        });
        return;
      }
      const retrievalStarted = Date.now();
      const retrievalPayload = await runNineRouterCloudRetrieval({
        message,
        session_id: sessionId,
        demo_mode: req.body?.demo_mode ?? true,
        return_debug: req.body?.return_debug ?? true,
        runtime_mode: 'api_9router_fast',
        force_cloud_embedding: true,
        return_generation_payload: true,
        previous_query: session.lastQuery ?? null,
        previous_focus: session.lastFocus ?? null,
      });
      const retrievalLatencyMs = Date.now() - retrievalStarted;

      const retrievalDebug = retrievalPayload?.debug || {};
      const citations = retrievalPayload?.citations || [];
      const answerable = retrievalPayload?.status?.answerable === true && retrievalDebug.safety_mode === 'none';
      const cloudEmbeddingCalls = Number(retrievalDebug.cloud_embedding_calls || 0);
      const baseDebug = {
        ...retrievalDebug,
        runtime_mode: 'api_9router_fast',
        answer_generator: answerable ? '9router_api' : 'deterministic_safety_or_insufficient_evidence',
        provider_base_url: getNineRouterConfig().baseUrl,
        provider_model: getNineRouterConfig().model || 'not_configured',
        cloud_api_calls: cloudEmbeddingCalls,
        cloud_llm_calls: 0,
        cloud_embedding_calls: cloudEmbeddingCalls,
        external_network_calls: 0,
        local_retrieval: false,
        local_bm25_retrieval: retrievalDebug.local_bm25_retrieval ?? true,
        cloud_vector_retrieval: retrievalDebug.cloud_vector_retrieval ?? (cloudEmbeddingCalls > 0),
        api_used_for_answer_generation_only: false,
        api_used_for_embedding_and_answer_generation: answerable,
        no_cloud_mode: false,
        retrieval_latency_ms: retrievalLatencyMs,
        generation_latency_ms: 0,
        latency_ms: Date.now() - totalStarted,
        adapter_endpoint: '/api/9router-fast-chat',
      };

      if (!answerable) {
        res.json({
          answer: retrievalPayload?.answer || '9Router API mode không gọi provider vì context không đủ hoặc câu hỏi nằm ngoài phạm vi.',
          citations,
          debug: baseDebug,
          status: {
            answerable: false,
            safe: true,
            no_cloud: false,
            api_fast_mode: true,
            retrieval_local: false,
            bm25_local: true,
            cloud_embedding_retrieval: cloudEmbeddingCalls > 0,
          },
        });
        return;
      }

      if (!citations.length) {
        res.json({
          answer: 'Mình chưa tìm thấy nguồn đủ trực tiếp trong corpus để gọi 9Router API mode an toàn. Local no-cloud mode vẫn khả dụng.',
          citations: [],
          debug: {
            ...baseDebug,
            answer_generator: 'insufficient_context_guard',
            api_used_for_answer_generation_only: false,
            context_only_guard_issues: ['empty_citation_payload'],
          },
          status: {
            answerable: false,
            safe: true,
            no_cloud: false,
            api_fast_mode: true,
            retrieval_local: false,
            bm25_local: true,
            cloud_embedding_retrieval: cloudEmbeddingCalls > 0,
          },
        });
        return;
      }

      const config9Router = getNineRouterConfig();
      if (!config9Router.configured) {
        res.status(503).json({
          error: '9Router API mode is not configured',
          message: '9Router API mode is not configured. Local no-cloud mode is still available.',
          answer: '9Router API mode chưa được cấu hình. Bạn vẫn có thể dùng chế độ local/no-cloud qua /api/local-hybrid-chat.',
          citations,
          debug: {
            ...baseDebug,
            provider_configured: false,
            missing_config: config9Router.missing,
            answer_generator: '9router_api_not_configured',
          },
          status: {
            answerable: false,
            safe: true,
            no_cloud: false,
            api_fast_mode: true,
            retrieval_local: false,
            bm25_local: true,
            cloud_embedding_retrieval: cloudEmbeddingCalls > 0,
          },
        });
        return;
      }

      const providerResult = await generateWithNineRouter(message, retrievalPayload, config9Router);
      const postcheckIssues = postcheckApiAnswer(providerResult.answer, citations, true);
      if (postcheckIssues.length) {
        res.status(502).json({
          error: '9Router context-only postcheck failed',
          message: '9Router answer failed local citation/context guard.',
          answer: '9Router API mode trả về câu trả lời không đạt kiểm tra citation/context-only, nên hệ thống không hiển thị câu trả lời đó. Local no-cloud mode vẫn khả dụng.',
          citations,
          debug: {
            ...baseDebug,
            answer_generator: '9router_api',
            provider_configured: true,
            cloud_api_calls: cloudEmbeddingCalls + providerResult.api_cloud_calls,
            cloud_llm_calls: providerResult.api_cloud_calls,
            cloud_embedding_calls: cloudEmbeddingCalls,
            generation_latency_ms: providerResult.latency_ms,
            latency_ms: Date.now() - totalStarted,
            context_only_guard_issues: postcheckIssues,
          },
          status: {
            answerable: false,
            safe: false,
            no_cloud: false,
            api_fast_mode: true,
            retrieval_local: false,
            bm25_local: true,
            cloud_embedding_retrieval: cloudEmbeddingCalls > 0,
          },
        });
        return;
      }

      if (retrievalPayload?.status?.answerable && retrievalDebug.safety_mode === 'none') {
        updateWebChatSession(sessionId, {
          lastQuery: message,
          lastFocus: retrievalDebug.normalized_query || retrievalDebug.rewritten_query || message,
        });
      }

      res.json({
        answer: providerResult.answer,
        citations,
        debug: {
          ...baseDebug,
          answer_generator: providerResult.provider,
          provider_configured: true,
          provider_model: providerResult.model,
          cloud_api_calls: cloudEmbeddingCalls + providerResult.api_cloud_calls,
          cloud_llm_calls: providerResult.api_cloud_calls,
          cloud_embedding_calls: cloudEmbeddingCalls,
          generation_latency_ms: providerResult.latency_ms,
          latency_ms: Date.now() - totalStarted,
          context_only_guard_issues: [],
        },
        status: {
          answerable: true,
          safe: true,
          no_cloud: false,
          api_fast_mode: true,
          retrieval_local: false,
          bm25_local: true,
          cloud_embedding_retrieval: cloudEmbeddingCalls > 0,
        },
      });
    } catch (error) {
      console.error('❌ 9Router API-fast chat error:', (error as Error).message);
      res.status(500).json({
        error: '9Router API-fast runtime error',
        message: (error as Error).message,
        answer: '9Router API-fast mode gặp lỗi runtime. Local no-cloud mode vẫn khả dụng.',
        citations: [],
        status: {
          answerable: false,
          safe: false,
          no_cloud: false,
          api_fast_mode: true,
          retrieval_local: true,
        },
        debug: {
          runtime_mode: 'api_9router_fast',
          retrieval_mode: 'runtime_error',
          answer_generator: '9router_api',
          cloud_api_calls: 0,
          cloud_llm_calls: 0,
          cloud_embedding_calls: 0,
          external_network_calls: 0,
          api_used_for_answer_generation_only: true,
          no_cloud_mode: false,
          latency_ms: Date.now() - totalStarted,
          fallback_reason: (error as Error).message,
        },
      });
    }
  });

  /** Dataset statistics */
  app.get('/api/stats', (_req, res) => {
    res.json({
      events: dataset.canonicalEvents.length,
      synthesis: dataset.canonicalSynthesis.length,
      disambiguation_rules: dataset.canonicalDisambiguationRules.length,
      total_indexed: eventBM25.size() + synthesisBM25.size() + rulesBM25.size(),
      sources: dataset.sources.size,
      links_provenance: dataset.runtimeLinks.length,
      entity_links: dataset.eventEntityLinks.length,
      synthesis_links: dataset.synthesisEventLinks.length,
      qa_benchmark: dataset.qaBenchmark.length,
      bm25_terms: {
        events: eventBM25.vocabularySize(),
        synthesis: synthesisBM25.vocabularySize(),
        disambiguation_rules: rulesBM25.vocabularySize(),
      },
      vector: {
        event_loaded: eventVectorStore !== null,
        synthesis_loaded: synthesisVectorStore !== null,
        event_vectors: eventVectorStore?.size() ?? 0,
        synthesis_vectors: synthesisVectorStore?.size() ?? 0,
        event_cache_path: config.eventEmbeddingCachePath,
        synthesis_cache_path: config.synthesisEmbeddingCachePath,
      },
      mode: modeLabel,
    });
  });

  /** Chat endpoint — supports mode selection (bm25/hybrid/vector) */
  app.post('/api/chat', async (req, res) => {
    try {
      const { question, retrieval_mode } = req.body;
      if (!question || typeof question !== 'string') {
        res.status(400).json({ error: 'Missing "question" field' });
        return;
      }
      const sessionId = String(req.body?.session_id || 'api-chat-default');
      const session = getWebChatSession(sessionId);

      // Select pipeline config based on retrieval_mode from frontend
      let cfg: ChatPipelineConfig;
      let modeUsed: string;
      if (retrieval_mode === 'hybrid' && useHybrid) {
        cfg = pipelineConfig; // hybrid uses API for embedQuery()
        modeUsed = 'hybrid_live';
      } else if (retrieval_mode === 'vector' && useHybrid) {
        // Vector-only: use pipeline but with vector-heavy weight
        cfg = pipelineConfig;
        modeUsed = 'vector_live';
      } else {
        cfg = bm25OnlyConfig;
        modeUsed = 'bm25_fallback';
      }

      console.log(`\n📨 Chat: mode=${modeUsed} q="${question.substring(0, 50)}..."`);
      const response = await processChat(question, cfg, {
        session_id: sessionId,
        session_memory: session.sessionMemory ?? null,
      });
      rememberProcessChatTurn(sessionId, question, response);
      const mode = detectResponseMode(response);
      res.json({ ...response, mode: modeUsed === 'bm25_fallback' ? mode : modeUsed });
    } catch (error) {
      console.error('❌ Chat error:', error);
      res.status(500).json({
        error: 'Internal server error',
        message: (error as Error).message,
      });
    }
  });

  /** Demo chat endpoint — uses LOCAL_DEMO-R hybrid E2E path */
  app.post('/api/demo-chat', async (req, res) => {
    try {
      const { question, demo_id, debug: showDebug } = req.body;
      if (!question || typeof question !== 'string') {
        res.status(400).json({ error: 'Missing "question" field' });
        return;
      }
      const sessionId = String(req.body?.session_id || 'demo-chat-default');
      const session = getWebChatSession(sessionId);

      // Load demo map to find cache key
      const { readFileSync, existsSync } = await import('fs');
      const mapPath = resolve(__dirname, '../../demo/demo_query_map.json');
      let queryCacheKey: string | null = null;
      if (demo_id && existsSync(mapPath)) {
        const map = JSON.parse(readFileSync(mapPath, 'utf8'));
        queryCacheKey = map.mapping?.[demo_id]?.query_cache_key ?? null;
      }

      // Try hybrid path
      if (queryCacheKey && demoCacheLoaded && demoEventStore && demoSynthesisStore) {
        const { lookupByQueryId, resetQueryCache } = await import('../retrieval/query-embedding-cache.js');
        const { rrfFusionHardened } = await import('../retrieval/rrf-fusion.js');
        const { routeQuery } = await import('../routing/query-router.js');
        const { rerankResults } = await import('../reranking/reranker.js');
        const { selectEvidence } = await import('../evidence/evidence-selector.js');
        const { buildContextBundle } = await import('../context/context-builder.js');
        const { generateAnswer } = await import('../generation/answer-generator.js');
        const { applyHardNegativeGuard } = await import('../reranking/hard-negative-guard.js');
        type RRFLane = import('../retrieval/rrf-fusion.js').RRFLane;

        resetQueryCache();
        const lookup = lookupByQueryId(queryCacheKey);
        if (lookup.found && lookup.embedding) {
          console.log(`\n🔀 Demo hybrid: ${demo_id} → ${queryCacheKey}`);
          const TOP_K = 10, LANE_K = 30, RRF_K = 60;
          const bm25Evt = eventBM25.search(question, LANE_K, { canonical_only: true });
          const bm25Syn = synthesisBM25.search(question, LANE_K, { canonical_only: true });
          const vecEvt = await demoEventStore.search(lookup.embedding, LANE_K);
          const vecSyn = await demoSynthesisStore.search(lookup.embedding, LANE_K);

          const makeLane = (name: RRFLane['name'], r: any[], t: 'event' | 'synthesis'): RRFLane => ({
            name,
            results: r.slice(0, LANE_K).map((x: any) => ({
              doc_id: x.doc_id, score: x.score,
              title: (x.metadata as any)?.title || x.doc_id,
              source_type: t, metadata: x.metadata as unknown as Record<string, unknown>,
            })),
          });
          const lanes = [
            makeLane('bm25_event', bm25Evt, 'event'),
            makeLane('bm25_synthesis', bm25Syn, 'synthesis'),
            makeLane('vector_event', vecEvt, 'event'),
            makeLane('vector_synthesis', vecSyn, 'synthesis'),
          ];
          const rrfResults = rrfFusionHardened(lanes, question, {
            k: RRF_K, topK: TOP_K, preserveBm25EventTopN: 10,
            eventPrimaryBoost: 0.003, eventPrimaryBoostEnabled: true,
          });
          const hybridResults = rrfResults.map(r => ({
            doc_id: r.doc_id,
            vector_score: r.raw_scores.vector ?? 0,
            bm25_score: r.raw_scores.bm25 ?? 0,
            combined_score: r.adjusted_score,
            metadata: r.metadata as any,
          }));

          // Pipeline steps
          const routing = await routeQuery(question);
          const reranked = await rerankResults(question, hybridResults, dataset);
          const guarded = applyHardNegativeGuard(question, reranked, dataset);
          const evidence = selectEvidence(guarded, dataset, routing.query_frame, question);
          const filtered = guarded.filter(c =>
            [...evidence.primary, ...evidence.supporting, ...evidence.contrast].find(e => e.doc_id === c.doc_id)
          );
          for (const c of filtered) {
            const p = evidence.primary.find(e => e.doc_id === c.doc_id);
            if (p) { c.evidence_role = 'primary'; c.evidence_role_score = p.role_score; c.evidence_reasons = p.reasons.map(r => r.code); c.evidence_rank = 0; }
            else { c.evidence_role = 'supporting'; c.evidence_role_score = 0; c.evidence_reasons = []; c.evidence_rank = 1; }
          }
          const ctx = buildContextBundle(routing.intent, filtered, dataset, routing.query_frame, { eventBM25, synthesisBM25 });
          const response = await generateAnswer(question, ctx, hybridResults, dataset, routing.query_frame);

          // Strip internal IDs from user-facing fields unless debug
          const sanitized = showDebug ? response : sanitizeResponse(response);
          updateWebChatSession(sessionId, {
            lastQuery: question,
            lastFocus: question,
          });
          res.json({
            ...sanitized,
            mode: 'hybrid_cached',
            debug: showDebug ? {
              query_cache_key: queryCacheKey,
              retrieval_mode: 'hybrid_rrf_hardened',
              target_rank: null,
              api_attempts: 0,
              routing: { intent: routing.intent },
              retrieval_doc_ids: hybridResults.map(r => r.doc_id).slice(0, 5),
            } : undefined,
          });
          return;
        }
      }

      // Fallback to BM25
      const response = await processChat(question, bm25OnlyConfig, {
        session_id: sessionId,
        session_memory: session.sessionMemory ?? null,
      });
      rememberProcessChatTurn(sessionId, question, response);
      const mode = detectResponseMode(response);
      res.json({
        ...response,
        mode,
        warnings: ['Câu hỏi này chưa có query embedding cache. Hệ thống dùng BM25 fallback.'],
      });
    } catch (error) {
      console.error('❌ Demo chat error:', error);
      res.status(500).json({ error: 'Internal server error', message: (error as Error).message });
    }
  });

  // ─── Start Server ────────────────────────────────────────

  app.listen(config.port, () => {
    console.log(`\n🚀 Server running at http://localhost:${config.port}`);
    console.log(`   Mode: ${modeLabel}`);
    console.log(`   Event index: ${eventBM25.size()} docs, ${eventBM25.vocabularySize()} terms${eventVectorStore ? `, ${eventVectorStore.size()} vectors` : ''}`);
    console.log(`   Synthesis index: ${synthesisBM25.size()} docs, ${synthesisBM25.vocabularySize()} terms${synthesisVectorStore ? `, ${synthesisVectorStore.size()} vectors` : ''}`);
    console.log(`   Rules index: ${rulesBM25.size()} docs, ${rulesBM25.vocabularySize()} terms`);
    console.log(`   Links provenance: ${dataset.runtimeLinks.length} links`);
    console.log(`   Demo hybrid: ${demoCacheLoaded ? 'READY' : 'NOT AVAILABLE'}`);
    console.log(`   UI: http://localhost:${config.port}/\n`);
  });
}

main().catch(err => {
  console.error('❌ Server failed to start:', err);
  process.exit(1);
});

