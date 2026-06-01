/**
 * Query Monitoring — PATCH 9I
 *
 * Structured technical logging for demo/product analysis.
 * No secrets, no unnecessary personal data.
 *
 * Env flags:
 * - RAG_MONITORING_LOG_RAW_QUERY=0/1 (default 0)
 * - RAG_MONITORING_ENABLED=0/1 (default 1)
 */

import { createHash } from 'crypto';
import { appendFileSync, mkdirSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { CapabilityBucket, AnswerPolicy, EvidenceQuality, UpgradeSignal } from '../policy/capability-types.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ─── Monitoring Record ───────────────────────────────────────

export interface MonitoringRecord {
  timestamp: string;
  queryHash: string;
  queryText?: string;
  routeIntent: string;
  capabilityBucket: CapabilityBucket;
  answerPolicy: AnswerPolicy;
  evidenceQuality: EvidenceQuality;
  citationCount: number;
  confidence: string;
  upgradeSignals: UpgradeSignal[];
  outcomeTags: string[];
  apiAttempted: boolean;
}

// ─── Outcome Tags Builder ────────────────────────────────────

export function buildOutcomeTags(
  bucket: CapabilityBucket,
  policy: AnswerPolicy,
  upgradeSignals: UpgradeSignal[],
): string[] {
  const tags: string[] = [];

  // Primary outcome tag
  switch (policy) {
    case 'FULL_ANSWER': tags.push('full_answer'); break;
    case 'FULL_ANSWER_OR_CAUTION': tags.push('full_answer'); break;
    case 'HONEST_PARTIAL': tags.push('honest_partial'); break;
    case 'LOW_EVIDENCE_CAUTION': tags.push('low_evidence'); break;
    case 'ASK_CLARIFICATION': tags.push('clarification'); break;
    case 'REFUSE_OOS': tags.push('oos'); break;
    case 'CORPUS_GAP_NOTICE': tags.push('corpus_gap'); break;
  }

  // Bucket-specific tags
  if (bucket === 'RETRIEVAL_WEAK') tags.push('retrieval_weak');
  if (bucket === 'CORPUS_GAP') tags.push('corpus_gap');

  // Upgrade signal tags
  for (const sig of upgradeSignals) {
    switch (sig) {
      case 'needs_vector_retrieval': tags.push('vector_needed'); break;
      case 'needs_llm_synthesis': tags.push('llm_needed'); break;
      case 'needs_corpus_expansion': tags.push('corpus_needed'); break;
      case 'needs_alias_expansion': tags.push('alias_needed'); break;
      case 'needs_structured_role_relation': tags.push('structured_role_needed'); break;
      case 'needs_side_specific_retrieval': tags.push('vector_needed'); break;
    }
  }

  return [...new Set(tags)];
}

// ─── Logger ──────────────────────────────────────────────────

/** In-memory buffer for batch access by diagnostic scripts */
const monitoringBuffer: MonitoringRecord[] = [];

export function getMonitoringBuffer(): MonitoringRecord[] {
  return monitoringBuffer;
}

export function clearMonitoringBuffer(): void {
  monitoringBuffer.length = 0;
}

export function recordOutcome(record: MonitoringRecord): void {
  // Buffer for in-process access
  monitoringBuffer.push(record);

  // Check if monitoring is enabled
  const enabled = process.env.RAG_MONITORING_ENABLED !== '0';
  if (!enabled) return;

  // Build output path
  const rootDir = resolve(__dirname, '..', '..');
  const monitorDir = resolve(rootDir, 'reports', 'monitoring');
  if (!existsSync(monitorDir)) {
    mkdirSync(monitorDir, { recursive: true });
  }

  // Sanitize: remove queryText if not explicitly enabled
  const logRaw = process.env.RAG_MONITORING_LOG_RAW_QUERY === '1';
  const sanitized = { ...record };
  if (!logRaw) {
    delete sanitized.queryText;
  }

  // Append to JSONL
  const logFile = resolve(monitorDir, 'query-outcomes.jsonl');
  try {
    appendFileSync(logFile, JSON.stringify(sanitized) + '\n', 'utf8');
  } catch {
    // Silently fail — monitoring is non-critical
  }
}

/**
 * Create a MonitoringRecord from pipeline context.
 */
export function createMonitoringRecord(opts: {
  query: string;
  routeIntent: string;
  bucket: CapabilityBucket;
  policy: AnswerPolicy;
  evidenceQuality: EvidenceQuality;
  citationCount: number;
  confidence: string;
  upgradeSignals: UpgradeSignal[];
}): MonitoringRecord {
  const queryHash = createHash('sha256').update(opts.query).digest('hex').substring(0, 16);
  const logRaw = process.env.RAG_MONITORING_LOG_RAW_QUERY === '1';

  return {
    timestamp: new Date().toISOString(),
    queryHash,
    queryText: logRaw ? opts.query : undefined,
    routeIntent: opts.routeIntent,
    capabilityBucket: opts.bucket,
    answerPolicy: opts.policy,
    evidenceQuality: opts.evidenceQuality,
    citationCount: opts.citationCount,
    confidence: opts.confidence,
    upgradeSignals: opts.upgradeSignals,
    outcomeTags: buildOutcomeTags(opts.bucket, opts.policy, opts.upgradeSignals),
    apiAttempted: false,
  };
}
