/**
 * RRF Fusion — PATCH 10C-R (hardened)
 * 
 * Reciprocal Rank Fusion with:
 * - BM25 event preservation: ensures BM25 event lane top-N docs survive fusion
 * - Event-primary boost: small bonus for event docs on direct fact/event queries
 * - Lane weight support: configurable per-lane weights
 * 
 * Formula: score(d) = Σ weight_lane × 1/(k + rank_i) + boosts
 * Default k = 60.
 */

export interface RRFCandidate {
  doc_id: string;
  title: string;
  source_type: 'event' | 'synthesis' | 'unknown';
  rrf_score: number;
  adjusted_score: number;
  lanes: {
    bm25_event_rank?: number;
    bm25_synthesis_rank?: number;
    vector_event_rank?: number;
    vector_synthesis_rank?: number;
  };
  raw_scores: {
    bm25?: number;
    vector?: number;
  };
  metadata: Record<string, unknown>;
  // Diagnostics
  protected_by_bm25_event: boolean;
  event_primary_boost_applied: boolean;
  synthesis_guard_applied: boolean;
  applied_boosts: string[];
}

export interface RRFLane {
  name: 'bm25_event' | 'bm25_synthesis' | 'vector_event' | 'vector_synthesis';
  results: Array<{ doc_id: string; score: number; title: string; source_type: 'event' | 'synthesis' | 'unknown'; metadata: Record<string, unknown> }>;
}

export interface RRFHardenedOptions {
  k?: number;
  topK?: number;
  /** Lane weights. Default all 1.0 */
  laneWeights?: {
    bm25_event?: number;
    bm25_synthesis?: number;
    vector_event?: number;
    vector_synthesis?: number;
  };
  /** Preserve BM25 event candidates up to this rank. Default 10 */
  preserveBm25EventTopN?: number;
  /** Small bonus for event docs. Default 0.003 */
  eventPrimaryBoost?: number;
  /** Enable event-primary boost for direct queries. Default true */
  eventPrimaryBoostEnabled?: boolean;
  /** Whether query is a direct event/fact lookup (activates boost). */
  isDirectEventQuery?: boolean;
}

/** Vietnamese patterns indicating direct event/fact queries */
const DIRECT_EVENT_PATTERNS = [
  /sự kiện nào/i, /trận đánh/i, /văn kiện nào/i, /tổ chức nào/i,
  /năm nào/i, /khi nào/i, /ở đâu/i, /\bai\b/i, /hiệp định nào/i,
  /phong trào/i, /chiến dịch/i, /cuộc .+ nào/i, /mở đầu/i,
  /kết thúc/i, /giành chính quyền/i, /thành lập/i,
];

/** Detect if query is a direct event/fact lookup */
export function isDirectEventQuery(query: string): boolean {
  return DIRECT_EVENT_PATTERNS.some(p => p.test(query));
}

/**
 * Hardened RRF fusion with BM25 preservation and event-primary boost.
 */
export function rrfFusionHardened(lanes: RRFLane[], query: string, options?: RRFHardenedOptions): RRFCandidate[] {
  const k = options?.k ?? 60;
  const topK = options?.topK ?? 10;
  const preserveN = options?.preserveBm25EventTopN ?? 10;
  const eventBoost = options?.eventPrimaryBoost ?? 0.003;
  const eventBoostEnabled = options?.eventPrimaryBoostEnabled ?? true;
  const directQuery = options?.isDirectEventQuery ?? isDirectEventQuery(query);
  const weights = {
    bm25_event: options?.laneWeights?.bm25_event ?? 1.0,
    bm25_synthesis: options?.laneWeights?.bm25_synthesis ?? 1.0,
    vector_event: options?.laneWeights?.vector_event ?? 1.0,
    vector_synthesis: options?.laneWeights?.vector_synthesis ?? 1.0,
  };

  const candidateMap = new Map<string, RRFCandidate>();

  // Identify BM25 event protected set
  const bm25EventLane = lanes.find(l => l.name === 'bm25_event');
  const bm25ProtectedIds = new Set<string>();
  if (bm25EventLane) {
    for (let i = 0; i < Math.min(preserveN, bm25EventLane.results.length); i++) {
      bm25ProtectedIds.add(bm25EventLane.results[i].doc_id);
    }
  }

  // Standard RRF accumulation with lane weights
  for (const lane of lanes) {
    const w = weights[lane.name];
    for (let rank = 0; rank < lane.results.length; rank++) {
      const r = lane.results[rank];
      const rrfContrib = w * (1 / (k + rank + 1));

      let candidate = candidateMap.get(r.doc_id);
      if (!candidate) {
        candidate = {
          doc_id: r.doc_id, title: r.title, source_type: r.source_type,
          rrf_score: 0, adjusted_score: 0,
          lanes: {}, raw_scores: {}, metadata: r.metadata,
          protected_by_bm25_event: false, event_primary_boost_applied: false,
          synthesis_guard_applied: false, applied_boosts: [],
        };
        candidateMap.set(r.doc_id, candidate);
      }

      candidate.rrf_score += rrfContrib;

      const laneRank = rank + 1;
      switch (lane.name) {
        case 'bm25_event': candidate.lanes.bm25_event_rank = laneRank; break;
        case 'bm25_synthesis': candidate.lanes.bm25_synthesis_rank = laneRank; break;
        case 'vector_event': candidate.lanes.vector_event_rank = laneRank; break;
        case 'vector_synthesis': candidate.lanes.vector_synthesis_rank = laneRank; break;
      }

      if (lane.name.startsWith('bm25')) {
        candidate.raw_scores.bm25 = Math.max(candidate.raw_scores.bm25 ?? 0, r.score);
      } else {
        candidate.raw_scores.vector = Math.max(candidate.raw_scores.vector ?? 0, r.score);
      }
    }
  }

  // Apply boosts
  for (const candidate of candidateMap.values()) {
    candidate.adjusted_score = candidate.rrf_score;

    // Mark BM25 event protected
    if (bm25ProtectedIds.has(candidate.doc_id) && candidate.lanes.bm25_event_rank !== undefined) {
      candidate.protected_by_bm25_event = true;
    }

    // Event-primary boost for direct queries
    if (directQuery && eventBoostEnabled && candidate.source_type === 'event') {
      candidate.adjusted_score += eventBoost;
      candidate.event_primary_boost_applied = true;
      candidate.applied_boosts.push(`event_primary_boost: +${eventBoost}`);
    }
  }

  // Sort by adjusted score
  const sorted = Array.from(candidateMap.values()).sort((a, b) => b.adjusted_score - a.adjusted_score);

  // BM25 event preservation: ensure protected docs make it into topK
  const topKResult = sorted.slice(0, topK);
  const topKIds = new Set(topKResult.map(c => c.doc_id));

  // Find protected docs that were cut
  const missingProtected: RRFCandidate[] = [];
  for (const candidate of sorted) {
    if (candidate.protected_by_bm25_event && !topKIds.has(candidate.doc_id)) {
      missingProtected.push(candidate);
    }
  }

  // Inject missing protected docs by replacing lowest-scored non-protected entries
  if (missingProtected.length > 0) {
    for (const missing of missingProtected) {
      // Find lowest non-protected entry in topK to replace
      for (let i = topKResult.length - 1; i >= 0; i--) {
        if (!topKResult[i].protected_by_bm25_event) {
          missing.applied_boosts.push('bm25_event_preservation: injected');
          topKResult[i] = missing;
          break;
        }
      }
    }
    // Re-sort after injection
    topKResult.sort((a, b) => b.adjusted_score - a.adjusted_score);
  }

  return topKResult;
}

/**
 * Original unmodified RRF for backward compatibility.
 */
export function rrfFusion(lanes: RRFLane[], options?: { k?: number; topK?: number }): RRFCandidate[] {
  const k = options?.k ?? 60;
  const topK = options?.topK ?? 10;
  const candidateMap = new Map<string, RRFCandidate>();

  for (const lane of lanes) {
    for (let rank = 0; rank < lane.results.length; rank++) {
      const r = lane.results[rank];
      const rrfContrib = 1 / (k + rank + 1);
      let candidate = candidateMap.get(r.doc_id);
      if (!candidate) {
        candidate = {
          doc_id: r.doc_id, title: r.title, source_type: r.source_type,
          rrf_score: 0, adjusted_score: 0,
          lanes: {}, raw_scores: {}, metadata: r.metadata,
          protected_by_bm25_event: false, event_primary_boost_applied: false,
          synthesis_guard_applied: false, applied_boosts: [],
        };
        candidateMap.set(r.doc_id, candidate);
      }
      candidate.rrf_score += rrfContrib;
      candidate.adjusted_score = candidate.rrf_score;
      const laneRank = rank + 1;
      switch (lane.name) {
        case 'bm25_event': candidate.lanes.bm25_event_rank = laneRank; break;
        case 'bm25_synthesis': candidate.lanes.bm25_synthesis_rank = laneRank; break;
        case 'vector_event': candidate.lanes.vector_event_rank = laneRank; break;
        case 'vector_synthesis': candidate.lanes.vector_synthesis_rank = laneRank; break;
      }
      if (lane.name.startsWith('bm25')) {
        candidate.raw_scores.bm25 = Math.max(candidate.raw_scores.bm25 ?? 0, r.score);
      } else {
        candidate.raw_scores.vector = Math.max(candidate.raw_scores.vector ?? 0, r.score);
      }
    }
  }
  const sorted = Array.from(candidateMap.values()).sort((a, b) => b.rrf_score - a.rrf_score);
  return sorted.slice(0, topK);
}
