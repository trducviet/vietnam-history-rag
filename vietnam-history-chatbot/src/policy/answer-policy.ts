/**
 * Answer Policy Layer — PATCH 9I
 *
 * Converts a CapabilityDecision into runtime answer adjustments.
 * Applied AFTER answer generation, BEFORE final return.
 *
 * Light-touch design: adjusts confidence ceiling, adds caution phrasing
 * for certain policies, but does NOT regenerate answers or remove
 * valid citations for SAFE_* cases.
 */

import type { CapabilityDecision } from './capability-types.js';
import type { ChatResponse } from '../shared/types.js';

export interface PolicyResult {
  adjustedResponse: ChatResponse;
  policyApplied: string;
  adjustments: string[];
}

/**
 * Apply the answer policy from a CapabilityDecision to the ChatResponse.
 */
export function applyAnswerPolicy(
  response: ChatResponse,
  decision: CapabilityDecision,
): PolicyResult {
  const adjustments: string[] = [];
  const adjusted = { ...response };

  // ── Confidence ceiling enforcement ──
  if (decision.confidenceCeiling === 'low' && adjusted.confidence !== 'low') {
    adjusted.confidence = 'low';
    adjustments.push(`confidence_capped_to_low`);
  }
  if (decision.confidenceCeiling === 'medium' && adjusted.confidence === 'high') {
    adjusted.confidence = 'medium';
    adjustments.push(`confidence_capped_to_medium`);
  }

  // ── Citation enforcement ──
  if (!decision.citationsAllowed && adjusted.citations.length > 0) {
    adjusted.citations = [];
    adjustments.push(`citations_cleared_by_policy`);
  }

  // ── Policy-specific adjustments ──
  switch (decision.policy) {
    case 'FULL_ANSWER':
      // No adjustments needed — answer as-is
      break;

    case 'FULL_ANSWER_OR_CAUTION':
      // If evidence is narrow, already handled by existing verifier
      // Just ensure confidence isn't overriding
      break;

    case 'HONEST_PARTIAL':
      // 9I-R: Enforce missing evidence acknowledgement
      if (decision.mustAcknowledgeMissingEvidence) {
        const al = adjusted.answer.toLowerCase();
        const hasAck = al.includes('chưa có đủ') || al.includes('chưa có bằng chứng') ||
          al.includes('không đủ bằng chứng') || al.includes('ngoài phạm vi') ||
          al.includes('thiếu ngữ cảnh') || al.includes('ngữ cảnh hiện có chưa') ||
          al.includes('tài liệu hiện có chỉ') || al.includes('chỉ tìm được bằng chứng cho một phần') ||
          al.includes('corpus hiện tại chưa') || al.includes('chưa đủ dữ liệu');
        if (!hasAck) {
          adjusted.answer += '\n\n⚠️ Lưu ý: Tài liệu hiện có chỉ cung cấp bằng chứng một phần cho câu hỏi này. Một số khía cạnh chưa có đủ dữ liệu trong nguồn hiện tại.';
          adjustments.push('missing_evidence_ack_injected');
        }
        adjustments.push('honest_partial_policy_applied');
      }
      break;

    case 'LOW_EVIDENCE_CAUTION': {
      // 9I-R: Also inject missing evidence ack for low-evidence caution
      const alLow = adjusted.answer.toLowerCase();
      const hasAckLow = alLow.includes('chưa có đủ') || alLow.includes('tài liệu hiện có chỉ') ||
        alLow.includes('chưa đủ dữ liệu') || alLow.includes('chưa có bằng chứng');
      if (!hasAckLow) {
        adjusted.answer += '\n\n⚠️ Lưu ý: Tài liệu hiện có chỉ cung cấp bằng chứng một phần cho câu hỏi này. Một số khía cạnh chưa có đủ dữ liệu trong nguồn hiện tại.';
        adjustments.push('low_evidence_ack_injected');
      }
      adjustments.push('low_evidence_caution_policy_applied');
      break;
    }

    case 'ASK_CLARIFICATION':
      // Already handled by scope guard early exit
      if (adjusted.citations.length > 0) {
        adjusted.citations = [];
        adjustments.push('clarification_citations_cleared');
      }
      break;

    case 'REFUSE_OOS':
      // Already handled by scope guard early exit
      if (adjusted.citations.length > 0) {
        adjusted.citations = [];
        adjustments.push('oos_citations_cleared');
      }
      break;

    case 'CORPUS_GAP_NOTICE': {
      // Ensure low confidence for corpus gap
      adjusted.confidence = 'low';
      // 9I-R: Also inject ack for corpus gap
      const alGap = adjusted.answer.toLowerCase();
      const hasAckGap = alGap.includes('chưa có đủ') || alGap.includes('tài liệu hiện có chỉ') ||
        alGap.includes('chưa đủ dữ liệu') || alGap.includes('chưa có bằng chứng');
      if (!hasAckGap) {
        adjusted.answer += '\n\n⚠️ Lưu ý: Tài liệu hiện có chỉ cung cấp bằng chứng một phần cho câu hỏi này. Một số khía cạnh chưa có đủ dữ liệu trong nguồn hiện tại.';
        adjustments.push('corpus_gap_ack_injected');
      }
      adjustments.push('corpus_gap_notice_applied');
      break;
    }
  }

  return {
    adjustedResponse: adjusted,
    policyApplied: decision.policy,
    adjustments,
  };
}
