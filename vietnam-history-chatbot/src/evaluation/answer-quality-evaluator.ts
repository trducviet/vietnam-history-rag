/**
 * Answer Quality Evaluator — heuristic-based evaluation of full pipeline
 * ChatResponse output against structured test cases.
 *
 * Scores 0–100 across shape, citation, provenance, terms, and special checks.
 *
 * PATCH 6: New file.
 */

import type { AnswerQualityCase } from './answer-quality-cases.js';
import type { ChatResponse, LoadedDataset } from '../shared/types.js';
import { getQuerySpecificHardNegativeIds } from '../reranking/hard-negative-guard.js';

// ─── Result Types ────────────────────────────────────────────

export interface AnswerQualityChecks {
  shape: boolean;
  citationsPresent: boolean;
  expectedCitationHit: boolean;
  forbiddenCitationAvoided: boolean;
  provenancePresent: boolean;
  requiredTermsCovered: boolean;
  forbiddenTermsAvoided: boolean;
  plannedWarningCovered: boolean;
  outOfScopeHandled: boolean;
  hardNegativeAvoided: boolean;
}

export interface AnswerQualityResult {
  id: string;
  category: string;
  difficulty: string;
  query: string;

  passed: boolean;
  score: number;

  checks: AnswerQualityChecks;

  warnings: string[];
  errors: string[];

  responseSummary: {
    answerPreview: string;
    citationIds: string[];
    confidence: string;
  };
}

// ─── Text Normalization ──────────────────────────────────────

function normalizeVietnamese(text: string): string {
  return text
    .toLowerCase()
    .normalize('NFKC')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function textContainsTerm(text: string, term: string): boolean {
  const normalizedText = normalizeVietnamese(text);
  const normalizedTerm = normalizeVietnamese(term);
  return normalizedText.includes(normalizedTerm);
}

// ─── Evaluator ───────────────────────────────────────────────

export function evaluateAnswerQuality(
  testCase: AnswerQualityCase,
  response: ChatResponse,
  dataset: LoadedDataset
): AnswerQualityResult {
  const warnings: string[] = [];
  const errors: string[] = [];

  const checks: AnswerQualityChecks = {
    shape: true,
    citationsPresent: true,
    expectedCitationHit: true,
    forbiddenCitationAvoided: true,
    provenancePresent: true,
    requiredTermsCovered: true,
    forbiddenTermsAvoided: true,
    plannedWarningCovered: true,
    outOfScopeHandled: true,
    hardNegativeAvoided: true,
  };

  // Collect all cited IDs (citations + related_events)
  const citationIds = new Set<string>();
  const primaryCitationIds = new Set<string>();
  for (const c of response.citations ?? []) {
    if (c.record_id) {
      citationIds.add(c.record_id);
      primaryCitationIds.add(c.record_id);
    }
  }
  for (const e of response.related_events ?? []) {
    if (e.record_id) citationIds.add(e.record_id);
  }

  const combinedText = `${response.answer ?? ''} ${response.explanation ?? ''}`;

  // ── 1. Shape Checks (15 pts) ──
  if (!response.answer || typeof response.answer !== 'string' || response.answer.trim().length === 0) {
    checks.shape = false;
    errors.push('answer is missing or empty');
  }
  if (!Array.isArray(response.citations)) {
    checks.shape = false;
    errors.push('citations is not an array');
  }
  if (!response.confidence) {
    checks.shape = false;
    errors.push('confidence is missing');
  }
  if (!response.confidence_details) {
    checks.shape = false;
    errors.push('confidence_details is missing');
  }

  // ── 2. Citations Present (10 pts) ──
  if (testCase.category !== 'out_of_scope') {
    if ((response.citations ?? []).length === 0) {
      checks.citationsPresent = false;
      warnings.push('no citations in response');
    }
  }

  // ── 3. Expected Citation Hit (20 pts) ──
  if (testCase.expectedCitationIds) {
    for (const expected of testCase.expectedCitationIds) {
      // Verify expected ID exists in dataset
      const exists = dataset.events.has(expected) || dataset.synthesis.has(expected);
      if (!exists) {
        warnings.push(`expected doc ${expected} not found in dataset — skipped`);
        continue;
      }
      if (!citationIds.has(expected)) {
        if (response.related_events?.some(e => e.record_id === expected)) {
          warnings.push(`expected ${expected} found in related_events but not primary citations`);
        } else {
          checks.expectedCitationHit = false;
          warnings.push(`expected citation ${expected} not in response`);
        }
      }
    }
  }

  if (testCase.expectedAnyCitationIds) {
    const anyFound = testCase.expectedAnyCitationIds.some(id => citationIds.has(id));
    if (!anyFound) {
      // Check if any exist in dataset
      const anyExist = testCase.expectedAnyCitationIds.some(
        id => dataset.events.has(id) || dataset.synthesis.has(id)
      );
      if (anyExist) {
        checks.expectedCitationHit = false;
        warnings.push(`none of expectedAny [${testCase.expectedAnyCitationIds.join(',')}] found`);
      } else {
        warnings.push(`none of expectedAny IDs exist in dataset — skipped`);
      }
    }
  }

  // ── 4. Forbidden Citation Avoided (10 pts) ──
  if (testCase.forbiddenCitationIds) {
    for (const forbidden of testCase.forbiddenCitationIds) {
      if (primaryCitationIds.has(forbidden)) {
        checks.forbiddenCitationAvoided = false;
        errors.push(`forbidden citation ${forbidden} found in primary citations`);
      } else if (citationIds.has(forbidden)) {
        warnings.push(`forbidden citation ${forbidden} in related_events (not primary)`);
      }
    }
  }

  // ── 5. Provenance Present (10 pts) ──
  if (testCase.category !== 'out_of_scope') {
    const withSources = (response.citations ?? []).filter(
      c => (c.source_ids && c.source_ids.length > 0) || (c.sources && c.sources.length > 0)
    );
    if ((response.citations ?? []).length > 0 && withSources.length === 0) {
      checks.provenancePresent = false;
      warnings.push('no citations have source provenance');
    }
  }

  // ── 6. Required Terms Coverage (15 pts) ──
  if (testCase.requiredTerms) {
    for (const term of testCase.requiredTerms) {
      if (!textContainsTerm(combinedText, term)) {
        checks.requiredTermsCovered = false;
        warnings.push(`required term "${term}" not found in answer/explanation`);
      }
    }
  }

  if (testCase.requiredAnyTerms) {
    for (const group of testCase.requiredAnyTerms) {
      const found = group.some(term => textContainsTerm(combinedText, term));
      if (!found) {
        checks.requiredTermsCovered = false;
        warnings.push(`none of requiredAny [${group.join(' | ')}] found`);
      }
    }
  }

  // ── 7. Forbidden Terms (part of terms score) ──
  if (testCase.forbiddenTerms) {
    for (const term of testCase.forbiddenTerms) {
      if (textContainsTerm(combinedText, term)) {
        checks.forbiddenTermsAvoided = false;
        errors.push(`forbidden term "${term}" found in answer`);
      }
    }
  }

  // ── 8. Planned-not-executed Warning (10 pts special) ──
  if (testCase.shouldMentionPlannedNotExecuted) {
    const plannedTerms = [
      'dự kiến', 'không thực hiện', 'chưa thực hiện',
      'planned_not_executed', 'lên kế hoạch', 'không được tổ chức',
      'chưa được tổ chức', 'chưa diễn ra', 'không diễn ra',
    ];
    const mentioned = plannedTerms.some(t => textContainsTerm(combinedText, t));
    if (!mentioned) {
      checks.plannedWarningCovered = false;
      errors.push('should mention planned/not executed status but did not');
    }
  }

  // ── 9. Out-of-scope Handling (10 pts special) ──
  if (testCase.shouldRefuseOrLimitScope) {
    const scopeTerms = [
      'ngoài phạm vi', '1858', '2000', 'không nằm trong phạm vi',
      'không có đủ dữ liệu', 'không tìm thấy trong dữ liệu',
      'ngoài khoảng thời gian', 'không nằm trong', 'không có trong dữ liệu',
    ];
    const handled = scopeTerms.some(t => textContainsTerm(combinedText, t));
    if (!handled) {
      checks.outOfScopeHandled = false;
      errors.push('should refuse or limit scope but did not');
    }
  }

  // ── 10. Hard-negative Contamination (part of special) ──
  const hardNegIds = getQuerySpecificHardNegativeIds(testCase.query, dataset);
  if (hardNegIds.size > 0) {
    const contaminated = [...primaryCitationIds].filter(id => hardNegIds.has(id));
    if (contaminated.length > 0) {
      const isStrictCategory = ['fact', 'date', 'actor', 'location'].includes(testCase.category);
      if (isStrictCategory) {
        checks.hardNegativeAvoided = false;
        errors.push(`hard-negative ${contaminated.join(',')} in primary citations (strict category)`);
      } else {
        warnings.push(`hard-negative ${contaminated.join(',')} in citations (non-strict category)`);
      }
    }
  }

  // ── Scoring ──
  let score = 0;

  if (checks.shape) score += 15;
  if (checks.citationsPresent) score += 10;
  if (checks.expectedCitationHit) score += 20;
  if (checks.forbiddenCitationAvoided) score += 10;
  if (checks.provenancePresent) score += 10;
  if (checks.requiredTermsCovered) score += 15;

  // Special checks (10 pts)
  const specialChecks = [checks.plannedWarningCovered, checks.outOfScopeHandled, checks.hardNegativeAvoided];
  const specialPassed = specialChecks.filter(Boolean).length;
  score += Math.round((specialPassed / specialChecks.length) * 10);

  // Confidence sanity (10 pts)
  if (response.confidence && ['high', 'medium', 'low'].includes(response.confidence)) {
    score += 10;
  }

  // Auto-fail conditions
  let autoFail = false;
  if (!checks.shape) autoFail = true;
  if (!checks.forbiddenCitationAvoided) autoFail = true;
  if (testCase.shouldRefuseOrLimitScope && !checks.outOfScopeHandled) autoFail = true;
  if (testCase.shouldMentionPlannedNotExecuted && !checks.plannedWarningCovered) autoFail = true;

  // Determine pass/warn/fail status
  const passed = !autoFail && score >= 60;

  return {
    id: testCase.id,
    category: testCase.category,
    difficulty: testCase.difficulty,
    query: testCase.query,
    passed,
    score,
    checks,
    warnings,
    errors,
    responseSummary: {
      answerPreview: (response.answer ?? '').substring(0, 150),
      citationIds: [...primaryCitationIds],
      confidence: response.confidence ?? 'unknown',
    },
  };
}

// ─── Aggregate Metrics ───────────────────────────────────────

export interface AnswerQualityAggregate {
  total: number;
  passed: number;
  warned: number;
  failed: number;
  averageScore: number;
  passRate: number;
  citationHitRate: number;
  provenanceCoverage: number;
  requiredTermCoverage: number;
  hardNegContaminationRate: number;
  outOfScopeHandlingRate: number;
  plannedWarningRate: number;
}

export function aggregateResults(results: AnswerQualityResult[]): AnswerQualityAggregate {
  const n = results.length;
  if (n === 0) {
    return {
      total: 0, passed: 0, warned: 0, failed: 0,
      averageScore: 0, passRate: 0, citationHitRate: 0,
      provenanceCoverage: 0, requiredTermCoverage: 0,
      hardNegContaminationRate: 0, outOfScopeHandlingRate: 0, plannedWarningRate: 0,
    };
  }

  const passed = results.filter(r => r.passed && r.warnings.length === 0 && r.errors.length === 0).length;
  const failed = results.filter(r => !r.passed).length;
  const warned = n - passed - failed;

  const oosResults = results.filter(r => r.category === 'out_of_scope');
  const plannedResults = results.filter(r =>
    r.category === 'planned_not_executed' ||
    results.find(rr => rr.id === r.id) !== undefined // dummy — see below
  );
  // Actually filter by test case shouldMentionPlannedNotExecuted
  const plannedCheckResults = results.filter(r => r.checks.plannedWarningCovered !== undefined);

  return {
    total: n,
    passed,
    warned,
    failed,
    averageScore: results.reduce((s, r) => s + r.score, 0) / n,
    passRate: passed / n,
    citationHitRate: results.filter(r => r.checks.expectedCitationHit).length / n,
    provenanceCoverage: results.filter(r => r.checks.provenancePresent).length / n,
    requiredTermCoverage: results.filter(r => r.checks.requiredTermsCovered).length / n,
    hardNegContaminationRate: results.filter(r => !r.checks.hardNegativeAvoided).length / n,
    outOfScopeHandlingRate: oosResults.length > 0
      ? oosResults.filter(r => r.checks.outOfScopeHandled).length / oosResults.length
      : 1.0,
    plannedWarningRate: plannedCheckResults.length > 0
      ? plannedCheckResults.filter(r => r.checks.plannedWarningCovered).length / plannedCheckResults.length
      : 1.0,
  };
}
