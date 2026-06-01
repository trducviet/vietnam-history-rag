# Robustness Testing Guide

## Overview

The robustness test suite (Patch 7I) extends the core 20-case answer quality benchmark with 60 additional cases designed to test system generalization across diverse question types, edge cases, and adversarial inputs.

**Key difference from core benchmark**: Robustness cases evaluate *behavior* (term coverage, scope handling, disambiguation phrasing) rather than *exact citation IDs*. This avoids overfitting to specific document retrieval ordering.

## Quick Start

```bash
npx tsx scripts/run-robustness-eval.ts
npx tsx scripts/run-robustness-eval.ts --json   # JSON output only
```

## Case Categories (60 total)

| Category | Count | Focus |
|---|---|---|
| date_lookup | 6 | Date/year extraction accuracy |
| actor_lookup | 5 | Person identification |
| location_lookup | 5 | Geographic reference |
| organization_lookup | 5 | Political/military organizations |
| treaty_lookup | 5 | Treaty content and provisions |
| clause_lookup | 3 | Specific treaty clauses |
| timeline | 5 | Chronological summaries |
| comparison | 6 | Cross-event analysis |
| disambiguation | 6 | Distinguishing similar events |
| misconception_check | 5 | Yes/no correction |
| out_of_scope | 5 | Scope boundary enforcement |
| adversarial | 4 | Vague/subjective queries |

## Status Definitions

| Status | Score | Meaning |
|---|---|---|
| PASS | ≥ 85, no warnings | Clean pass |
| WARN | ≥ 70, with warnings | Acceptable with minor issues |
| NEEDS_REVIEW | 60–84 | Uncertain — manual review recommended |
| FAIL | < 50 or critical error | Critical failure |

## Failure Modes

| Mode | Description |
|---|---|
| routing_error | Query routed to wrong intent/index |
| retrieval_error | Correct document not retrieved |
| evidence_role_error | Wrong document used as primary evidence |
| citation_error | Citation presence/absence mismatch |
| answer_wording_error | Missing required terms or phrasing |
| scope_error | OOS not detected or false OOS |
| weak_grounding | Answer too short or ungrounded |
| dataset_gap | Topic not covered in corpus |
| acceptable | Non-critical, expected behavior |

## Reports

After running, reports are generated in `reports/`:
- `ROBUSTNESS_EVAL_REPORT.md` — Human-readable summary with recommendations
- `robustness-results.json` — Machine-readable full results

## Relationship to Core Benchmark

The robustness suite is **exploratory** — it does not gate deployments. The core 20-case benchmark (`run-answer-quality-eval.ts`) remains the primary quality gate. Robustness results inform future improvement priorities.
