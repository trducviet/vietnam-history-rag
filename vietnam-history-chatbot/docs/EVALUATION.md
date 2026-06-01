# Evaluation Guide

## Overview

The Vietnamese Historical RAG Chatbot includes a two-track evaluation framework:

1. **Retrieval Evaluation** — measures document retrieval quality (48 queries)
2. **Answer Quality Evaluation** — measures end-to-end answer correctness (20 cases)

Both tracks run offline in BM25-only mode without requiring an OpenAI API key.

## Quick Start

```bash
# 1. Compile
npx tsc --noEmit

# 2. Build
npm run build

# 3. Ingest data (skip embeddings for BM25-only)
npx tsx scripts/ingest.ts --skip-embeddings

# 4. Run retrieval evaluation
npx tsx scripts/run-retrieval-eval.ts

# 5. Run answer quality evaluation
npx tsx scripts/run-answer-quality-eval.ts

# 6. (Optional) Generate summary report from results
npx tsx scripts/generate-final-evaluation-report.ts
```

## Expected Results

### Retrieval

| Metric | Expected |
|---|---|
| Recall@5 | 100.0% |
| MRR | 0.957 |
| Failures@5 | 0 |

### Answer Quality

| Metric | Expected |
|---|---|
| Total cases | 20 |
| Failed | 0 |
| Average Score | ≥ 93 |
| Hard-neg Contamination | 0% |

## Reports

After running evaluations, reports are generated in `reports/`:

| Report | Description |
|---|---|
| `FINAL_EVALUATION_REPORT.md` | Comprehensive evaluation summary |
| `EVALUATION_METHODOLOGY.md` | Metric definitions and methodology |
| `METRICS_HISTORY.md` | Performance progression across patches |
| `FINAL_REGRESSION_CHECKLIST.md` | Pre-demonstration quality gates |
| `answer-quality-report.md` | Per-case answer quality results (auto-generated) |
| `answer-quality-results.json` | Machine-readable AQ results (auto-generated) |

## Evaluation Architecture

```
Dataset (526 docs) → BM25 Index
                   ↓
Query → Route → Retrieve → Rerank → Evidence Select → Context Build → Generate
                                                                      ↓
                                                               Answer Verifier
                                                                      ↓
                                                               Evaluated Response
```

For detailed methodology, see [EVALUATION_METHODOLOGY.md](../reports/EVALUATION_METHODOLOGY.md).
