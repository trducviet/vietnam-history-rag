# Vietnam History Chatbot Final

Final GitHub package for the Vietnamese history RAG chatbot.

## Scope

The system supports key verified topics from the internal corpus for Vietnamese history in the 1930-1975 experimental scope. It should not be presented as covering every possible history question from 1930-1975 or 1858-2000.

Final runtime:

- Web UI: Express static frontend.
- API server: Node.js/Express.
- RAG runtime: persistent Python service.
- Retrieval: local BM25/FAISS hybrid retrieval with RRF.
- LLM: 9Router OpenAI-compatible `chatx`.
- Safety: citation validation, guard/OOS handling, and session memory.

## Final Metrics

Answer quality, 500 questions:

- Mean score: `8.766/10`
- Median: `10/10`
- HTTP success: `489/500`
- `>= 8` score rate: `79.20%`
- Critical fail: `65/500`

Retrieval benchmark, 300 queries:

- Recall@1: `77.33%`
- Recall@3: `93.00%`
- Recall@5: `94.33%`
- Precision@5: `65.07%`
- MRR@5: `0.8398`
- NDCG@5: `0.8069`
- Evidence coverage: `93.53%`
- Citation alignment: `93.67%`

## Repository Layout

```text
vietnam-history-chatbot-final/
  vietnam-history-chatbot/     Node.js web/API project
  scripts/web-demo/            Persistent Python RAG runtime
  scripts/data-pack/           Runtime profile modules used by the final profile
  data_packs/                  Minimal final runtime corpus/index/metadata
  reports/                     Final reports and evaluation artifacts
```

## Setup

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Install Node.js dependencies:

```powershell
cd vietnam-history-chatbot
npm install
```

Create environment file:

```powershell
copy .env.example .env
```

Then edit `vietnam-history-chatbot/.env` and set:

```env
9ROUTER_API_KEY=replace_me
9ROUTER_BASE_URL=http://localhost:20128/v1
9ROUTER_MODEL=chatx
```

## Run

Terminal 1, start the persistent RAG service:

```powershell
.\start_runtime.ps1
```

Terminal 2, start the web server:

```powershell
.\start_web.ps1
```

Open:

```text
http://localhost:3000
```

## Smoke Test

After starting the runtime service:

```powershell
.\smoke_test.ps1
```

Expected behavior:

- In-scope history questions return concise answers with citations.
- OOS questions return:

```text
Câu hỏi này chưa có nguồn phù hợp trong tư liệu nội bộ, nên hệ thống không trả lời ngoài phạm vi đã kiểm chứng.
```

## Reports

Final project report:

```text
reports/final_project_report/final_project_report.md
reports/final_project_report/final_metrics_summary.json
reports/final_project_report/final_category_metrics.csv
```

Retrieval 300-query benchmark:

```text
reports/retrieval_300/
```

Answer-quality 500-question artifacts:

```text
reports/answer_quality_500/
```

Thesis DOCX:

```text
reports/thesis_docx/Bao_cao_do_an_chatbot_lich_su_viet_nam.docx
```

## Notes

- Real API keys are intentionally excluded.
- The package excludes intermediate repair folders, old logs, `node_modules`, Python caches, and older experimental artifacts.
- Keep the final claim conservative: the system supports verified key topics from its internal corpus, not all possible historical content.

