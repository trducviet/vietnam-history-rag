#!/usr/bin/env python3
"""9Router cloud-embedding retrieval adapter for API-fast web mode.

This adapter intentionally does not replace the local/no-cloud adapter. It
uses the Stage 14C cloud embedding cache and FAISS index, plus local BM25 and
RRF, then returns the same retrieval/citation payload shape expected by the
web API. Answer generation still happens in the TypeScript API route.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[2]
LOCAL_CLI_PATH = BASE / "scripts" / "web-demo" / "local-hybrid-chat-cli.py"
CORPUS_FILE = BASE / "vietnam_history_dataset_runtime_optimal_pack" / "corpus" / "runtime" / "combined_runtime_hybrid.jsonl"
SOURCES_FILE = BASE / "vietnam_history_dataset_runtime_optimal_pack" / "corpus" / "sources.jsonl"
CLOUD_EMB_DIR = BASE / "data_packs" / "embeddings" / "stage14c_9router_cloud_embedding_index_and_dual_retrieval"
CLOUD_INDEX_PATH = CLOUD_EMB_DIR / "index" / "faiss.index"
CLOUD_METADATA_PATH = CLOUD_EMB_DIR / "cache" / "embedding_metadata.jsonl"
ENV_PATH = BASE / "vietnam-history-chatbot" / ".env"

_LOCAL_CLI_CACHE: Any | None = None
_CLOUD_RUNTIME_CACHE: "CloudEmbeddingHybridRuntime | None" = None
DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")


def load_local_cli() -> Any:
    global _LOCAL_CLI_CACHE
    if _LOCAL_CLI_CACHE is not None:
        return _LOCAL_CLI_CACHE
    spec = importlib.util.spec_from_file_location("local_hybrid_chat_cli", LOCAL_CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load local hybrid CLI from {LOCAL_CLI_PATH}")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    _LOCAL_CLI_CACHE = module
    return module


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class CloudEmbeddingHybridRuntime:
    def __init__(self, stage13a: Any) -> None:
        self.stage13a = stage13a
        env = {**load_env_file(ENV_PATH), **os.environ}
        self.base_url = (env.get("9ROUTER_BASE_URL") or "http://localhost:20128/v1").rstrip("/")
        self.api_key = env.get("9ROUTER_API_KEY") or ""
        self.embedding_model = env.get("9ROUTER_EMBEDDING_MODEL") or ""
        self.timeout_s = int(env.get("9ROUTER_EMBEDDING_TIMEOUT_MS") or "120000") / 1000
        self.normalize = str(env.get("9ROUTER_EMBEDDING_NORMALIZE") or "true").lower() in {"1", "true", "yes", "on"}
        self.cloud_query_embedding_calls = 0
        self.local_query_embedding_calls = 0
        self.faiss_search_calls = 0
        self.bm25_search_calls = 0
        self.rrf_fusion_calls = 0
        self.local_llm_calls = 0
        self.last_query_embedding_latency_ms = 0.0
        self.last_vector_search_latency_ms = 0.0
        self.query_embedding_cache_hits = 0
        self.query_embedding_cache_misses = 0
        self.query_embedding_cache: dict[str, np.ndarray] = {}
        self.usable: list[dict[str, Any]] = []
        self.chunk_map: dict[str, dict[str, Any]] = {}
        self.sources: dict[str, dict[str, Any]] = {}
        self.vec_to_chunk: dict[int, str] = {}
        self.embedding_dimension = 0
        self.index_dimension = 0
        self.index: Any = None
        self.bm25: Any = None

    def load(self) -> None:
        if not self.api_key or not self.embedding_model:
            raise RuntimeError("9Router cloud embedding mode is missing 9ROUTER_API_KEY or 9ROUTER_EMBEDDING_MODEL")
        if not CLOUD_INDEX_PATH.exists() or not CLOUD_METADATA_PATH.exists():
            raise RuntimeError("Stage14C cloud FAISS index/cache is missing; build cloud embeddings first")
        chunks = [json.loads(line) for line in CORPUS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.usable = [c for c in chunks if c.get("canonical", True) and (c.get("text_for_embedding", "") or "").strip()]
        self.chunk_map = {c.get("doc_id", ""): c for c in self.usable}
        self.sources = {
            s.get("source_id", ""): s
            for s in (json.loads(line) for line in SOURCES_FILE.read_text(encoding="utf-8").splitlines() if line.strip())
        }
        self.index = faiss.read_index(str(CLOUD_INDEX_PATH))
        self.index_dimension = int(self.index.d)
        meta = [json.loads(line) for line in CLOUD_METADATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.vec_to_chunk = {int(m["vector_id"]): m["chunk_id"] for m in meta}
        self.embedding_dimension = int(meta[0].get("embedding_dimension") or self.index_dimension) if meta else self.index_dimension
        if self.embedding_dimension != self.index_dimension:
            raise RuntimeError(f"Cloud embedding dimension mismatch: metadata={self.embedding_dimension}, index={self.index_dimension}")
        if self.index.ntotal != len(meta):
            raise RuntimeError(f"Cloud FAISS metadata mismatch: index={self.index.ntotal}, metadata={len(meta)}")
        self.bm25 = self.stage13a.BM25Okapi(
            [
                self.stage13a.vn_tokenize(c.get("text_for_embedding", "") + " " + c.get("title", ""))
                for c in self.usable
            ]
        )

    def row(self, c: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return {
            "chunk_id": extra.pop("chunk_id", c.get("doc_id", "")),
            "title": c.get("title", ""),
            "text": c.get("text_for_embedding", ""),
            "summary": c.get("summary", ""),
            "source_ids": c.get("source_ids", []),
            "doc_source": c.get("doc_source", ""),
            "year": c.get("year"),
            "bm25_rank": extra.get("bm25_rank"),
            "bm25_score": extra.get("bm25_score"),
            "vector_rank": extra.get("vector_rank"),
            "vector_score": extra.get("vector_score"),
            "rrf_score": None,
        }

    def search_bm25(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.stage13a.TOP_K
        self.bm25_search_calls += 1
        scores = self.bm25.get_scores(self.stage13a.vn_tokenize(query))
        results = []
        for rank, idx in enumerate(np.argsort(scores)[::-1][:top_k]):
            if scores[int(idx)] <= 0:
                continue
            results.append(self.row(self.usable[int(idx)], bm25_rank=rank + 1, bm25_score=round(float(scores[int(idx)]), 4)))
        return results

    def embed_query(self, query: str) -> np.ndarray:
        started = time.perf_counter()
        key = f"{self.embedding_model}:{self.index_dimension}:{query}"
        if key in self.query_embedding_cache:
            self.query_embedding_cache_hits += 1
            self.last_query_embedding_latency_ms = round((time.perf_counter() - started) * 1000, 1)
            return self.query_embedding_cache[key].copy()
        self.query_embedding_cache_misses += 1
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.embedding_model, "input": [query]},
            timeout=self.timeout_s,
        )
        self.cloud_query_embedding_calls += 1
        if response.status_code >= 400:
            raise RuntimeError(f"9Router embedding error {response.status_code}: {response.text[:240]}")
        data = response.json()
        embedding = data.get("data", [{}])[0].get("embedding")
        if not embedding:
            raise RuntimeError("9Router embedding response did not include an embedding vector")
        vec = np.asarray([embedding], dtype=np.float32)
        if vec.shape[1] != self.index_dimension:
            raise RuntimeError(f"Query embedding dimension {vec.shape[1]} does not match cloud FAISS dimension {self.index_dimension}")
        if self.normalize:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vec = vec / norms
        self.last_query_embedding_latency_ms = round((time.perf_counter() - started) * 1000, 1)
        self.query_embedding_cache[key] = vec.astype(np.float32).copy()
        return vec.astype(np.float32)

    def search_vector(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.stage13a.TOP_K
        qvec = self.embed_query(query)
        started = time.perf_counter()
        self.faiss_search_calls += 1
        scores, indices = self.index.search(qvec, top_k)
        self.last_vector_search_latency_ms = round((time.perf_counter() - started) * 1000, 1)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            cid = self.vec_to_chunk.get(int(idx), "")
            if cid in self.chunk_map:
                results.append(self.row(self.chunk_map[cid], chunk_id=cid, vector_rank=rank + 1, vector_score=round(float(score), 4)))
        return results

    def exact_date_matches(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        match = DATE_RE.search(query or "")
        if not match:
            return []
        day, month, year = match.groups()
        variants = {
            f"{int(day)}/{int(month)}/{year}",
            f"{int(day):02d}/{int(month):02d}/{year}",
            f"{int(day)}-{int(month)}-{year}",
            f"{int(day):02d}-{int(month):02d}-{year}",
        }
        rows: list[dict[str, Any]] = []
        for chunk in self.usable:
            haystack = " ".join(
                str(chunk.get(key) or "")
                for key in ("title", "summary", "text_for_embedding")
            )
            if any(variant in haystack for variant in variants):
                row = self.row(chunk, bm25_rank=1, bm25_score=999.0)
                row["exact_date_promoted"] = True
                rows.append(row)
                if len(rows) >= top_k:
                    break
        return rows

    def rrf_fuse(self, bm25_res: list[dict[str, Any]], vec_res: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.stage13a.TOP_K
        self.rrf_fusion_calls += 1
        scores: dict[str, float] = defaultdict(float)
        bm = {r["chunk_id"]: r for r in bm25_res}
        vc = {r["chunk_id"]: r for r in vec_res}
        for r in bm25_res:
            scores[r["chunk_id"]] += 1.0 / (self.stage13a.RRF_K + r["bm25_rank"])
        for r in vec_res:
            scores[r["chunk_id"]] += 1.0 / (self.stage13a.RRF_K + r["vector_rank"])
        rows = []
        for rank, cid in enumerate(sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]):
            base = dict(bm.get(cid, vc.get(cid, {})))
            base["rrf_score"] = round(scores[cid], 6)
            base["hybrid_rank"] = rank + 1
            if cid in bm:
                base["bm25_rank"] = bm[cid]["bm25_rank"]
                base["bm25_score"] = bm[cid]["bm25_score"]
            if cid in vc:
                base["vector_rank"] = vc[cid]["vector_rank"]
                base["vector_score"] = vc[cid]["vector_score"]
            rows.append(base)
        return rows

    def hybrid_retrieve(self, query: str, is_comparison: bool = False, entities: list[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        entities = entities or []
        t0 = self.stage13a.now_ms()
        if is_comparison and len(entities) >= 2:
            a, b = entities[0], entities[1]
            bm_a, bm_b = self.search_bm25(a, self.stage13a.COMP_SIDE_K), self.search_bm25(b, self.stage13a.COMP_SIDE_K)
            vc_a, vc_b = self.search_vector(a, self.stage13a.COMP_SIDE_K), self.search_vector(b, self.stage13a.COMP_SIDE_K)
            bm_g, vc_g = self.search_bm25(query, self.stage13a.TOP_K), self.search_vector(query, self.stage13a.TOP_K)
            seen: set[str] = set()
            merged: list[dict[str, Any]] = []
            for row in (bm_a + vc_a)[:2]:
                if row["chunk_id"] not in seen:
                    row["comparison_side"] = "A"
                    merged.append(row)
                    seen.add(row["chunk_id"])
            for row in (bm_b + vc_b)[:2]:
                if row["chunk_id"] not in seen:
                    row["comparison_side"] = "B"
                    merged.append(row)
                    seen.add(row["chunk_id"])
            for row in self.rrf_fuse(bm_g, vc_g):
                if row["chunk_id"] not in seen and len(merged) < self.stage13a.TOP_K:
                    merged.append(row)
                    seen.add(row["chunk_id"])
            for i, row in enumerate(merged):
                row["hybrid_rank"] = i + 1
            return merged, {"mode": "comparison_balanced_cloud_embedding", "latency_ms": round(self.stage13a.now_ms() - t0, 1), "rrf_k": self.stage13a.RRF_K}
        bm, vc = self.search_bm25(query, self.stage13a.TOP_K), self.search_vector(query, self.stage13a.TOP_K)
        fused = self.rrf_fuse(bm, vc)
        promoted = self.exact_date_matches(query)
        if promoted:
            seen = {row["chunk_id"] for row in promoted}
            fused = promoted + [row for row in fused if row.get("chunk_id") not in seen]
            for i, row in enumerate(fused[: self.stage13a.TOP_K]):
                row["hybrid_rank"] = i + 1
            fused = fused[: self.stage13a.TOP_K]
        return fused, {
            "mode": "hybrid_rrf_cloud_embedding",
            "latency_ms": round(self.stage13a.now_ms() - t0, 1),
            "rrf_k": self.stage13a.RRF_K,
            "exact_date_promoted": bool(promoted),
        }

    def source_title(self, source_id: str) -> str:
        return self.sources.get(source_id, {}).get("title") or source_id


def get_cloud_runtime(stage13a: Any) -> CloudEmbeddingHybridRuntime:
    global _CLOUD_RUNTIME_CACHE
    if _CLOUD_RUNTIME_CACHE is None:
        runtime = CloudEmbeddingHybridRuntime(stage13a)
        with contextlib.redirect_stdout(sys.stderr):
            runtime.load()
        _CLOUD_RUNTIME_CACHE = runtime
    return _CLOUD_RUNTIME_CACHE


def run(req: dict[str, Any]) -> dict[str, Any]:
    local_cli = load_local_cli()
    local_cli.get_runtime = get_cloud_runtime
    response = local_cli.run(req)
    runtime = _CLOUD_RUNTIME_CACHE
    debug = response.setdefault("debug", {})
    status = response.setdefault("status", {})
    if runtime is not None:
        debug.update(
            {
                "runtime_mode": "api_9router_fast",
                "embedding_provider": "9router_embedding",
                "vector_index_provider": "9router_stage14c",
                "query_embedding_dimension": runtime.embedding_dimension,
                "faiss_index_dimension": runtime.index_dimension,
                "embedding_model": runtime.embedding_model,
                "cloud_embedding_calls": runtime.cloud_query_embedding_calls,
                "cloud_api_calls": runtime.cloud_query_embedding_calls,
                "cloud_query_embedding_calls": runtime.cloud_query_embedding_calls,
                "cloud_llm_calls": 0,
                "local_query_embedding_calls": 0,
                "query_embedding_generated": runtime.cloud_query_embedding_calls > 0,
                "query_embedding_cache": {
                    "enabled": True,
                    "hits": runtime.query_embedding_cache_hits,
                    "misses": runtime.query_embedding_cache_misses,
                    "provider": "9router_embedding",
                    "model": runtime.embedding_model,
                },
                "query_embedding_latency_ms": runtime.last_query_embedding_latency_ms,
                "cloud_vector_search_latency_ms": runtime.last_vector_search_latency_ms,
                "local_bm25_retrieval": True,
                "cloud_vector_retrieval": runtime.faiss_search_calls > 0,
                "corpus_sha256": hashlib.sha256(CORPUS_FILE.read_bytes()).hexdigest(),
            }
        )
        if debug.get("retrieval_mode") == "hybrid_rrf":
            debug["retrieval_mode"] = "hybrid_rrf"
        trace = debug.get("retrieval_trace") or {}
        if trace.get("mode") in {"hybrid_rrf", "comparison_balanced"}:
            trace["mode"] = "hybrid_rrf_cloud_embedding"
            debug["retrieval_trace"] = trace
    status["no_cloud"] = False
    status["api_fast_mode"] = True
    status["cloud_embedding_retrieval"] = bool(runtime and runtime.faiss_search_calls > 0)
    status["retrieval_local"] = False
    status["bm25_local"] = True
    return response


def main() -> int:
    try:
        request = json.loads(sys.stdin.read() or "{}")
        response = run(request)
        print(json.dumps(response, ensure_ascii=False))
        return 0
    except Exception as exc:
        payload = {
            "error": "9router_cloud_retrieval_runtime_error",
            "message": str(exc),
            "cloud_embedding_calls": _CLOUD_RUNTIME_CACHE.cloud_query_embedding_calls if _CLOUD_RUNTIME_CACHE else 0,
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
