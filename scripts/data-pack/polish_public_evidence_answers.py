#!/usr/bin/env python3
"""Stage20F0 local style candidate.

This module projects Stage20D3 evidence into user-visible fields without
changing the indexed retrieval text or record ordering. The persistent service
can load it as an opt-in local/no-cloud profile and reuse the Stage20D3 vector
index after the invariant check created by ``--build`` passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STAGE = "DATA_PACKS_STAGE_20F0_LOCAL_ANSWER_NATURALNESS_AND_PUBLIC_EVIDENCE_RENDER_POLISH_NO_CLOUD"
BASE = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = BASE / "data_packs" / "human_gold" / "stage20d3_residual_template_polish_and_full_dual_mode_capture"
UPSTREAM_CORPUS = UPSTREAM_ROOT / "corpus" / "stage20d3_repaired_template_polish_corpus_candidate.jsonl"
UPSTREAM_INDEX = UPSTREAM_ROOT / "index" / "local_faiss.index"
UPSTREAM_METADATA = UPSTREAM_ROOT / "cache" / "local_embedding_metadata.jsonl"
UPSTREAM_EMBEDDINGS = UPSTREAM_ROOT / "cache" / "local_embeddings.float32.npy"
OUT = BASE / "data_packs" / "answer_style" / "stage20f0_local_answer_naturalness_public_evidence_render_polish_no_cloud"
CORPUS_FILE = OUT / "corpus" / "stage20f0_public_evidence_corpus_candidate.jsonl"
INDEX_FILE = OUT / "index" / "local_faiss.index"
METADATA_FILE = OUT / "cache" / "local_embedding_metadata.jsonl"
EMBEDDINGS_FILE = OUT / "cache" / "local_embeddings.float32.npy"
SUMMARY_FILE = OUT / "public_projection" / "public_projection_summary.json"
INVARIANT_FILE = OUT / "governance" / "retrieval_invariant_check.json"
BASE_RUNTIME_SCRIPT = BASE / "scripts" / "data-pack" / "build_canonical_corpus.py"
MODEL_NAME = "intfloat/multilingual-e5-base"
GENERATED_AT = datetime.now().astimezone().isoformat(timespec="seconds")

FORBIDDEN_PUBLIC_RE = re.compile(
    r"(Ý chính liên quan:|Cụm nhận diện chủ đề:|Các ý chính cần nhắc gồm:|"
    r"Khi hỏi nguồn|trọng tâm citation|Nội dung nguồn:|Nguồn chuẩn hóa|"
    r"Mốc chuẩn hóa|Từ khóa kiểm chứng:|canonical answer evidence)",
    re.IGNORECASE,
)
REMOVE_POINT_PREFIX_RE = re.compile(
    r"^\s*(Ý chính liên quan:|Cụm nhận diện chủ đề:|Các ý chính cần nhắc gồm:|"
    r"Khi hỏi nguồn|trọng tâm citation|Từ khóa kiểm chứng:|Mốc thời gian cần nhớ:)",
    re.IGNORECASE,
)
GENERATED_SOURCE_TITLE_RE = re.compile(
    r"(Nguồn chuẩn hóa|Nguồn timeline chuẩn hóa|Nguồn Stage20|Nguồn exact[- ]date|"
    r"Nguồn comparison|canonical answer evidence)",
    re.IGNORECASE,
)
PLACEHOLDER_COMPARISON_RE = re.compile(
    r"(Điểm cần phân biệt là phạm vi, thời điểm và vai trò|"
    r"Khi trả lời cần nêu rõ các từ khóa)",
    re.IGNORECASE,
)
SUPPORT_ONLY_POINT_RE = re.compile(
    r"^(Sự kiện này gắn với|Giai đoạn này gắn với|Chủ đề này còn gắn với)",
    re.IGNORECASE,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_BASE = load_module("stage20f0_base_unified_runtime", BASE_RUNTIME_SCRIPT)


def configure_runtime_root(root: Path | str = OUT) -> None:
    global OUT
    OUT = Path(root)
    _BASE.OUT = OUT


configure_runtime_root()
UnifiedRetriever = _BASE.UnifiedRetriever
query_intent = _BASE.query_intent
fold = _BASE.fold


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(rows: list[dict[str, Any]], field: str) -> str:
    value = "\n".join(str(row.get(field) or "") for row in rows)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n-;")


def clean_title(value: str, fallback: str = "") -> str:
    title = re.sub(r"\s*-\s*canonical answer evidence\b", "", str(value or ""), flags=re.IGNORECASE)
    if GENERATED_SOURCE_TITLE_RE.search(title):
        title = fallback or title
    title = re.sub(r"^\s*Bối cảnh chính của\s+", "", title, flags=re.IGNORECASE)
    return normalize_space(title) or normalize_space(fallback) or "Nguồn lịch sử"


def clean_text(value: str) -> str:
    text = str(value or "").replace("\r", " ")
    text = re.sub(r"\s*Từ khóa kiểm chứng:.*$", "", text, flags=re.IGNORECASE)
    segments: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        item = normalize_space(part)
        if not item or FORBIDDEN_PUBLIC_RE.search(item) or PLACEHOLDER_COMPARISON_RE.search(item):
            continue
        segments.append(item)
    return normalize_space(" ".join(segments))


def _join_terms(value: str) -> str:
    terms = [normalize_space(item) for item in re.split(r"[;,]", value) if normalize_space(item)]
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    return ", ".join(terms[:-1]) + " và " + terms[-1]


def clean_points(row: dict[str, Any]) -> list[str]:
    points: list[str] = []
    seen: set[str] = set()
    for raw in row.get("answer_points") or []:
        value = str(raw or "").strip()
        relation_match = re.match(r"^\s*Ý chính liên quan:\s*(.+)$", value, re.IGNORECASE)
        timeline_match = re.match(r"^\s*Các ý chính cần nhắc gồm:\s*(.+)$", value, re.IGNORECASE)
        topic_match = re.match(r"^\s*Cụm nhận diện chủ đề:\s*(.+)$", value, re.IGNORECASE)
        if relation_match:
            terms = _join_terms(relation_match.group(1).rstrip("."))
            value = f"Sự kiện này gắn với {terms}." if terms else ""
        elif timeline_match:
            terms = _join_terms(timeline_match.group(1).rstrip("."))
            value = f"Giai đoạn này gắn với {terms}." if terms else ""
        elif topic_match:
            terms = _join_terms(topic_match.group(1).rstrip("."))
            value = f"Chủ đề này còn gắn với {terms}." if terms else ""
        elif REMOVE_POINT_PREFIX_RE.search(value) or PLACEHOLDER_COMPARISON_RE.search(value):
            continue
        value = clean_text(value)
        key = fold(value) if value else ""
        if value and key not in seen:
            points.append(value)
            seen.add(key)
    if not points:
        summary = clean_text(str(row.get("summary") or ""))
        if summary:
            points.append(summary)
    return points


def public_source_cards(row: dict[str, Any], points: list[str], public_summary: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    fallback_excerpt = normalize_space(" ".join(points[:4])) or public_summary
    public_title = clean_title(str(row.get("title") or ""))
    for raw_card in row.get("source_cards") or []:
        card = deepcopy(raw_card)
        raw_title = str(card.get("source_title") or "")
        raw_excerpt = str(card.get("source_excerpt") or "")
        excerpt = clean_text(raw_excerpt)
        if GENERATED_SOURCE_TITLE_RE.search(raw_title) or not excerpt:
            excerpt = fallback_excerpt
        dates = list(dict.fromkeys(re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", raw_excerpt)))
        missing_dates = [date for date in dates if date not in excerpt]
        if missing_dates:
            excerpt = normalize_space(f"{excerpt} Mốc thời gian được ghi nhận: {', '.join(missing_dates)}.")
        if not excerpt:
            continue
        card["source_title"] = clean_title(raw_title, public_title)
        card["source_excerpt"] = excerpt
        cards.append(card)
    if not cards and fallback_excerpt:
        cards.append({
            "source_id": str(row.get("source_id") or row.get("canonical_id") or ""),
            "source_doc_id": str(row.get("original_doc_id") or row.get("canonical_id") or ""),
            "source_title": public_title,
            "source_excerpt": fallback_excerpt,
            "direct_evidence_pass": True,
        })
    return cards


def project_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = deepcopy(raw)
    points = clean_points(row)
    summary = clean_text(str(row.get("summary") or "")) or normalize_space(" ".join(points[:2]))
    row["public_title"] = clean_title(str(row.get("title") or ""))
    row["public_summary"] = summary
    row["public_answer_points"] = points
    row["public_source_cards"] = public_source_cards(row, points, summary)
    row["public_surface_version"] = "stage20f0"
    row["raw_metadata_removed_from_answer_context"] = True
    return row


def build_projection() -> dict[str, Any]:
    for path in (UPSTREAM_CORPUS, UPSTREAM_INDEX, UPSTREAM_METADATA, UPSTREAM_EMBEDDINGS):
        if not path.exists():
            raise FileNotFoundError(path)
    upstream_rows = read_jsonl(UPSTREAM_CORPUS)
    candidate_rows = [project_row(row) for row in upstream_rows]
    write_jsonl(CORPUS_FILE, candidate_rows)
    for source, target in (
        (UPSTREAM_INDEX, INDEX_FILE),
        (UPSTREAM_METADATA, METADATA_FILE),
        (UPSTREAM_EMBEDDINGS, EMBEDDINGS_FILE),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    invariant = {
        "stage": STAGE,
        "generated_at": GENERATED_AT,
        "upstream_record_count": len(upstream_rows),
        "candidate_record_count": len(candidate_rows),
        "record_order_preserved": [row.get("canonical_id") for row in upstream_rows] == [row.get("canonical_id") for row in candidate_rows],
        "title_sequence_unchanged": sequence_sha256(upstream_rows, "title") == sequence_sha256(candidate_rows, "title"),
        "text_for_embedding_sequence_unchanged": sequence_sha256(upstream_rows, "text_for_embedding") == sequence_sha256(candidate_rows, "text_for_embedding"),
        "upstream_index_sha256": file_sha256(UPSTREAM_INDEX),
        "candidate_index_sha256": file_sha256(INDEX_FILE),
        "index_binary_identical": file_sha256(UPSTREAM_INDEX) == file_sha256(INDEX_FILE),
        "upstream_metadata_sha256": file_sha256(UPSTREAM_METADATA),
        "candidate_metadata_sha256": file_sha256(METADATA_FILE),
        "metadata_binary_identical": file_sha256(UPSTREAM_METADATA) == file_sha256(METADATA_FILE),
        "embedding_cache_rebuilt": False,
        "retrieval_fields_modified": False,
    }
    invariant["pass"] = all(invariant[key] for key in (
        "record_order_preserved",
        "title_sequence_unchanged",
        "text_for_embedding_sequence_unchanged",
        "index_binary_identical",
        "metadata_binary_identical",
    ))
    if not invariant["pass"]:
        raise RuntimeError("Stage20F0 retrieval invariant failed; index cannot be reused.")
    answer_point_cleaned = sum(
        1 for before, after in zip(upstream_rows, candidate_rows)
        if before.get("answer_points") != after.get("public_answer_points")
    )
    source_cards_cleaned = sum(
        1 for before, after in zip(upstream_rows, candidate_rows)
        if before.get("source_cards") != after.get("public_source_cards")
    )
    summary = {
        "stage": STAGE,
        "generated_at": GENERATED_AT,
        "profile": "stage20f0_local_style_candidate",
        "upstream_profile": "stage20d3_candidate",
        "records_projected": len(candidate_rows),
        "records_with_public_answer_point_cleanup": answer_point_cleaned,
        "records_with_public_source_card_cleanup": source_cards_cleaned,
        "embedding_cache_rebuilt": False,
        "index_rebuilt": False,
        "retrieval_invariant_pass": invariant["pass"],
        "candidate_corpus_path": str(CORPUS_FILE),
        "candidate_index_path": str(INDEX_FILE),
        "policy": "Retrieval text remains unchanged; only public answer and source-card fields are rendered.",
    }
    write_json(INVARIANT_FILE, invariant)
    write_json(SUMMARY_FILE, summary)
    return summary


def _selected_rows(base_render: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("canonical_id") or ""): row for row in results}
    selected: list[dict[str, Any]] = []
    for citation in base_render.get("citations") or []:
        row = by_id.get(str(citation.get("canonical_id") or ""))
        if row is not None and row not in selected:
            selected.append(row)
    if selected:
        return selected
    return [
        row for row in results
        if row.get("answer_permission") == "direct" and row.get("citation_ready")
    ][:3]


def _citations(rows: list[dict[str, Any]], max_items: int = 5) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cards = row.get("public_source_cards") or public_source_cards(
            row, row.get("public_answer_points") or clean_points(row), row.get("public_summary") or ""
        )
        for card in cards:
            key = (str(card.get("source_id") or ""), str(card.get("source_doc_id") or ""))
            if not key[0] or key in seen:
                continue
            seen.add(key)
            citations.append({
                "marker": f"[{len(citations) + 1}]",
                "title": card.get("source_title") or row.get("public_title") or clean_title(row.get("title", "")),
                "source_id": card.get("source_id"),
                "snippet": card.get("source_excerpt"),
                "doc_id": card.get("source_doc_id") or row.get("canonical_id"),
                "canonical_id": row.get("canonical_id"),
                "evidence_tier": row.get("evidence_tier"),
                "direct_evidence_pass": bool(card.get("direct_evidence_pass")),
            })
            if len(citations) >= max_items:
                return citations
    return citations


def _display_points(row: dict[str, Any]) -> list[str]:
    points = row.get("public_answer_points") or clean_points(row)
    primary = [point for point in points if not SUPPORT_ONLY_POINT_RE.search(str(point))]
    return primary or points


def _point_lines(rows: list[dict[str, Any]], citations: list[dict[str, Any]], limit: int) -> list[str]:
    lines: list[str] = []
    for index, row in enumerate(rows):
        marker = citations[min(index, len(citations) - 1)]["marker"] if citations else ""
        points = _display_points(row)
        for point in points:
            point = clean_text(point)
            if point and point not in lines:
                lines.append(f"{point} {marker}".rstrip())
                break
        if len(lines) >= limit:
            break
    return lines


def _timeline_point_lines(rows: list[dict[str, Any]], citations: list[dict[str, Any]], limit: int) -> list[str]:
    lines: list[str] = []
    for index, row in enumerate(rows):
        marker = citations[min(index, len(citations) - 1)]["marker"] if citations else ""
        points = _display_points(row)
        for point in points[:3]:
            point = clean_text(point)
            rendered = f"{point} {marker}".rstrip()
            if point and rendered not in lines:
                lines.append(rendered)
            if len(lines) >= limit:
                return lines
    return lines


def _clean_fallback(base_render: dict[str, Any]) -> dict[str, Any]:
    answer = clean_text(str(base_render.get("answer") or ""))
    if not answer:
        answer = "Tôi chưa tìm thấy nguồn trực tiếp đủ phù hợp để trả lời câu hỏi này."
    return {
        "answer": answer,
        "citations": [],
        "answer_policy": str(base_render.get("answer_policy") or "stage20f0_insufficient_guard").replace("v16_", "stage20f0_"),
        "public_evidence_renderer": True,
        "style_postcheck": {"forbidden_public_language_count": 0, "passed": True},
    }


def render_unified_answer(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    base_render = _BASE.render_unified_answer(query, results)
    if not base_render.get("citations"):
        return _clean_fallback(base_render)
    intent = query_intent(query)
    selected = _selected_rows(base_render, results)
    citations = _citations(selected, max_items=6 if intent in {"year_timeline", "period_timeline"} else 3)
    if not citations:
        return _clean_fallback(base_render)
    title = str(selected[0].get("public_title") or clean_title(str(selected[0].get("title") or "")))
    points = _display_points(selected[0])
    marker = citations[0]["marker"]
    if intent == "citation_source":
        answer = "Nguồn được truy xuất cho câu hỏi này:\n" + "\n".join(
            f"- {citation['marker']} {citation['title']}: {citation['snippet']}" for citation in citations[:3]
        )
    elif intent == "year_timeline":
        year_match = re.search(r"\b(19[3-7]\d)\b", query)
        label = year_match.group(1) if year_match else "được hỏi"
        lines = _timeline_point_lines(selected, citations, 8)
        answer = f"Năm {label}, các mốc nổi bật gồm:\n" + "\n".join(f"- {line}" for line in lines)
    elif intent == "period_timeline":
        period = str(selected[0].get("period") or "được hỏi")
        lines = _timeline_point_lines(selected, citations, 8)
        answer = f"Trong giai đoạn {period}, các mốc chính gồm:\n" + "\n".join(f"- {line}" for line in lines)
    elif intent == "exact_date_lookup":
        sentence = " ".join(points[:4]) or str(selected[0].get("public_summary") or "")
        answer = f"{sentence.rstrip('.')} {marker}."
    elif intent == "comparison":
        meaningful = [point for point in points if not PLACEHOLDER_COMPARISON_RE.search(point)]
        if meaningful:
            first = citations[0]["marker"] if citations else ""
            second = citations[2]["marker"] if len(citations) >= 3 else (citations[-1]["marker"] if citations else "")
            both = "".join(value for value in (first, second) if value)
            markers = [first, second, both]
            answer = "So sánh theo nguồn được truy xuất:\n" + "\n".join(
                f"- {point} {markers[min(index, len(markers) - 1)]}".rstrip()
                for index, point in enumerate(meaningful[:3])
            )
        else:
            answer = (
                f"Nguồn truy xuất có đề cập đến {title}, nhưng chưa đủ chi tiết để trình bày "
                f"đầy đủ các điểm giống và khác nhau {marker}."
            )
    elif intent == "meaning":
        sentence = " ".join(points[:4]) or str(selected[0].get("public_summary") or "")
        impact_words = r"(y nghia|vai tro|gop phan|mo ra|khang dinh|danh dau|tao tien de|cham dut|co vu)"
        if re.search(impact_words, fold(sentence), re.IGNORECASE):
            answer = f"{sentence} {marker}".strip()
        else:
            answer = (
                f"Về {title}, nguồn hiện xác nhận: {sentence.rstrip('.')} {marker}. "
                "Nguồn truy xuất chưa nêu trực tiếp một đánh giá đầy đủ về ý nghĩa lịch sử."
            )
    elif intent == "topic_overview":
        sentence = " ".join(points[:4]) or str(selected[0].get("public_summary") or "")
        answer = f"{title}: {sentence} {marker}".strip()
    else:
        sentence = " ".join(points[:4]) or str(selected[0].get("public_summary") or "")
        answer = f"{sentence} {marker}".strip()
    answer = re.sub(r"\s+([.,])", r"\1", answer)
    answer = re.sub(r"(\[[0-9]+\])(?:\s*\1)+", r"\1", answer)
    leakage = FORBIDDEN_PUBLIC_RE.findall(answer)
    return {
        "answer": answer,
        "citations": citations,
        "answer_policy": f"stage20f0_natural_{intent}_template",
        "public_evidence_renderer": True,
        "style_postcheck": {
            "forbidden_public_language_count": len(leakage),
            "passed": len(leakage) == 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the opt-in public evidence candidate.")
    args = parser.parse_args()
    if args.build:
        summary = build_projection()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

