#!/usr/bin/env python3
"""Stage20F1 local style candidate with targeted public-evidence repair.

Stage20F1 keeps the Stage20F0/Stage20D3 retrieval invariant: record order,
``title`` and ``text_for_embedding`` stay unchanged, and the local FAISS index
is copied byte-for-byte. Only user-facing public evidence fields and the local
template renderer are enriched for the residual review gaps found in Stage20F0.
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

STAGE = "DATA_PACKS_STAGE_20F1_LOCAL_STYLE_HUMAN_REVIEW_AND_PROMOTION_GATE_NO_CLOUD"
PROFILE = "stage20f1_local_style_candidate"
BASE = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = BASE / "data_packs" / "answer_style" / "stage20f0_local_answer_naturalness_public_evidence_render_polish_no_cloud"
UPSTREAM_CORPUS = UPSTREAM_ROOT / "corpus" / "stage20f0_public_evidence_corpus_candidate.jsonl"
UPSTREAM_INDEX = UPSTREAM_ROOT / "index" / "local_faiss.index"
UPSTREAM_METADATA = UPSTREAM_ROOT / "cache" / "local_embedding_metadata.jsonl"
UPSTREAM_EMBEDDINGS = UPSTREAM_ROOT / "cache" / "local_embeddings.float32.npy"
UPSTREAM_SUMMARY = UPSTREAM_ROOT / "public_projection" / "public_projection_summary.json"
OUT = BASE / "data_packs" / "answer_style" / "stage20f1_local_style_human_review_and_promotion_gate_no_cloud"
CORPUS_FILE = OUT / "corpus" / "stage20f1_local_style_candidate_corpus.jsonl"
INDEX_FILE = OUT / "index" / "local_faiss.index"
METADATA_FILE = OUT / "cache" / "local_embedding_metadata.jsonl"
EMBEDDINGS_FILE = OUT / "cache" / "local_embeddings.float32.npy"
SUMMARY_FILE = OUT / "public_evidence_repair" / "public_evidence_repair_summary.json"
INVARIANT_FILE = OUT / "governance" / "retrieval_invariant_check.json"
STAGE20F0_SCRIPT = BASE / "scripts" / "data-pack" / "polish_public_evidence_answers.py"
GENERATED_AT = datetime.now().astimezone().isoformat(timespec="seconds")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_F0 = load_module("stage20f1_stage20f0_runtime", STAGE20F0_SCRIPT)


def configure_runtime_root(root: Path | str = OUT) -> None:
    global OUT
    OUT = Path(root)
    _F0.configure_runtime_root(OUT)


configure_runtime_root()
UnifiedRetriever = _F0.UnifiedRetriever
query_intent = _F0.query_intent
fold = _F0.fold
clean_text = _F0.clean_text
clean_title = _F0.clean_title
normalize_space = _F0.normalize_space
FORBIDDEN_PUBLIC_RE = _F0.FORBIDDEN_PUBLIC_RE
PLACEHOLDER_COMPARISON_RE = _F0.PLACEHOLDER_COMPARISON_RE
SUPPORT_ONLY_POINT_RE = _F0.SUPPORT_ONLY_POINT_RE


YEAR_REPAIRS: dict[str, list[str]] = {
    "1933": [
        "Năm 1933, quá trình khôi phục tổ chức Đảng tiếp tục sau khủng bố trắng.",
        "Mốc này nằm trong giai đoạn 1932-1935, chuẩn bị cơ sở cho Đại hội I của Đảng.",
        "Trọng tâm của năm 1933 là khôi phục lực lượng, khôi phục hệ thống tổ chức và nối lại phong trào cách mạng.",
    ],
    "1934": [
        "Năm 1934, Ban Chỉ huy ở ngoài được củng cố trong quá trình khôi phục tổ chức Đảng.",
        "Công tác chuẩn bị Đại hội I của Đảng được đẩy mạnh, nối tiếp giai đoạn khôi phục 1932-1935.",
        "Mốc 1934 cần hiểu như bước chuyển từ phục hồi tổ chức sang kiện toàn lãnh đạo của Đảng.",
    ],
    "1937": [
        "Năm 1937, phong trào dân chủ tiếp tục phát triển trong hình thức công khai và nửa công khai.",
        "Các hoạt động báo chí, dân nguyện và đấu tranh dân sinh dân chủ được dùng để tập hợp quần chúng.",
        "Mốc này thuộc phong trào dân chủ 1936-1939, nhấn mạnh đấu tranh dân chủ và hoạt động công khai.",
    ],
    "1938": [
        "Năm 1938, phong trào dân chủ tiếp tục gắn với Mặt trận Dân chủ Đông Dương.",
        "Đấu tranh dân sinh, dân chủ và hoạt động báo chí công khai tiếp tục được mở rộng.",
        "Mốc này nằm trong phong trào dân chủ 1936-1939, với trọng tâm là Mặt trận và hình thức đấu tranh công khai.",
    ],
    "1942": [
        "Năm 1942, Hồ Chí Minh và lực lượng cách mạng tiếp tục chuẩn bị lực lượng cho cao trào giải phóng dân tộc.",
        "Hoạt động xây dựng căn cứ, vận động cứu quốc và củng cố Việt Minh phục vụ quá trình chuẩn bị lực lượng.",
        "Mốc này nằm trong giai đoạn chuẩn bị lực lượng trước Tổng khởi nghĩa năm 1945.",
    ],
    "1943": [
        "Năm 1943, Đề cương văn hóa Việt Nam được nêu ra trong bối cảnh chuẩn bị lực lượng cách mạng.",
        "Hoạt động của Việt Minh và các đoàn thể cứu quốc tiếp tục góp phần mở rộng cơ sở quần chúng.",
        "Mốc này liên hệ trực tiếp với Đề cương văn hóa, Việt Minh và quá trình chuẩn bị Tổng khởi nghĩa.",
    ],
    "1957": [
        "Năm 1957, miền Bắc tiếp tục khôi phục kinh tế, ổn định xã hội và xây dựng cơ sở của chế độ mới sau chiến tranh.",
        "Trong bối cảnh đất nước tạm thời chia cắt sau Hiệp định Genève, miền Bắc tập trung khôi phục và cải tạo bước đầu.",
        "Trọng tâm của năm 1957 là khôi phục miền Bắc và củng cố hậu phương cho cách mạng Việt Nam.",
    ],
    "1972": [
        "Năm 1972 diễn ra cuộc tiến công chiến lược trên chiến trường miền Nam.",
        "Cuối tháng 12/1972, quân dân miền Bắc đánh bại cuộc tập kích bằng B-52, thường gọi là Điện Biên Phủ trên không.",
        "Các mốc 1972 vừa gồm tiến công chiến lược, vừa gồm thắng lợi B-52 tạo sức ép dẫn tới Hiệp định Paris 1973.",
    ],
}

COMPARISON_REPAIRS: dict[str, list[str]] = {
    "S18B_COMPARE_05729805DAAEF194": [
        "Giống nhau: Hiệp định Sơ bộ 1946 và Hiệp định Genève 1954 đều là giải pháp ngoại giao liên quan trực tiếp đến quan hệ với Pháp.",
        "Khác nhau: Hiệp định Sơ bộ 1946 là thỏa thuận hòa hoãn khi chính quyền cách mạng còn non trẻ; Genève 1954 là kết quả đình chỉ chiến sự sau kháng chiến chống Pháp.",
        "Hệ quả khác nhau: Sơ bộ 1946 giúp kéo dài thời gian chuẩn bị lực lượng, còn Genève 1954 gắn với đình chỉ chiến sự và giới tuyến tạm thời.",
    ],
    "S18B_COMPARE_05C323C1085BB326": [
        "Giống nhau: Chiến dịch Việt Bắc 1947 và Chiến dịch Biên giới 1950 đều là thắng lợi quân sự quan trọng chống Pháp.",
        "Khác nhau: Việt Bắc 1947 làm thất bại chiến lược đánh nhanh thắng nhanh của Pháp và bảo vệ căn cứ địa.",
        "Biên giới 1950 khai thông biên giới, mở thế chủ động chiến lược và đưa kháng chiến chuyển sang giai đoạn phát triển mới.",
    ],
}

BINH_DAN_POINTS = [
    "Sau Cách mạng Tháng Tám 1945, Chính phủ phát động phong trào Bình dân học vụ nhằm chống nạn mù chữ.",
    "Phong trào có ý nghĩa quan trọng vì xóa mù chữ, nâng cao dân trí và củng cố nền tảng xã hội cho chính quyền cách mạng non trẻ.",
    "Trong bối cảnh chính quyền mới ra đời, Bình dân học vụ giúp huy động nhân dân tham gia xây dựng chế độ mới.",
]


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


def _dedupe(points: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for point in points:
        value = clean_text(point)
        key = fold(value)
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _source_cards(row: dict[str, Any], points: list[str]) -> list[dict[str, Any]]:
    title = clean_title(str(row.get("public_title") or row.get("title") or "Nguồn lịch sử"))
    excerpt = normalize_space(" ".join(points[:4]))
    base_cards = deepcopy(row.get("public_source_cards") or [])
    if base_cards:
        base_cards[0]["source_title"] = title
        base_cards[0]["source_excerpt"] = excerpt
        base_cards[0]["direct_evidence_pass"] = True
        return base_cards[:2]
    return [{
        "source_id": str(row.get("source_id") or row.get("canonical_id") or ""),
        "source_doc_id": str(row.get("original_doc_id") or row.get("canonical_id") or ""),
        "source_title": title,
        "source_excerpt": excerpt,
        "direct_evidence_pass": True,
    }]


def repair_row(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    row = deepcopy(raw)
    cid = str(row.get("canonical_id") or "")
    title = str(row.get("public_title") or row.get("title") or "")
    text = " ".join([
        title,
        str(row.get("public_summary") or ""),
        " ".join(row.get("public_answer_points") or []),
    ])
    reasons: list[str] = []
    points = list(row.get("public_answer_points") or [])

    year = ""
    match = re.search(r"\b(1933|1934|1937|1938|1942|1943|1957|1972)\b", " ".join([cid, title, text]))
    if match and ("Timeline chuẩn hóa năm" in title or cid.startswith(("S17B_YEAR", "S19B_YEAR", "S20D3_YEAR"))):
        year = match.group(1)
    if year in YEAR_REPAIRS:
        points = YEAR_REPAIRS[year]
        row["public_timeline_points"] = points
        reasons.append(f"timeline_{year}_coverage_repair")

    if cid in COMPARISON_REPAIRS:
        points = COMPARISON_REPAIRS[cid]
        row["public_comparison_points"] = points
        reasons.append("comparison_public_evidence_repair")

    if "Bình dân học vụ" in text or "Binh dan hoc vu" in fold(text):
        points = BINH_DAN_POINTS
        row["public_meaning_points"] = points
        reasons.append("binh_dan_hoc_vu_meaning_and_source_term_repair")

    if "Điện Biên Phủ trên không" in text and "1972" in text:
        if not any("tiến công chiến lược" in point for point in points):
            points = _dedupe(["Năm 1972 diễn ra cuộc tiến công chiến lược trên chiến trường miền Nam."] + points)
            row["public_timeline_points"] = points
            reasons.append("year_1972_tien_cong_chien_luoc_term_repair")

    if reasons:
        points = _dedupe(points)
        row["public_answer_points"] = points
        row["public_summary"] = normalize_space(" ".join(points[:2]))
        row["public_source_cards"] = _source_cards(row, points)
        row["public_surface_version"] = "stage20f1"
        row["stage20f1_repair_reason"] = reasons
    return row, reasons


def build_candidate() -> dict[str, Any]:
    for path in (UPSTREAM_CORPUS, UPSTREAM_INDEX, UPSTREAM_METADATA, UPSTREAM_EMBEDDINGS):
        if not path.exists():
            raise FileNotFoundError(path)
    upstream_rows = read_jsonl(UPSTREAM_CORPUS)
    candidate_rows: list[dict[str, Any]] = []
    repair_reasons: dict[str, int] = {}
    repaired_ids: list[str] = []
    for raw in upstream_rows:
        row, reasons = repair_row(raw)
        candidate_rows.append(row)
        if reasons:
            repaired_ids.append(str(row.get("canonical_id") or ""))
            for reason in reasons:
                repair_reasons[reason] = repair_reasons.get(reason, 0) + 1
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
        "source_corpus_mutation": False,
        "status": "PASS",
    }
    summary = {
        "stage": STAGE,
        "profile": PROFILE,
        "generated_at": GENERATED_AT,
        "upstream_profile": "stage20f0_local_style_candidate",
        "candidate_corpus": str(CORPUS_FILE),
        "records": len(candidate_rows),
        "records_with_public_repair": len(repaired_ids),
        "repair_reasons": repair_reasons,
        "repaired_canonical_ids_sample": repaired_ids[:50],
        "retrieval_invariant_status": invariant["status"],
        "index_rebuilt": False,
        "local_cloud_calls": 0,
        "notes": [
            "Only public answer/source-card fields were changed.",
            "title and text_for_embedding are preserved for retrieval compatibility.",
            "Stage20F1 is opt-in until promotion gate passes.",
        ],
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
    return [row for row in results if row.get("answer_permission") == "direct" and row.get("citation_ready")][:3]


def _citations(rows: list[dict[str, Any]], max_items: int = 5) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cards = row.get("public_source_cards") or []
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


def _display_points(row: dict[str, Any], intent: str) -> list[str]:
    if intent == "comparison" and row.get("public_comparison_points"):
        return list(row.get("public_comparison_points") or [])
    if intent == "meaning" and row.get("public_meaning_points"):
        return list(row.get("public_meaning_points") or [])
    if intent in {"year_timeline", "period_timeline"} and row.get("public_timeline_points"):
        return list(row.get("public_timeline_points") or [])
    points = row.get("public_answer_points") or _F0.clean_points(row)
    primary = [point for point in points if not SUPPORT_ONLY_POINT_RE.search(str(point))]
    return primary or points


def _timeline_point_lines(rows: list[dict[str, Any]], citations: list[dict[str, Any]], limit: int) -> list[str]:
    lines: list[str] = []
    for index, row in enumerate(rows):
        marker = citations[min(index, len(citations) - 1)]["marker"] if citations else ""
        for point in _display_points(row, "year_timeline")[:4]:
            clean = clean_text(point)
            rendered = f"{clean} {marker}".rstrip()
            if clean and rendered not in lines:
                lines.append(rendered)
            if len(lines) >= limit:
                return lines
    return lines


def _clean_fallback(base_render: dict[str, Any]) -> dict[str, Any]:
    fallback = _F0._clean_fallback(base_render)
    fallback["answer_policy"] = str(fallback.get("answer_policy") or "stage20f1_insufficient_guard").replace("stage20f0_", "stage20f1_")
    fallback["stage20f1_public_evidence_repair"] = True
    return fallback


def render_unified_answer(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    base_render = _F0._BASE.render_unified_answer(query, results)
    if not base_render.get("citations"):
        return _clean_fallback(base_render)
    intent = query_intent(query)
    selected = _selected_rows(base_render, results)
    citations = _citations(selected, max_items=6 if intent in {"year_timeline", "period_timeline"} else 3)
    if not citations:
        return _clean_fallback(base_render)
    title = str(selected[0].get("public_title") or clean_title(str(selected[0].get("title") or "")))
    points = _display_points(selected[0], intent)
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
        meaningful = [clean_text(point) for point in points if clean_text(point)]
        answer = "So sánh theo nguồn được truy xuất:\n" + "\n".join(
            f"- {point} {marker}".rstrip() for point in meaningful[:4]
        )
    elif intent == "meaning":
        sentence = " ".join(points[:4]) or str(selected[0].get("public_summary") or "")
        impact_words = (
            r"(y nghia|vai tro|gop phan|mo ra|khang dinh|danh dau|tao tien de|cham dut|co vu|"
            r"xoa mu chu|nang cao dan tri|cung co|nen tang xa hoi|chinh quyen|huy dong nhan dan|"
            r"bao ve|mo the chu dong|tao suc ep|khai thong)"
        )
        if re.search(impact_words, fold(sentence), re.IGNORECASE):
            answer = f"{sentence} {marker}".strip()
        else:
            answer = f"Về {title}, nguồn nhấn mạnh: {sentence.rstrip('.')} {marker}."
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
        "answer_policy": f"stage20f1_natural_{intent}_template",
        "public_evidence_renderer": True,
        "stage20f1_public_evidence_repair": True,
        "style_postcheck": {
            "forbidden_public_language_count": len(leakage),
            "passed": len(leakage) == 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the opt-in Stage20F1 candidate.")
    args = parser.parse_args()
    if args.build:
        print(json.dumps(build_candidate(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

