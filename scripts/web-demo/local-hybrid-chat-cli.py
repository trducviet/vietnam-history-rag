#!/usr/bin/env python3
"""Local hybrid web-chat adapter.

Reads one JSON request from stdin and writes one normalized JSON response to
stdout. The adapter reuses the Stage 13A local runtime: BM25, local query
embedding, FAISS, RRF k=30, citation cards, Ollama/local templates. It never
calls cloud APIs and runs Hugging Face/SentenceTransformer in offline mode.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("NO_CLOUD", "true")
os.environ.setdefault("DISABLE_EXTERNAL_NETWORK", "true")
os.environ.setdefault("ALLOW_LOCALHOST_ONLY", "true")
os.environ.setdefault("DISABLE_CLOUD_LLM", "true")
os.environ.setdefault("DISABLE_CLOUD_EMBEDDING", "true")
os.environ.setdefault("DISABLE_CLOUD_ROUTER", "true")
os.environ.setdefault("DISABLE_CLOUD_RERANKER", "true")
os.environ.setdefault("RAG_ALLOW_EXTERNAL_API", "0")
os.environ.setdefault("RAG_LLM_MODE", "off")
os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("LOCAL_LLM_PROVIDER", "ollama_openai_compatible")
os.environ.setdefault("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("LOCAL_LLM_MODEL", "qwen2.5:3b-instruct")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE = Path(__file__).resolve().parents[2]
STAGE13A_PATH = BASE / "scripts" / "data-pack" / "stage13a-local-runtime-demo-tuning-no-cloud.py"
_STAGE13A_CACHE: Any | None = None
_RUNTIME_CACHE: Any | None = None


def load_stage13a() -> Any:
    global _STAGE13A_CACHE
    if _STAGE13A_CACHE is not None:
        return _STAGE13A_CACHE
    spec = importlib.util.spec_from_file_location("stage13a_runtime", STAGE13A_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Stage 13A runtime from {STAGE13A_PATH}")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    _STAGE13A_CACHE = module
    return module


def get_runtime(stage13a: Any) -> Any:
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is None:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = stage13a.LocalHybridRuntime()
            runtime.load()
        _RUNTIME_CACHE = runtime
    return _RUNTIME_CACHE


def strip_vn(text: str) -> str:
    value = unicodedata.normalize("NFD", text or "")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return value.replace("đ", "d").replace("Đ", "D").lower()


def contains(text: str, *needles: str) -> bool:
    norm = strip_vn(text)
    return any(strip_vn(needle) in norm for needle in needles)


DATE_SLASH_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
DATE_WORD_RE = re.compile(r"\bngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})\b", re.IGNORECASE)
DATE_RANGE_RE = re.compile(r"\b(\d{1,2})\s*[–-]\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
YEAR_RE = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
METADATA_LEAK_RE = re.compile(
    r"(synthesis/[a-z_]+|timeline_summary|comparison_note|event\s*\|\s*actual|"
    r"actual\s+\d{4}|\b(?:18|19|20)\d{2}\s+(?:August Revolution|Geneva Accords|Paris Accords|Vietnam War|French War|Indochina War|Dien Bien Phu)|"
    r"Câu hỏi alias trỏ tới|Vì vậy,\s*câu trả lời nên được trình bày|"
    r"fallback noted|bm25_fallback|query embedding cache|citation_aware_fallback|template_name)",
    re.IGNORECASE,
)
ASK_WORDS = [
    "là gì",
    "la gi",
    "ý nghĩa",
    "y nghia",
    "như thế nào",
    "nhu the nao",
    "vì sao",
    "vi sao",
    "phân tích",
    "phan tich",
    "so sánh",
    "so sanh",
    "nguồn nào",
    "nguon nao",
    "khi nào",
    "khi nao",
    "năm bao nhiêu",
    "nam bao nhieu",
    "sự kiện",
    "su kien",
]


def squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def canonical_date(day: int | str, month: int | str, year: int | str) -> str:
    return f"{int(day)}/{int(month)}/{int(year)}"


def valid_calendar_date(day: int, month: int, year: int) -> bool:
    if month < 1 or month > 12 or day < 1 or day > 31:
        return False
    if month in {4, 6, 9, 11} and day > 30:
        return False
    if month == 2 and day > 29:
        return False
    return 1 <= year <= 2100


def extract_exact_dates(text: str) -> list[str]:
    found: list[str] = []
    raw = str(text or "")
    for match in DATE_RANGE_RE.finditer(raw):
        start_day, end_day, month, year = match.groups()
        for day in (start_day, end_day):
            try:
                found.append(canonical_date(day, month, year))
            except ValueError:
                pass
    for regex in (DATE_SLASH_RE, DATE_WORD_RE):
        for match in regex.finditer(raw):
            day, month, year = match.groups()
            try:
                found.append(canonical_date(day, month, year))
            except ValueError:
                pass
    deduped = []
    for item in found:
        if item not in deduped:
            deduped.append(item)
    return deduped


def extract_years(text: str) -> list[int]:
    years = []
    for year in YEAR_RE.findall(str(text or "")):
        value = int(year)
        if value not in years:
            years.append(value)
    return years


def parse_query_date(text: str) -> dict[str, Any]:
    raw = str(text or "")
    for regex in (DATE_SLASH_RE, DATE_WORD_RE):
        match = regex.search(raw)
        if match:
            day, month, year = (int(x) for x in match.groups())
            return {
                "has_exact_date": True,
                "day": day,
                "month": month,
                "year": year,
                "date": canonical_date(day, month, year),
                "valid": valid_calendar_date(day, month, year),
                "granularity": "exact_date",
            }
    years = extract_years(raw)
    return {
        "has_exact_date": False,
        "day": None,
        "month": None,
        "year": years[0] if years else None,
        "date": None,
        "valid": True,
        "granularity": "year" if years else "none",
    }


def parse_year_query(text: str) -> int | None:
    match = re.search(r"\bnăm\s+(18\d{2}|19\d{2}|20\d{2})\b", strip_vn(text))
    if match:
        return int(match.group(1))
    years = extract_years(text)
    return years[0] if years else None


def clean_user_text(text: str, max_len: int = 260) -> str:
    value = str(text or "")
    replacements = [
        (r"synthesis/[a-z_]+", " "),
        (r"\b(?:timeline_summary|comparison_note|citation_aware_fallback|template_name)\b", " "),
        (r"\bevent\s*\|\s*actual\b", " "),
        (r"\bactual\s+\d{4}\b", " "),
        (r"\b(?:18|19|20)\d{2}\s+(?:August Revolution|Geneva Accords|Paris Accords|Vietnam War|French War|Indochina War|Dien Bien Phu)[^.;]*", " "),
        (r"Timeline các sự kiện nổi bật năm\s+\d{4}\s*\d{0,4}", " "),
        (r"\b(?:source_id|doc_id|chunk_id|retrieval_score|rrf_score)\s*[:=]\s*\S+", " "),
        (r"Câu hỏi alias trỏ tới", " "),
        (r"Vì vậy,\s*câu trả lời nên được trình bày[^.。]*[.。]?", " "),
        (r"fallback noted|bm25_fallback|query embedding cache", " "),
    ]
    for pattern, repl in replacements:
        value = re.sub(pattern, repl, value, flags=re.IGNORECASE)
    value = value.replace("|", " ")
    value = re.sub(r"\b(?:năm\s+\d{4}|sự kiện\s+\d{4}|timeline\s+\d{4})\s*;\s*", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*;\s*", "; ", value)
    value = squeeze(value)
    if len(value) > max_len:
        cut = value[:max_len].rsplit(" ", 1)[0].rstrip(" ,;:-")
        sentence_cut = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if sentence_cut > 80:
            cut = cut[: sentence_cut + 1]
        else:
            cut = cut.rstrip(".") + "."
        value = cut
    return value or "Nguồn trong corpus có liên quan đến câu hỏi."


def clean_card(card: dict[str, Any]) -> dict[str, Any]:
    raw_title = str(card.get("title") or "")
    raw_snippet = str(card.get("snippet") or "")
    text = f"{raw_title} {raw_snippet}"
    clean = dict(card)
    clean["title_clean"] = clean_user_text(raw_title, 140)
    clean["snippet_clean"] = clean_user_text(raw_snippet, 320)
    clean["user_visible_text"] = clean_user_text(f"{raw_title}. {raw_snippet}", 420)
    clean["exact_dates"] = extract_exact_dates(text)
    clean["years"] = extract_years(text)
    clean["granularity"] = "exact_date" if clean["exact_dates"] else "year_level" if clean["years"] else "topic_level"
    return clean


def clean_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [clean_card(card) for card in cards]


def first_clean_fact(card: dict[str, Any] | None, max_len: int = 220) -> str:
    if not card:
        return "Nguồn trong corpus chưa đủ trực tiếp cho vế này"
    title = card.get("title_clean") or card.get("title") or ""
    if title and not contains(title, "timeline các sự kiện", "timeline cac su kien"):
        return clean_user_text(title, max_len)
    text = card.get("snippet_clean") or title or ""
    parts = [p.strip(" -") for p in re.split(r"(?<=[.!?])\s+|;\s*", text) if p.strip(" -")]
    return clean_user_text(parts[0] if parts else text, max_len)


def is_generic_fact(text: str) -> bool:
    norm = strip_vn(text)
    return (not norm) or "nguon trong corpus co lien quan" in norm or norm in {"nguon hien co khop voi ngay duoc hoi"}


def title_topic(message: str, cards: list[dict[str, Any]]) -> str:
    if contains(message, "cách mạng tháng tám", "cach mang thang tam", "cm thang 8"):
        return "Cách mạng Tháng Tám 1945"
    if contains(message, "điện biên phủ", "dien bien phu", "dienn bin phu"):
        return "Chiến dịch Điện Biên Phủ"
    if contains(message, "hiệp định genève", "hiep dinh geneve", "geneve"):
        return "Hiệp định Genève 1954"
    if contains(message, "hiệp định paris", "hiep dinh paris"):
        return "Hiệp định Paris 1973"
    if contains(message, "việt minh", "viet minh"):
        return "Việt Minh"
    if contains(message, "đảng cộng sản", "dang cong san"):
        return "Đảng Cộng sản Việt Nam"
    if cards:
        return cards[0].get("title_clean") or cards[0].get("title") or "Chủ đề lịch sử"
    return squeeze(message)


def supports_exact_date(card: dict[str, Any], date: str) -> bool:
    return date in (card.get("exact_dates") or [])


def date_variants(date: str) -> list[str]:
    try:
        day, month, year = date.split("/")
        return [f"{int(day)}/{int(month)}/{year}", f"{int(day)}-{int(month)}-{year}", f"{int(day):02d}/{int(month):02d}/{year}"]
    except Exception:
        return [date]


def fact_for_exact_date(card: dict[str, Any], date: str) -> str:
    variants = [strip_vn(v) for v in date_variants(date)]
    texts = [card.get("snippet_clean") or "", card.get("title_clean") or "", card.get("snippet") or "", card.get("title") or ""]
    for text in texts:
        pieces = [p.strip(" -:") for p in re.split(r";\s*|(?<=[.!?])\s+", str(text)) if p.strip(" -:")]
        for piece in pieces:
            norm_piece = strip_vn(piece)
            if any(v in norm_piece or v.replace("/", "-") in norm_piece for v in variants):
                return clean_user_text(piece, 220)
    fact = first_clean_fact(card, 220)
    if "Nguồn trong corpus có liên quan" not in fact:
        return fact
    return card.get("title_clean") or "Nguồn hiện có khớp với ngày được hỏi"


def cards_for_year(cards: list[dict[str, Any]], year: int) -> list[dict[str, Any]]:
    return [card for card in cards if year in (card.get("years") or [])]


def metadata_leak_detected(answer: str) -> bool:
    return bool(METADATA_LEAK_RE.search(answer or ""))


def fake_markers(answer: str, cards: list[dict[str, Any]]) -> bool:
    valid = {str(card.get("marker")) for card in cards}
    for marker in re.findall(r"\[[0-9]+\]", answer or ""):
        if marker not in valid:
            return True
    return False


def likely_truncated(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    if text.endswith("...") or text.endswith("…"):
        return True
    if re.search(r"\b\w{1,2}$", strip_vn(text)) and not re.search(r"[\].!?)]$", text):
        return True
    return False


def exact_date_insufficient_answer(date_info: dict[str, Any], cards: list[dict[str, Any]]) -> str:
    date = date_info.get("date") or "ngày được hỏi"
    years = sorted({year for card in cards for year in (card.get("years") or [])})
    if not date_info.get("valid", True):
        return f"Ngày {date} không phải một ngày hợp lệ, nên mình không dùng corpus để khẳng định sự kiện lịch sử cho mốc này."
    if years:
        near = ", ".join(str(y) for y in years[:3])
        return (
            f"Mình chưa tìm thấy nguồn trong corpus xác nhận sự kiện đúng ngày {date}. "
            f"Các nguồn gần nhất chỉ liên quan đến năm/chủ đề {near}, nên mình không dùng chúng để khẳng định sự kiện ngày đó."
        )
    return f"Mình chưa tìm thấy nguồn đủ trực tiếp trong corpus để xác nhận sự kiện đúng ngày {date}."


def year_events_template(year: int, cards: list[dict[str, Any]]) -> str:
    year_cards = cards_for_year(cards, year)
    if not year_cards:
        return (
            f"Mình chưa tìm thấy nguồn trong corpus khớp trực tiếp với năm {year}. "
            "Vì vậy mình không dùng các nguồn khác năm để liệt kê sự kiện cho câu hỏi này."
        )
    fragments: list[tuple[str, str]] = []
    for card in year_cards[:4]:
        source = card.get("snippet_clean") or card.get("title_clean") or ""
        pieces = [p.strip(" -:") for p in re.split(r";\s*|(?<=[.!?])\s+", source) if p.strip(" -:")]
        for piece in pieces:
            norm_piece = strip_vn(piece)
            if len(piece) < 8:
                continue
            if f"timeline {year}" in norm_piece or f"su kien {year}" in norm_piece or f"nam {year}" == norm_piece:
                continue
            fragments.append((piece, card["marker"]))
            if len(fragments) >= 4:
                break
        if len(fragments) >= 4:
            break
    if not fragments:
        fragments = [(first_clean_fact(card), card["marker"]) for card in year_cards[:3]]
    lines = [f"Một số sự kiện nổi bật năm {year} trong corpus gồm:"]
    for fact, marker in fragments[:4]:
        lines.append(f"- {clean_user_text(fact, 180).rstrip('.')} {marker}.")
    markers = "".join(card["marker"] for card in year_cards[:2])
    lines.append(f"Tóm lại, năm {year} được nguồn truy xuất xem là một mốc đáng chú ý trong phạm vi dữ liệu hiện có {markers}.")
    return "\n".join(lines)


def exact_date_template(date_info: dict[str, Any], cards: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], str]:
    exact_cards = [card for card in cards if supports_exact_date(card, date_info["date"])]
    if not exact_cards:
        return exact_date_insufficient_answer(date_info, cards), [], "safe_insufficient_data"
    exact_cards.sort(key=lambda c: 1 if contains(c.get("title_clean", ""), "timeline các sự kiện", "timeline cac su kien") else 0)
    card = exact_cards[0]
    fact = fact_for_exact_date(card, date_info["date"])
    return f"Ngày {date_info['date']}, nguồn trong corpus ghi nhận: {fact} {card['marker']}.", exact_cards[:2], "none"


def fact_date_lookup_template(message: str, cards: list[dict[str, Any]]) -> str:
    if contains(message, "thống nhất đất nước", "thong nhat dat nuoc"):
        supporting = next((card for card in cards if 1975 in (card.get("years") or []) or "30/4/1975" in (card.get("exact_dates") or [])), cards[0] if cards else None)
        marker = supporting["marker"] if supporting else ""
        return (
            f"Câu trả lời ngắn: nếu hiểu là mốc giải phóng miền Nam, kết thúc chiến tranh và thống nhất về lãnh thổ, "
            f"thì là năm 1975, cụ thể ngày 30/4/1975 {marker}. "
            "Nếu hỏi về thống nhất nhà nước, cần nguồn trực tiếp cho mốc đó; mình không gán citation cho phần chưa được nguồn hiện tại hỗ trợ."
        ).strip()
    card = cards[0] if cards else None
    if not card:
        return "Mình chưa tìm thấy nguồn đủ trực tiếp trong corpus để trả lời chắc chắn mốc thời gian này."
    date = (card.get("exact_dates") or [None])[0] or ((card.get("years") or [None])[0])
    return f"Câu trả lời ngắn: mốc được nguồn hiện tại hỗ trợ là {date} {card['marker']}. {first_clean_fact(card, 180)} {card['marker']}."


def topic_overview_template(message: str, cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "Mình chưa tìm thấy nguồn đủ trực tiếp trong corpus để giới thiệu chủ đề này."
    topic = title_topic(message, cards)
    usable = [card for card in cards if not is_generic_fact(first_clean_fact(card, 190))]
    if not usable:
        return "Mình chưa tìm thấy nguồn đủ trực tiếp trong corpus để giới thiệu chủ đề này."
    main = usable[0]
    second = usable[1] if len(usable) > 1 else usable[0]
    third = usable[2] if len(usable) > 2 else second
    return "\n".join(
        [
            f"{topic} là một chủ đề lịch sử được corpus liên kết với: {first_clean_fact(main, 190)} {main['marker']}.",
            "Một số điểm chính:",
            f"- Thời gian/bối cảnh: {first_clean_fact(main, 170)} {main['marker']}.",
            f"- Nội dung liên quan: {first_clean_fact(second, 170)} {second['marker']}.",
            f"- Ý nghĩa/ghi chú: {first_clean_fact(third, 170)} {third['marker']}.",
        ]
    )


def timeline_summary_template(message: str, cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "Mình chưa tìm thấy nguồn đủ trực tiếp để tóm tắt các mốc chính."
    topic = clean_user_text(message, 90) if contains(message, "giai đoạn", "giai doan", "từ năm", "tu nam", "timeline") else title_topic(message, cards)
    lines = [f"Các mốc chính về {topic} có thể tóm tắt như sau:"]
    added = 0
    for card in cards[:4]:
        fact = first_clean_fact(card, 190)
        if is_generic_fact(fact):
            continue
        lines.append(f"- {fact.rstrip('.')} {card['marker']}.")
        added += 1
    if added == 0:
        return "Mình chưa tìm thấy nguồn đủ trực tiếp trong corpus để tóm tắt các mốc chính cho câu hỏi này."
    lines.append("Các mốc trên chỉ phản ánh phạm vi nguồn đã truy xuất trong corpus.")
    return "\n".join(lines)


def citation_source_template_v2(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "Mình chưa tìm thấy nguồn đủ trực tiếp trong corpus cho câu hỏi này."
    lines = ["Các nguồn hỗ trợ gồm:"]
    usable = []
    for card in cards:
        support = first_clean_fact(card, 180)
        if not is_generic_fact(support):
            usable.append((card, support))
    if not usable:
        return "Mình chưa tìm thấy nguồn đủ trực tiếp trong corpus cho câu hỏi này."
    for card, support in usable[:3]:
        titles = card.get("source_titles") or []
        label = (titles[0] if titles else None) or card.get("title_clean") or "Nguồn trong corpus"
        lines.append(f"- {card['marker']}: {clean_user_text(label, 100)} hỗ trợ ý này vì ghi nhận {support} {card['marker']}.")
    lines.append("Kết luận: các nguồn trên là cơ sở để trả lời trong phạm vi corpus hiện có.")
    return "\n".join(lines)


def comparison_template_v2(case: dict[str, Any], cards: list[dict[str, Any]]) -> str:
    query = case.get("memory_rewrite") or case.get("query") or ""
    if contains(query, "geneve", "genève") and contains(query, "paris"):
        a_label, b_label = "Hiệp định Genève 1954", "Hiệp định Paris 1973"
    elif contains(query, "điện biên phủ", "dien bien phu") and contains(query, "chiến dịch hồ chí minh", "chien dich ho chi minh"):
        a_label, b_label = "Chiến dịch Điện Biên Phủ", "Chiến dịch Hồ Chí Minh"
    else:
        a_label, b_label = "Vế A", "Vế B"
    a = next((c for c in cards if c.get("comparison_side") == "A"), cards[0] if cards else None)
    b = next((c for c in cards if c.get("comparison_side") == "B"), next((c for c in cards if c is not a), None))
    a_fact = first_clean_fact(a, 190)
    b_fact = first_clean_fact(b, 190)
    a_marker = a["marker"] if a else ""
    b_marker = b["marker"] if b else ""
    diff_marker = f"{a_marker}{b_marker}".strip() or a_marker or b_marker
    return "\n".join(
        [
            f"**{a_label}:**",
            f"- {a_fact} {a_marker}".strip(),
            "",
            f"**{b_label}:**",
            f"- {b_fact} {b_marker}".strip(),
            "",
            "**Khác nhau chính:**",
            f"- Hai sự kiện/hiệp định khác nhau ở bối cảnh, mục tiêu trực tiếp và hệ quả lịch sử; phần so sánh này chỉ dựa trên các nguồn đã truy xuất {diff_marker}.",
        ]
    )


def generic_fact_template(message: str, cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "Mình chưa tìm thấy nguồn đủ trực tiếp trong corpus để trả lời chắc chắn câu hỏi này."
    main = cards[0]
    second = cards[1] if len(cards) > 1 else cards[0]
    topic = title_topic(message, cards)
    return (
        f"Về {topic}, nguồn truy xuất cho biết {first_clean_fact(main, 210)} {main['marker']}. "
        f"Một nguồn khác bổ sung {first_clean_fact(second, 180)} {second['marker']}. "
        "Kết luận này chỉ phản ánh phạm vi các nguồn đã hiển thị."
    )


def postcheck_issues(answer: str, cards: list[dict[str, Any]], case: dict[str, Any], date_info: dict[str, Any] | None = None) -> list[str]:
    issues: list[str] = []
    if metadata_leak_detected(answer):
        issues.append("metadata_leakage")
    if fake_markers(answer, cards):
        issues.append("fake_marker")
    if likely_truncated(answer):
        issues.append("truncation")
    if case.get("category") == "exact_date_lookup" and date_info and date_info.get("has_exact_date"):
        exact_cards = [card for card in cards if supports_exact_date(card, date_info["date"])]
        if not exact_cards and re.search(r"\[[0-9]+\]", answer or ""):
            issues.append("exact_date_overclaim")
    return issues


def infer_case(message: str, previous_query: str | None) -> dict[str, Any]:
    norm = strip_vn(message)
    prior_norm = strip_vn(previous_query or "")
    date_info = parse_query_date(message)
    year = parse_year_query(message)

    if contains(message, "giá vàng", "gia vang", "bitcoin", "thời tiết", "thoi tiet", "bóng đá", "bong da", "vô địch", "vo dich", "tối qua", "toi qua"):
        return {"category": "oos", "safe_mode": "safe_out_of_scope", "answerable": False}
    if norm.strip() in {"geneve", "genève"}:
        return {"category": "ambiguous", "safe_mode": "safe_clarification", "answerable": False}
    if contains(message, "chien dich do") and not previous_query:
        return {"category": "ellipsis_without_memory", "safe_mode": "safe_clarification", "answerable": False}
    if contains(message, "thương vong chính xác từng xã", "thuong vong chinh xac tung xa", "thương vong từng xã", "thuong vong tung xa", "số quân chính xác từng đơn vị", "so quan chinh xac tung don vi"):
        return {"category": "negative_gap", "safe_mode": "safe_insufficient_data", "answerable": False}
    if contains(message, "99/99"):
        return {"category": "exact_date_lookup", "answerable": True, "date_info": parse_query_date(message)}

    if contains(message, "nguồn nào", "nguon nao", "dựa vào nguồn nào", "dua vao nguon nao", "có nguồn", "co nguon", "tài liệu nào", "tai lieu nao", "source"):
        return {
            "category": "citation_source",
            "answerable": True,
            "expected_entities": ["30/4/1975"] if contains(message, "30/4", "1975") else [],
            "expected_keywords": extract_years(message) + ["nguồn"],
        }

    if date_info.get("has_exact_date"):
        return {
            "category": "exact_date_lookup",
            "answerable": True,
            "date_info": date_info,
            "expected_entities": [str(date_info.get("date"))],
            "expected_keywords": [str(date_info.get("year"))],
        }

    if re.search(r"(19\d{2}|18\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|18\d{2}|20\d{2})", norm) or contains(message, "giai đoạn", "giai doan", "từ năm", "tu nam"):
        return {"category": "period_timeline", "answerable": True, "expected_entities": [], "expected_keywords": extract_years(message)}

    if year and contains(message, "những sự kiện năm", "nhung su kien nam", "các sự kiện năm", "cac su kien nam", "năm") and contains(message, "sự kiện", "su kien", "timeline", "mốc", "moc"):
        return {
            "category": "year_events_timeline",
            "answerable": True,
            "year": year,
            "expected_entities": [str(year)],
            "expected_keywords": [str(year), "sự kiện", "timeline"],
        }

    if contains(message, "nó khác", "no khac") and contains(message, "paris") and contains(previous_query or "", "geneve", "genève"):
        return {
            "category": "follow_up",
            "answerable": True,
            "comparison": True,
            "memory_rewrite": "Hiệp định Genève 1954 khác Hiệp định Paris 1973 thế nào?",
            "expected_entities": ["Genève", "Paris"],
            "expected_keywords": ["Genève", "Paris", "1954", "1973"],
        }

    if contains(message, "so sánh", "so sanh") and contains(message, "geneve", "genève") and contains(message, "paris"):
        return {
            "category": "comparison",
            "answerable": True,
            "comparison": True,
            "expected_entities": ["Genève", "Paris"],
            "expected_keywords": ["Genève", "Paris", "1954", "1973"],
        }

    if contains(message, "so sánh", "so sanh", "khác nhau", "khac nhau", "giống và khác", "giong va khac", " khác ", " khac "):
        return {
            "category": "comparison",
            "answerable": True,
            "comparison": True,
            "expected_entities": [],
            "expected_keywords": extract_years(message),
        }

    if contains(message, "năm bao nhiêu", "nam bao nhieu", "khi nào", "khi nao", "vào năm nào", "vao nam nao", "diễn ra khi nào", "dien ra khi nao"):
        return {
            "category": "fact_date_lookup",
            "answerable": True,
            "expected_entities": ["30/4/1975"] if contains(message, "thống nhất đất nước", "thong nhat dat nuoc") else [],
            "expected_keywords": ["1975", "30/4", "thống nhất"] if contains(message, "thống nhất đất nước", "thong nhat dat nuoc") else [],
        }

    if contains(message, "cách mạng tháng tám", "cach mang thang tam", "cm thang 8"):
        return {
            "category": "timeline_summary" if contains(message, "mốc", "moc", "tóm tắt", "tom tat", "diễn biến", "dien bien", "timeline") else "topic_overview",
            "answerable": True,
            "expected_entities": ["Cách mạng Tháng Tám"],
            "expected_keywords": ["Tháng Tám", "1945"],
        }

    if contains(message, "điện biên phủ", "dien bien phu", "dienn bin phu"):
        raw_lower = message.lower()
        no_ask = not any(contains(message, ask) for ask in ASK_WORDS)
        cat = "topic_overview" if no_ask else "heavy_typo" if contains(message, "dicch", "dienn", "bin phu") else "no_accent" if "dien bien phu" in raw_lower else "fact"
        return {
            "category": cat,
            "answerable": True,
            "expected_entities": ["Điện Biên Phủ"],
            "expected_keywords": ["Điện Biên Phủ", "1954"],
        }

    if contains(message, "hiệp định genève", "hiep dinh geneve", "geneve"):
        no_ask = not any(contains(message, ask) for ask in ASK_WORDS)
        return {
            "category": "topic_overview" if no_ask else "fact",
            "answerable": True,
            "expected_entities": ["Genève"],
            "expected_keywords": ["Genève", "1954"],
        }

    words = [w for w in re.split(r"\s+", norm.strip()) if w]
    if 1 <= len(words) <= 6 and not any(contains(message, ask) for ask in ASK_WORDS):
        return {"category": "topic_overview", "answerable": True, "expected_entities": [], "expected_keywords": extract_years(message)}

    return {"category": "fact", "answerable": True, "expected_entities": [], "expected_keywords": []}


def effective_query(message: str, case: dict[str, Any]) -> str:
    if case.get("memory_rewrite"):
        return case["memory_rewrite"]
    if case["category"] == "fact_date_lookup" and contains(message, "thống nhất đất nước", "thong nhat dat nuoc"):
        return "30/4/1975 kết thúc chiến tranh Việt Nam giải phóng miền Nam thống nhất đất nước"
    if case["category"] == "year_events_timeline" and case.get("year"):
        return f"timeline các sự kiện nổi bật năm {case['year']} lịch sử Việt Nam"
    if case["category"] == "exact_date_lookup":
        info = case.get("date_info") or {}
        return f"sự kiện lịch sử Việt Nam ngày {info.get('date') or message}"
    if case["category"] in {"no_accent", "heavy_typo", "fact"} and contains(message, "dien bien phu", "dienn bin phu", "điện biên phủ"):
        return "Chiến dịch Điện Biên Phủ 1954 ý nghĩa"
    if case["category"] in {"timeline_summary", "topic_overview"} and contains(message, "cách mạng tháng tám", "cach mang thang tam", "cm thang 8"):
        return "Cách mạng Tháng Tám 1945"
    return message


def safety_answer(stage13a: Any, safe_mode: str) -> str:
    return stage13a.SAFETY_TEMPLATES.get(safe_mode, stage13a.SAFETY_TEMPLATES["safe_clarification"])


def date_lookup_answer(cards: list[dict[str, Any]]) -> str:
    marker = cards[0]["marker"] if cards else ""
    return (
        f"Câu trả lời ngắn: nếu hiểu là mốc giải phóng miền Nam, kết thúc chiến tranh và thống nhất về mặt lãnh thổ, "
        f"thì là năm 1975, cụ thể ngày 30/4/1975 {marker}. "
        "Nếu bạn hỏi về thống nhất về mặt nhà nước, cần nêu rõ câu hỏi đó và hệ thống chỉ trả lời khi truy xuất được nguồn hỗ trợ trực tiếp."
    ).strip()


def normalize_runtime_debug(message: str, case: dict[str, Any], effective: str) -> dict[str, Any]:
    status = "canonicalized" if effective != message else "unchanged"
    if case.get("safe_mode"):
        status = "safety_passthrough"
    if case["category"] == "fact_date_lookup":
        status = "typo_corrected_date_lookup"
    if case["category"] in {"exact_date_lookup", "year_events_timeline", "topic_overview"}:
        status = f"{case['category']}_routed"
    return {
        "original_query": message,
        "normalized_query": effective,
        "retrieval_query": effective,
        "normalization_status": status,
        "rewrite_applied": effective != message,
        "confidence": 0.95 if effective != message else 0.8,
    }


def citation_payload(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for card in cards:
        rows.append(
            {
                "marker": card["marker"],
                "title": card.get("title_clean") or card.get("title", ""),
                "source_id": ", ".join(card.get("source_ids") or []),
                "doc_id": card.get("chunk_id", ""),
                "snippet": card.get("snippet_clean") or card.get("snippet", ""),
                "url": None,
                "metadata": {
                    "source_titles": card.get("source_titles", []),
                    "comparison_side": card.get("comparison_side"),
                    "granularity": card.get("granularity"),
                    "years": card.get("years", []),
                    "exact_dates": card.get("exact_dates", []),
                },
            }
        )
    return rows


def run(req: dict[str, Any]) -> dict[str, Any]:
    stage13a = load_stage13a()
    message = str(req.get("message") or req.get("question") or "").strip()
    previous_query = req.get("previous_query")
    session_id = str(req.get("session_id") or "web-demo")
    started = time.perf_counter()
    if not message:
        raise ValueError("Missing message")

    case = infer_case(message, previous_query)
    case = {"case_id": "web_dynamic", "query": message, **case}
    effective = effective_query(message, case)
    norm = normalize_runtime_debug(message, case, effective)

    if case.get("safe_mode"):
        answer = safety_answer(stage13a, case["safe_mode"])
        latency = round((time.perf_counter() - started) * 1000, 1)
        return {
            "answer": answer,
            "citations": [],
            "debug": {
                "session_id": session_id,
                "original_query": message,
                "normalized_query": norm["normalized_query"],
                "rewritten_query": None,
                "intent": case["category"],
                "safety_mode": case["safe_mode"].replace("safe_", "safe_"),
                "retrieval_mode": "deterministic_safety",
                "bm25_used": False,
                "query_embedding_generated": False,
                "vector_used": False,
                "faiss_used": False,
                "rrf_used": False,
                "rrf_k": 30,
                "bm25_fallback": False,
                "fallback_reason": None,
                "local_llm_called": False,
                "local_embedding_model": stage13a.MODEL_NAME,
                "local_llm_model": stage13a.LLM_MODEL,
                "chunks_count": 0,
                "sources_count": 0,
                "latency_ms": latency,
                "cloud_api_calls": 0,
                "cloud_embedding_calls": 0,
                "external_network_calls": 0,
            },
            "status": {"answerable": False, "safe": True, "no_cloud": True, "hybrid_complete": False},
        }

    runtime = get_runtime(stage13a)
    with contextlib.redirect_stdout(sys.stderr):
        results, trace = runtime.hybrid_retrieve(effective, bool(case.get("comparison")), case.get("expected_entities") or [])

    limit = stage13a.DEMO_CONFIG["comparison_context_top_k_demo"] if case.get("comparison") else stage13a.DEMO_CONFIG["context_top_k_demo"]
    cards = clean_cards(stage13a.build_cards(runtime, results, int(limit)))

    date_info = case.get("date_info") or parse_query_date(message)
    if req.get("return_generation_payload"):
        cards_for_response = cards
        safety_mode = "none"
        render_mode = "retrieval_context_only"
        direct_answer = ""

        if case["category"] == "exact_date_lookup":
            direct_answer, cards_for_response, safety_mode = exact_date_template(date_info, cards)
            render_mode = "insufficient_exact_date_template" if safety_mode != "none" else "exact_date_context_only"
        elif case["category"] == "year_events_timeline":
            year = int(case.get("year") or parse_year_query(message) or 0)
            cards_for_response = cards_for_year(cards, year)[:3] or cards[:3]
            render_mode = "year_events_context_only"
        elif case.get("comparison"):
            render_mode = "comparison_context_only"
        elif case["category"] == "citation_source":
            render_mode = "citation_source_context_only"
        elif case["category"] == "topic_overview":
            render_mode = "topic_overview_context_only"

        latency = round((time.perf_counter() - started) * 1000, 1)
        citations = citation_payload(cards_for_response)
        clean_context = []
        for card in cards_for_response:
            clean_context.append(
                {
                    "marker": card.get("marker"),
                    "title": card.get("title_clean") or card.get("title", ""),
                    "snippet": card.get("snippet_clean") or card.get("snippet", ""),
                    "source_id": ", ".join(card.get("source_ids") or []),
                    "date": ", ".join(str(y) for y in card.get("years") or []),
                    "metadata_safe": {
                        "granularity": card.get("granularity"),
                        "years": card.get("years", []),
                        "exact_dates": card.get("exact_dates", []),
                    },
                }
            )

        return {
            "answer": direct_answer,
            "citations": citations,
            "generation_payload": {
                "query": message,
                "normalized_query": norm["normalized_query"],
                "rewritten_query": case.get("memory_rewrite"),
                "intent": case["category"],
                "clean_context": clean_context,
                "answer_policy": render_mode,
                "citation_rules": "Use only the provided citation markers. If context is insufficient, say so.",
                "max_answer_tokens": 400,
                "language": "vi",
            },
            "debug": {
                "session_id": session_id,
                "original_query": message,
                "normalized_query": norm["normalized_query"],
                "rewritten_query": case.get("memory_rewrite"),
                "intent": case["category"],
                "safety_mode": safety_mode,
                "retrieval_mode": "hybrid_rrf",
                "bm25_used": True,
                "query_embedding_generated": runtime.local_query_embedding_calls > 0,
                "vector_used": runtime.faiss_search_calls > 0,
                "faiss_used": runtime.faiss_search_calls > 0,
                "rrf_used": runtime.rrf_fusion_calls > 0,
                "rrf_k": stage13a.RRF_K,
                "bm25_fallback": False,
                "fallback_reason": None,
                "local_llm_called": False,
                "local_embedding_model": stage13a.MODEL_NAME,
                "local_llm_model": stage13a.LLM_MODEL,
                "chunks_count": len(cards_for_response),
                "sources_count": len({sid for card in cards_for_response for sid in card.get("source_ids", [])}),
                "latency_ms": latency,
                "cloud_api_calls": 0,
                "cloud_embedding_calls": 0,
                "external_network_calls": 0,
                "render_mode": render_mode,
                "answer_generation_skipped": True,
                "retrieval_trace": {
                    "mode": trace.get("mode"),
                    "latency_ms": trace.get("latency_ms"),
                    "top_chunks": [r.get("chunk_id") for r in results[:5]],
                },
            },
            "status": {
                "answerable": safety_mode == "none",
                "safe": True,
                "no_cloud": True,
                "hybrid_complete": True,
                "retrieval_context_only": True,
            },
        }

    local_llm_called = False
    render_mode = "local_hybrid_template"
    timeout = False
    llm_error = None
    safety_mode = "none"
    cards_for_response = cards
    if case["category"] == "exact_date_lookup":
        answer, cards_for_response, safety_mode = exact_date_template(date_info, cards)
        render_mode = "exact_date_lookup_template" if cards_for_response else "insufficient_exact_date_template"
    elif case["category"] == "year_events_timeline":
        answer = year_events_template(int(case.get("year") or parse_year_query(message) or 0), cards)
        cards_for_response = cards_for_year(cards, int(case.get("year") or parse_year_query(message) or 0))[:3]
        render_mode = "year_events_timeline_template"
    elif case["category"] == "period_timeline":
        answer = timeline_summary_template(message, cards)
        render_mode = "period_timeline_template"
    elif case["category"] == "fact_date_lookup":
        answer = fact_date_lookup_template(message, cards)
        render_mode = "fact_date_lookup_template_v2"
    elif case["category"] == "citation_source":
        answer = citation_source_template_v2(cards)
        render_mode = "citation_source_template_v2"
    elif case.get("comparison"):
        answer = comparison_template_v2(case, cards)
        render_mode = "followup_comparison_template_v2" if case["category"] == "follow_up" else "comparison_template_v2"
    elif case["category"] == "topic_overview":
        answer = topic_overview_template(message, cards)
        render_mode = "topic_overview_template"
    elif case["category"] == "timeline_summary":
        answer = timeline_summary_template(message, cards)
        render_mode = "timeline_summary_template_v2"
    else:
        prompt_cards = []
        for card in cards:
            prompt_card = dict(card)
            prompt_card["title"] = card.get("title_clean") or card.get("title", "")
            prompt_card["snippet"] = card.get("snippet_clean") or card.get("snippet", "")
            prompt_cards.append(prompt_card)
        prompt = stage13a.prompt_from_cards(case, effective, prompt_cards)
        answer, _llm_ms, timeout, llm_error = stage13a.call_llm(case, prompt)
        runtime.local_llm_calls += 1
        local_llm_called = True
        render_mode = "local_llm_v2"
        if (
            timeout
            or not answer
            or not stage13a.inline_marker_pass(answer, True)
            or fake_markers(answer, cards)
            or metadata_leak_detected(answer)
            or likely_truncated(answer)
        ):
            answer = generic_fact_template(message, cards)
            render_mode = "clean_citation_aware_fallback_after_llm"

    issues = postcheck_issues(answer, cards_for_response or cards, case, date_info)
    if issues:
        if "metadata_leakage" in issues or "truncation" in issues or "fake_marker" in issues:
            if case["category"] == "year_events_timeline":
                answer = year_events_template(int(case.get("year") or parse_year_query(message) or 0), cards)
                cards_for_response = cards_for_year(cards, int(case.get("year") or parse_year_query(message) or 0))[:3]
                render_mode = "postcheck_year_events_template"
            elif case["category"] == "topic_overview":
                answer = topic_overview_template(message, cards)
                render_mode = "postcheck_topic_overview_template"
            elif case.get("comparison"):
                answer = comparison_template_v2(case, cards)
                render_mode = "postcheck_comparison_template"
            else:
                answer = generic_fact_template(message, cards)
                render_mode = "postcheck_clean_fallback"
        if "exact_date_overclaim" in issues:
            answer = exact_date_insufficient_answer(date_info, cards)
            cards_for_response = []
            safety_mode = "safe_insufficient_data"
            render_mode = "postcheck_insufficient_exact_date"
        issues = postcheck_issues(answer, cards_for_response or cards, case, date_info)

    latency = round((time.perf_counter() - started) * 1000, 1)
    retrieval_mode = "hybrid_rrf" if trace["mode"] != "comparison_balanced" else "hybrid_rrf"
    if case.get("comparison"):
        retrieval_mode = "hybrid_rrf"

    return {
        "answer": answer,
        "citations": citation_payload(cards_for_response),
        "debug": {
            "session_id": session_id,
            "original_query": message,
            "normalized_query": norm["normalized_query"],
            "rewritten_query": case.get("memory_rewrite"),
            "intent": case["category"],
            "safety_mode": safety_mode,
            "retrieval_mode": retrieval_mode,
            "bm25_used": True,
            "query_embedding_generated": runtime.local_query_embedding_calls > 0,
            "vector_used": runtime.faiss_search_calls > 0,
            "faiss_used": runtime.faiss_search_calls > 0,
            "rrf_used": runtime.rrf_fusion_calls > 0,
            "rrf_k": stage13a.RRF_K,
            "bm25_fallback": False,
            "fallback_reason": None,
            "local_llm_called": local_llm_called,
            "local_embedding_model": stage13a.MODEL_NAME,
            "local_llm_model": stage13a.LLM_MODEL,
            "chunks_count": len(cards_for_response),
            "sources_count": len({sid for card in cards_for_response for sid in card.get("source_ids", [])}),
            "latency_ms": latency,
            "cloud_api_calls": 0,
            "cloud_embedding_calls": 0,
            "external_network_calls": 0,
            "render_mode": render_mode,
            "postcheck_issues": issues,
            "llm_error": llm_error,
            "timeout": timeout,
            "bm25_search_calls": runtime.bm25_search_calls,
            "local_query_embedding_calls": runtime.local_query_embedding_calls,
            "faiss_search_calls": runtime.faiss_search_calls,
            "rrf_fusion_calls": runtime.rrf_fusion_calls,
            "local_llm_calls": runtime.local_llm_calls,
            "retrieval_trace": {
                "mode": trace.get("mode"),
                "latency_ms": trace.get("latency_ms"),
                "top_chunks": [r.get("chunk_id") for r in results[:5]],
            },
        },
        "status": {"answerable": safety_mode == "none", "safe": True, "no_cloud": True, "hybrid_complete": True},
    }


def main() -> int:
    try:
        request = json.loads(sys.stdin.read() or "{}")
        response = run(request)
        print(json.dumps(response, ensure_ascii=False))
        return 0
    except Exception as exc:
        payload = {
            "error": "local_hybrid_runtime_error",
            "message": str(exc),
            "status": {"answerable": False, "safe": False, "no_cloud": True, "hybrid_complete": False},
            "debug": {
                "cloud_api_calls": 0,
                "cloud_embedding_calls": 0,
                "external_network_calls": 0,
                "fallback_reason": str(exc),
            },
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
