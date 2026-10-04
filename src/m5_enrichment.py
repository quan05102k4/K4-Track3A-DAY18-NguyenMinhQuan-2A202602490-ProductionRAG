from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=1)
    return _client


def _chat(system: str, user: str, max_tokens: int = 1024, retries: int = 4) -> str | None:
    """Gọi LLM. Trả về None khi không có API key hoặc lỗi để hàm gọi dùng fallback.
    Gặp 429 (rate limit) thì chờ rồi thử lại."""
    import time
    from openai import RateLimitError

    if not LLM_API_KEY:
        return None
    for attempt in range(retries):
        try:
            resp = _get_client().chat.completions.create(
                model=LLM_MODEL, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return (resp.choices[0].message.content or "").strip() or None
        except RateLimitError as e:
            if attempt == retries - 1:
                print(f"  ⚠️  LLM rate limit, bỏ qua: {str(e)[:120]}")
                return None
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"  ⚠️  LLM call failed: {str(e)[:200]}")
            return None
    return None


def _parse_json(raw: str | None) -> dict | None:
    """Parse JSON từ output LLM, bỏ code fence ```json nếu có."""
    if not raw:
        return None
    import json as _json
    import re
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        data = _json.loads(cleaned)
    except _json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extractive_summary(text: str) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


def _extractive_questions(text: str, n: int) -> list[str]:
    import re
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n]]


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    out = _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.", text, max_tokens=300)
    return out or _extractive_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    out = _chat(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                "Mỗi câu hỏi trên 1 dòng, không đánh số.", text, max_tokens=600)
    if out:
        questions = [q.strip().lstrip("0123456789.-) ") for q in out.split("\n") if q.strip()]
        return questions[:n_questions]
    return _extractive_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    out = _chat("Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
                "Chỉ trả về 1 câu.", f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=300)
    if out:
        return f"{out}\n\n{text}"
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    out = _chat('Trích xuất metadata từ đoạn văn. Chỉ trả về JSON: '
                '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
                text, max_tokens=300)
    data = _parse_json(out)
    if data is not None:
        return data
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    system = """Phân tích đoạn văn và chỉ trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""
    data = _parse_json(_chat(system, f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", max_tokens=600))
    if data is not None:
        if not isinstance(data.get("metadata"), dict):
            data["metadata"] = {}
        return data
    # Fallback khi không có API key hoặc lỗi: dùng các hàm trích xuất không cần LLM
    return {
        "summary": _extractive_summary(text),
        "questions": _extractive_questions(text, 3),
        "context": f"Trích từ {source}." if source else "",
        "metadata": {},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
