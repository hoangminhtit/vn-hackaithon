import os
import re
import subprocess
import sys
from typing import Dict, Optional, Tuple

from utils.llm import LLMClient
from utils.postprocess import parse_answer, valid_labels


_PYTHON_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def format_choices(choices: Dict[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in choices.items())


def extract_code(text: str) -> str:
    match = _PYTHON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _valid_letter(text: str, n_choices: int) -> Optional[str]:
    labels = set(valid_labels(n_choices))
    normalized = text.strip().upper()
    if normalized in labels:
        return normalized
    matches = re.findall(r"\b([A-Z])\b", normalized)
    for letter in reversed(matches):
        if letter in labels:
            return letter
    return None


def run_python_sandbox(code: str, timeout_s: float = 2.0) -> Tuple[bool, str]:
    """Run model-generated calculation code with restricted builtins/imports."""
    wrapper = r'''
import builtins
import sys

allowed_modules = {
    "math", "cmath", "statistics", "fractions", "decimal", "itertools",
    "functools", "operator", "collections", "sympy",
}

def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in allowed_modules:
        raise ImportError(f"Import not allowed: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)

safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr, "complex": complex,
    "dict": dict, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "len": len, "list": list, "map": map, "max": max, "min": min,
    "pow": pow, "print": print, "range": range, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "__import__": limited_import,
}

namespace = {"__builtins__": safe_builtins}
code = sys.stdin.read()
exec(compile(code, "<pot>", "exec"), namespace, namespace)
'''
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", wrapper],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, "Python timed out"
    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    if result.returncode != 0:
        return False, error or output or f"Python exited with {result.returncode}"
    return True, output


def should_use_pot_for_science(question: str, choices: Dict[str, str]) -> bool:
    """Xác định xem có nên sử dụng Program of Thought (PoT) để giải toán/khoa học hay không.

    Phương pháp này giúp sinh code Python để giải quyết các câu hỏi định lượng phức tạp,
    được gọi bởi pipeline._llm_answer_or_fallback.
    """
    if not _env_flag("LLM_USE_POT_SCIENCE", "1"):
        return False
    text = question.lower() + "\n" + "\n".join(choices.values()).lower()
    if any(token in text for token in ("$", "\\frac", "\\sqrt", "^", "_{")):
        return True
    if re.search(r"\d", question) and any(
        cue in question.lower()
        for cue in ("tính", "bao nhiêu", "xác định", "tìm", "giải", "tăng", "giảm")
    ):
        return True
    return len(choices) > 4 and re.search(r"\d", text) is not None


def solve_science_with_pot(
    llm_client: LLMClient,
    question: str,
    choices: Dict[str, str],
) -> Tuple[Optional[str], str]:
    labels = ", ".join(choices.keys())
    choice_block = format_choices(choices)
    system_prompt = (
        "Bạn là trợ lý giải trắc nghiệm định lượng bằng Python. "
        "Chỉ viết mã Python ngắn để chọn đáp án; không giải thích."
    )
    user_prompt = (
        f"Câu hỏi:\n{question}\n\n"
        f"Lựa chọn:\n{choice_block}\n\n"
        "Hãy viết một đoạn Python tự chứa để tính hoặc kiểm tra từng lựa chọn. "
        f"Ở dòng cuối, chỉ print đúng một chữ cái trong {{{labels}}}. "
        "Có thể dùng math, fractions, decimal, itertools, statistics; dùng sympy nếu cần và nếu có sẵn. "
        "Trả lời bằng một khối mã ```python ... ```."
    )
    max_tokens = int(os.getenv("LLM_POT_MAX_TOKENS", "512"))
    attempts = max(1, int(os.getenv("LLM_POT_RETRIES", "1")) + 1)
    last_trace = ""

    for attempt in range(attempts):
        raw = llm_client.chat(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            enable_thinking=False,
            apply_global_cap=False,
        )
        code = extract_code(raw)
        ok, output = run_python_sandbox(code, timeout_s=float(os.getenv("LLM_POT_TIMEOUT", "2.0")))
        letter = _valid_letter(output, len(choices)) if ok else None
        last_trace = f"[POT attempt={attempt + 1} ok={ok} output={output}]\n{code}"
        if letter:
            return letter, last_trace
        user_prompt = (
            f"Câu hỏi:\n{question}\n\n"
            f"Lựa chọn:\n{choice_block}\n\n"
            "Mã trước chưa in đúng một chữ cái đáp án.\n"
            f"Mã trước:\n```python\n{code}\n```\n\n"
            f"Lỗi/kết quả:\n{output}\n\n"
            f"Hãy sửa mã và ở dòng cuối chỉ print một chữ cái trong {{{labels}}}."
        )
    return None, last_trace


def should_use_cot(domain: str, question: str, choices: Dict[str, str]) -> bool:
    if domain == "should_correct":
        return _env_flag("LLM_USE_COT_SHOULD_CORRECT", "1")
    if domain != "multi_domain" or not _env_flag("LLM_USE_COT_MULTI", "1"):
        return False
    q = question.lower()
    hard_cues = (
        "theo ", "theo quan điểm", "theo luật", "theo hiến pháp", "theo tư tưởng",
        "hồ chí minh", "mác", "lênin", "chủ nghĩa", "luật", "pháp luật",
        "tất cả", "cả a", "khẳng định", "nhận định", "đúng", "sai",
    )
    return len(choices) > 4 or any(cue in q for cue in hard_cues) or any(
        "tất cả" in choice.lower() for choice in choices.values()
    )


def answer_with_cot(
    llm_client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    question: str,
    choices: Dict[str, str],
) -> Tuple[Optional[str], str]:
    choice_block = format_choices(choices)
    reasoning_max = int(os.getenv("LLM_COT_MAX_TOKENS", "384"))
    reasoning_system = (
        system_prompt
        + "\nHãy suy luận ngắn gọn theo các bước: xác định yêu cầu, đánh giá từng lựa chọn, loại trừ, kết luận."
    )
    reasoning_user = (
        user_prompt
        + "\n\nỞ lượt này, hãy phân tích ngắn từng lựa chọn và kết luận bằng 'Đáp án cuối: <chữ cái>'."
    )
    reasoning = llm_client.chat(
        reasoning_system,
        reasoning_user,
        max_tokens=reasoning_max,
        enable_thinking=False,
        apply_global_cap=False,
    )

    extract_system = (
        "Bạn là bộ trích xuất đáp án trắc nghiệm. "
        "Chỉ trả về đúng một dòng JSON, không giải thích."
    )
    extract_user = (
        f"Suy luận:\n{reasoning}\n\n"
        f"Câu hỏi:\n{question}\n\n"
        f"Lựa chọn:\n{choice_block}\n\n"
        'Trả lời đúng format: {"answer":"X"}'
    )
    raw = llm_client.chat(
        extract_system,
        extract_user,
        max_tokens=32,
        enable_thinking=False,
        apply_global_cap=False,
    )
    parsed = parse_answer(raw, len(choices))
    if parsed and parsed != "NONE":
        return parsed, f"[COT]\n{reasoning}\n[EXTRACT]\n{raw}"
    return None, f"[COT]\n{reasoning}\n[EXTRACT]\n{raw}"


def should_verify_answer(domain: str, n_choices: int) -> bool:
    if not _env_flag("LLM_USE_ANSWER_VERIFIER", "1"):
        return False
    if domain in {"rag", "should_correct"}:
        return True
    if domain == "multi_domain":
        return _env_flag("LLM_VERIFY_MULTI", "0") or (n_choices > 4 and _env_flag("LLM_VERIFY_MULTI_MANY_CHOICES", "0"))
    return False


def verify_answer(
    llm_client: LLMClient,
    domain: str,
    system_prompt: str,
    user_prompt: str,
    initial_answer: str,
    raw_answer: str,
    question: str,
    choices: Dict[str, str],
) -> Tuple[Optional[str], str]:
    labels = ", ".join(choices.keys())
    max_tokens = int(os.getenv("LLM_VERIFY_MAX_TOKENS", "320"))
    raw_excerpt = raw_answer[:1200]
    verifier_system = (
        "Bạn là bộ kiểm tra đáp án trắc nghiệm. Nhiệm vụ là phát hiện lỗi chọn đáp án, "
        "đặc biệt lỗi đọc sai yêu cầu ĐÚNG/SAI/KHÔNG, bỏ sót context, hoặc chọn 'tất cả' quá vội. "
        "Chỉ trả về JSON cuối cùng."
    )
    verifier_user = (
        f"Domain: {domain}\n\n"
        f"Prompt gốc:\n{user_prompt}\n\n"
        f"Đáp án ban đầu: {initial_answer}\n"
        f"Raw answer ban đầu:\n{raw_excerpt}\n\n"
        "Hãy kiểm tra lại thật ngắn gọn theo các tiêu chí:\n"
        "- Câu hỏi đang hỏi đáp án đúng, sai, ngoại trừ, hay thông tin theo context?\n"
        "- Nếu là RAG, đáp án có được context hỗ trợ trực tiếp không?\n"
        "- Nếu có 'tất cả/cả a,b,c', chỉ giữ nếu từng phương án đơn lẻ đều đúng.\n"
        "- Nếu đáp án ban đầu sai, sửa sang đáp án đúng nhất.\n"
        f"Cuối cùng chỉ in đúng một dòng JSON với chữ cái trong {{{labels}}}: "
        '{"answer":"X"}'
    )
    raw = llm_client.chat(
        verifier_system,
        verifier_user,
        max_tokens=max_tokens,
        enable_thinking=False,
        apply_global_cap=False,
    )
    parsed = parse_answer(raw, len(choices))
    if parsed and parsed != "NONE":
        return parsed, f"[VERIFY initial={initial_answer}]\n{raw}"
    return None, f"[VERIFY initial={initial_answer}]\n{raw}"


def should_use_rag_evidence() -> bool:
    return _env_flag("LLM_USE_RAG_EVIDENCE", "1")


def answer_rag_with_evidence(
    llm_client: LLMClient,
    user_prompt: str,
    question: str,
    choices: Dict[str, str],
) -> Tuple[Optional[str], str]:
    labels = ", ".join(choices.keys())
    max_tokens = int(os.getenv("LLM_RAG_EVIDENCE_MAX_TOKENS", "512"))
    system_prompt = (
        "Bạn là trợ lý trả lời trắc nghiệm RAG. Chỉ dùng context trong prompt. "
        "Phải tìm evidence trực tiếp trong context, không dùng kiến thức ngoài."
    )
    evidence_prompt = (
        f"{user_prompt}\n\n"
        "Hãy kiểm tra từng lựa chọn theo context. Với mỗi lựa chọn, ghi rất ngắn: "
        "SUPPORTED nếu context hỗ trợ trực tiếp, CONTRADICTED nếu mâu thuẫn, "
        "NOT_FOUND nếu không thấy evidence đủ rõ. "
        "Đặc biệt chú ý ngày/tháng/số liệu, câu phủ định/KHÔNG/ngoại trừ, và các lựa chọn chỉ khác nhau một chi tiết nhỏ.\n"
        f"Kết thúc bằng đúng dòng: Đáp án cuối: <một chữ cái trong {{{labels}}}>."
    )
    reasoning = llm_client.chat(
        system_prompt,
        evidence_prompt,
        max_tokens=max_tokens,
        enable_thinking=False,
        apply_global_cap=False,
    )

    extract_system = (
        "Bạn là bộ trích xuất đáp án. Chỉ trả về đúng một dòng JSON, không giải thích."
    )
    extract_user = (
        f"Phân tích RAG:\n{reasoning}\n\n"
        f"Câu hỏi:\n{question}\n\n"
        f"Lựa chọn:\n{format_choices(choices)}\n\n"
        'Trả lời đúng format: {"answer":"X"}'
    )
    raw = llm_client.chat(
        extract_system,
        extract_user,
        max_tokens=32,
        enable_thinking=False,
        apply_global_cap=False,
    )
    parsed = parse_answer(raw, len(choices))
    if parsed and parsed != "NONE":
        return parsed, f"[RAG_EVIDENCE]\n{reasoning}\n[EXTRACT]\n{raw}"
    return None, f"[RAG_EVIDENCE]\n{reasoning}\n[EXTRACT]\n{raw}"
