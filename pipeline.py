from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, List, Optional, Tuple

from domains import ignore_answer, math, multi_domain, rag, should_correct
from prompts import (
    DOMAIN_SYSTEM_PROMPTS,
    ROUTER_SYSTEM_PROMPT,
    domain_user_prompt,
    router_user_prompt,
)
from router import apply_route_fallback, route_question
from utils.bm25 import bm25_retrieve
from utils.config import rag_full_passage_chars
from utils.llm import LLMClient
from utils.postprocess import parse_answer, parse_route_output
from utils.preprocess import preprocess
from utils.reasoning import (
    answer_rag_with_evidence,
    answer_with_cot,
    should_use_rag_evidence,
    should_use_cot,
    should_use_pot_for_science,
    should_verify_answer,
    solve_science_with_pot,
    verify_answer,
)


DOMAIN_RUNNERS = {
    "rag": rag.solve,
    "science": math.solve,
    "multi_domain": multi_domain.solve,
    "should_correct": should_correct.solve,
    "ignore_answer": ignore_answer.solve,
}


def _llm_route_or_fallback(processed: Dict, llm_client: Optional[LLMClient]) -> Tuple[str, str, bool]:
    if llm_client is None:
        route_result = route_question(processed)
        return apply_route_fallback(route_result, processed["passage"]), "", True

    # Fast default for local inference: keep routing heuristic-only.
    if os.getenv("LLM_USE_LLM_ROUTE", "").strip() != "1":
        route_result = route_question(processed)
        domain = apply_route_fallback(route_result, processed["passage"])
        return domain, "", False

    try:
        raw_route = llm_client.chat(
            ROUTER_SYSTEM_PROMPT,
            router_user_prompt(processed["passage"], processed["question"], processed["num_choices"], processed.get("choices")),
            max_tokens=80,
        )
        route_result = parse_route_output(raw_route)
        used_fallback = False
    except Exception as exc:
        if os.getenv("DEBUG_LLM", "").strip() == "1":
            print(f"[DEBUG_LLM] route fallback for {processed.get('qid', 'unknown')}: {exc}")
        route_result = route_question(processed)
        raw_route = f"[ERROR] {exc}"
        used_fallback = True

    domain = apply_route_fallback(route_result, processed["passage"])
    return domain, raw_route, used_fallback


def _llm_answer_or_fallback(
    processed: Dict, domain: str, llm_client: Optional[LLMClient]
) -> Tuple[str, str, bool]:
    """Giải câu hỏi MCQ bằng cách kết hợp Heuristic Solvers và LLM với nhiều tầng fallback.

    Luồng ưu tiên:
    1. Heuristic-only: Nếu không dùng LLM, chạy trực tiếp heuristic solver tương ứng của domain.
    2. Ignore-answer: Early exit bằng heuristic quét pattern từ chối trực tiếp (bỏ qua LLM).
    3. Specialized math solvers: Kiểm tra nhanh các bài toán đặc thù bằng công thức cứng (Early exit).
    4. Program of Thought (PoT): Dùng Python sandbox để giải các câu hỏi định lượng/ký hiệu toán học.
    5. Chain of Thought (CoT): Suy luận từng bước trước khi trích xuất kết quả cho các câu hỏi lý thuyết phức tạp.
    6. LLM Standard chat: Chat trực tiếp với prompt chuyên biệt của từng domain.
    7. Fallback: Nếu tất cả các bước LLM/sandbox lỗi hoặc parse thất bại, quay về heuristic solver tương ứng.
    """
    if llm_client is None:
        return DOMAIN_RUNNERS.get(domain, multi_domain.solve)(processed), "", True

    if domain == "ignore_answer":
        answer = ignore_answer.solve(processed)
        return answer, "[HEURISTIC_DIRECT]", False

    specialized = math.solve_specialized(processed["question"], processed["choices"])
    if specialized:
        return specialized, "[HEURISTIC_SPECIALIZED]", False

    if domain == "multi_domain":
        specialized = multi_domain.solve_specialized(processed["question"], processed["choices"])
        if specialized:
            return specialized, "[HEURISTIC_SPECIALIZED_MULTI]", False

    if domain == "should_correct":
        specialized = should_correct.solve_specialized(processed["question"], processed["choices"])
        if specialized:
            return specialized, "[HEURISTIC_SPECIALIZED_SHOULD_CORRECT]", False

    if domain == "science" and should_use_pot_for_science(processed["question"], processed["choices"]):
        try:
            pot_answer, pot_trace = solve_science_with_pot(
                llm_client,
                processed["question"],
                processed["choices"],
            )
            if pot_answer:
                return pot_answer, pot_trace, False
        except Exception as exc:
            if os.getenv("DEBUG_LLM", "").strip() == "1":
                print(f"[DEBUG_LLM] PoT fallback for {processed.get('qid', 'unknown')}: {exc}")

    try:
        try:
            answer_max_tokens = int(os.getenv("LLM_ANSWER_MAX_TOKENS", "128"))
        except ValueError:
            answer_max_tokens = 128
        answer_max_tokens = max(16, min(answer_max_tokens, 512))

        domain_context = processed["passage"]
        if domain == "rag":
            full_limit = rag_full_passage_chars()
            if len(processed["passage"]) <= full_limit:
                domain_context = processed["passage"]
            else:
                domain_context = bm25_retrieve(processed["passage"], processed["question"])
        system_prompt = DOMAIN_SYSTEM_PROMPTS[domain]
        user_prompt = domain_user_prompt(domain, domain_context, processed["question"], processed["choices"])
        if domain == "multi_domain":
            hint = multi_domain.domain_hints(processed["question"], processed["choices"])
            if hint:
                user_prompt = f"{hint}\n\n{user_prompt}"

        if domain == "rag" and should_use_rag_evidence():
            rag_answer, rag_trace = answer_rag_with_evidence(
                llm_client,
                user_prompt,
                processed["question"],
                processed["choices"],
            )
            if rag_answer:
                if should_verify_answer(domain, processed["num_choices"]):
                    verified, verify_trace = verify_answer(
                        llm_client,
                        domain,
                        system_prompt,
                        user_prompt,
                        rag_answer,
                        rag_trace,
                        processed["question"],
                        processed["choices"],
                    )
                    if verified:
                        return verified, f"{rag_trace}\n{verify_trace}", False
                return rag_answer, rag_trace, False

        if should_use_cot(domain, processed["question"], processed["choices"]):
            cot_answer, cot_trace = answer_with_cot(
                llm_client,
                system_prompt,
                user_prompt,
                processed["question"],
                processed["choices"],
            )
            if cot_answer:
                if should_verify_answer(domain, processed["num_choices"]):
                    verified, verify_trace = verify_answer(
                        llm_client,
                        domain,
                        system_prompt,
                        user_prompt,
                        cot_answer,
                        cot_trace,
                        processed["question"],
                        processed["choices"],
                    )
                    if verified:
                        return verified, f"{cot_trace}\n{verify_trace}", False
                return cot_answer, cot_trace, False

        raw_answer = llm_client.chat(
            system_prompt,
            user_prompt,
            max_tokens=answer_max_tokens,
            enable_thinking=False,
        )
        parsed = parse_answer(raw_answer, processed["num_choices"])
        if parsed and parsed != "NONE":
            if should_verify_answer(domain, processed["num_choices"]):
                verified, verify_trace = verify_answer(
                    llm_client,
                    domain,
                    system_prompt,
                    user_prompt,
                    parsed,
                    raw_answer,
                    processed["question"],
                    processed["choices"],
                )
                if verified:
                    return verified, f"{raw_answer}\n{verify_trace}", False
            return parsed, raw_answer, False
    except Exception as exc:
        if os.getenv("DEBUG_LLM", "").strip() == "1":
            print(f"[DEBUG_LLM] answer fallback for {processed.get('qid', 'unknown')}: {exc}")
        raw_answer = f"[ERROR] {exc}"
    else:
        raw_answer = raw_answer if "raw_answer" in locals() else ""
    return DOMAIN_RUNNERS.get(domain, multi_domain.solve)(processed), raw_answer, True


def process_question(item: Dict, llm_client: Optional[LLMClient] = None) -> Dict:
    processed = preprocess(item)
    domain, llm_raw_route, route_fallback = _llm_route_or_fallback(processed, llm_client)
    answer, llm_raw_answer, used_fallback = _llm_answer_or_fallback(processed, domain, llm_client)
    trace_qid = os.getenv("TRACE_QID", "").strip()
    should_trace = os.getenv("TRACE_LLM", "").strip() == "1" and (
        not trace_qid or trace_qid == item.get("qid", "")
    )
    if should_trace:
        print(
            f"[TRACE_LLM] qid={item['qid']} domain={domain} "
            f"route_fallback={route_fallback} answer_fallback={used_fallback} answer={answer}"
        )
        if llm_raw_route:
            print(f"[TRACE_LLM] raw_route={llm_raw_route}")
        if llm_raw_answer:
            print(f"[TRACE_LLM] raw_answer={llm_raw_answer}")

    result = {"qid": item["qid"], "answer": answer, "domain": domain}
    result["llm_raw_route"] = llm_raw_route
    result["llm_raw_answer"] = llm_raw_answer
    result["route_fallback"] = route_fallback
    result["llm_fallback"] = used_fallback
    if "answer" in item:
        result["gold_answer"] = item["answer"]
        result["is_wrong"] = str(item["answer"]).strip().upper() != answer
    else:
        result["is_wrong"] = used_fallback
    return result


def run_pipeline(items: List[Dict], max_workers: int = 8, llm_client: Optional[LLMClient] = None) -> List[Dict]:
    worker_count = max(1, min(max_workers, 8))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(lambda x: process_question(x, llm_client), items))
