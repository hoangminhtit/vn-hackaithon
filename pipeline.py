from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, List

from domains import ignore_answer, math, multi_domain, rag, should_correct
from prompts import (
    DOMAIN_SYSTEM_PROMPTS,
    ROUTER_SYSTEM_PROMPT,
    domain_user_prompt,
    router_user_prompt,
)
from router import apply_route_fallback, route_question
from utils.bm25 import bm25_retrieve
from utils.llm import LLMClient
from utils.postprocess import parse_answer, parse_route_output
from utils.preprocess import preprocess


DOMAIN_RUNNERS = {
    "rag": rag.solve,
    "science": math.solve,
    "multi_domain": multi_domain.solve,
    "should_correct": should_correct.solve,
    "ignore_answer": ignore_answer.solve,
}


def _llm_route_or_fallback(processed: Dict, llm_client: LLMClient | None) -> tuple[str, str, bool]:
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
    processed: Dict, domain: str, llm_client: LLMClient | None
) -> tuple[str, str, bool]:
    if llm_client is None:
        return DOMAIN_RUNNERS.get(domain, multi_domain.solve)(processed), "", True

    if domain == "ignore_answer":
        answer = ignore_answer.solve(processed)
        return answer, "[HEURISTIC_DIRECT]", False

    try:
        try:
            answer_max_tokens = int(os.getenv("LLM_ANSWER_MAX_TOKENS", "128"))
        except ValueError:
            answer_max_tokens = 128
        answer_max_tokens = max(16, min(answer_max_tokens, 512))

        domain_context = processed["passage"]
        if domain == "rag":
            if len(processed["passage"]) <= 6000:
                domain_context = processed["passage"]
            else:
                domain_context = bm25_retrieve(processed["passage"], processed["question"])
        raw_answer = llm_client.chat(
            DOMAIN_SYSTEM_PROMPTS[domain],
            domain_user_prompt(domain, domain_context, processed["question"], processed["choices"]),
            max_tokens=answer_max_tokens,
            enable_thinking=False,
        )
        parsed = parse_answer(raw_answer, processed["num_choices"])
        if parsed and parsed != "NONE":
            return parsed, raw_answer, False
    except Exception as exc:
        if os.getenv("DEBUG_LLM", "").strip() == "1":
            print(f"[DEBUG_LLM] answer fallback for {processed.get('qid', 'unknown')}: {exc}")
        raw_answer = f"[ERROR] {exc}"
    else:
        raw_answer = raw_answer if "raw_answer" in locals() else ""
    return DOMAIN_RUNNERS.get(domain, multi_domain.solve)(processed), raw_answer, True


def process_question(item: Dict, llm_client: LLMClient | None = None) -> Dict:
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


def run_pipeline(items: List[Dict], max_workers: int = 8, llm_client: LLMClient | None = None) -> List[Dict]:
    worker_count = max(1, min(max_workers, 8))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(lambda x: process_question(x, llm_client), items))
