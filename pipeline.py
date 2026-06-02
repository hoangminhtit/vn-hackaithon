from concurrent.futures import ThreadPoolExecutor
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
    "math": math.solve,
    "multi_domain": multi_domain.solve,
    "should_correct": should_correct.solve,
    "ignore_answer": ignore_answer.solve,
}


def _llm_route_or_fallback(processed: Dict, llm_client: LLMClient | None) -> str:
    if llm_client is None:
        route_result = route_question(processed)
        return apply_route_fallback(route_result, processed["passage"])

    try:
        raw_route = llm_client.chat(
            ROUTER_SYSTEM_PROMPT,
            router_user_prompt(processed["passage"], processed["question"], processed["num_choices"]),
            max_tokens=80,
        )
        route_result = parse_route_output(raw_route)
    except Exception:
        route_result = route_question(processed)

    return apply_route_fallback(route_result, processed["passage"])


def _llm_answer_or_fallback(processed: Dict, domain: str, llm_client: LLMClient | None) -> str:
    if llm_client is None:
        return DOMAIN_RUNNERS.get(domain, multi_domain.solve)(processed)

    try:
        domain_context = processed["passage"]
        if domain == "rag":
            domain_context = bm25_retrieve(processed["passage"], processed["question"])
        raw_answer = llm_client.chat(
            DOMAIN_SYSTEM_PROMPTS[domain],
            domain_user_prompt(domain, domain_context, processed["question"], processed["choices"]),
            max_tokens=220 if domain in {"math", "should_correct"} else 120,
        )
        parsed = parse_answer(raw_answer, processed["num_choices"])
        if parsed and parsed != "NONE":
            return parsed
    except Exception:
        pass
    return DOMAIN_RUNNERS.get(domain, multi_domain.solve)(processed)


def process_question(item: Dict, llm_client: LLMClient | None = None) -> Dict:
    processed = preprocess(item)
    domain = _llm_route_or_fallback(processed, llm_client)
    answer = _llm_answer_or_fallback(processed, domain, llm_client)
    return {"qid": item["qid"], "answer": answer, "domain": domain}


def run_pipeline(items: List[Dict], max_workers: int = 8, llm_client: LLMClient | None = None) -> List[Dict]:
    worker_count = max(1, min(max_workers, 8))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(lambda x: process_question(x, llm_client), items))
