from typing import Dict

from domains.common import ngram_best_choice
from utils.bm25 import bm25_retrieve


def solve(processed: Dict) -> str:
    context = bm25_retrieve(processed["passage"], processed["question"])
    return ngram_best_choice(
        processed["question"],
        processed["choices"],
        context=context,
    )
