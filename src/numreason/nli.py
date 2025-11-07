from typing import Literal

from .parsing import extract_claim
from .reasoning import WorldState, query_quantity


Label = Literal["entailment", "contradiction", "unknown"]


def decide_nli(state: WorldState, hypothesis: str, tol: float = 1e-6) -> Label:
    subj, item, claimed = extract_claim(hypothesis)
    actual = query_quantity(state, subj, item)

    if claimed is None or actual is None:
        return "unknown"

    if abs(actual - float(claimed)) <= tol:
        return "entailment"
    else:
        return "contradiction"


