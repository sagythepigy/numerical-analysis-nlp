from typing import Optional

from .parsing import extract_claim
from .reasoning import WorldState, query_quantity


def predict_masked_number(state: WorldState, masked_sentence: str) -> Optional[float]:
    # Treat masked sentence like a claim with unknown number
    subj, item, _ = extract_claim(masked_sentence.replace("[MASK]", "0"))
    val = query_quantity(state, subj, item)
    return val


