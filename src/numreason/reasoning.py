from typing import Dict, Optional

from .parsing import classify_event, tokenize_sentences


WorldState = Dict[str, Dict[str, float]]  # subject -> item -> quantity


def ensure_entry(state: WorldState, subject: str, item: str) -> None:
    if subject not in state:
        state[subject] = {}
    if item not in state[subject]:
        state[subject][item] = 0.0


def apply_event(state: WorldState, sentence: str) -> None:
    subject, item, num, sign = classify_event(sentence)
    if subject is None or item is None or num is None or sign is None:
        return

    ensure_entry(state, subject, item)

    if sign == 0:
        # initialization/statement: set to num (best effort: overwrite)
        state[subject][item] = float(num)
    else:
        state[subject][item] += float(sign) * float(num)


def build_world_state(premise: str) -> WorldState:
    state: WorldState = {}
    for sent in tokenize_sentences(premise):
        apply_event(state, sent)
    return state


def query_quantity(state: WorldState, subject: Optional[str], item: Optional[str]) -> Optional[float]:
    if subject is None or item is None:
        return None
    if subject not in state or item not in state[subject]:
        return None
    return state[subject][item]


