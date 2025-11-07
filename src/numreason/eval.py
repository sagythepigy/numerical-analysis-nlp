import json
from typing import Tuple

from .reasoning import build_world_state
from .nli import decide_nli
from .mask_pred import predict_masked_number


def eval_nli(jsonl_path: str) -> Tuple[int, int]:
    correct = 0
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            state = build_world_state(ex["premise"]) 
            pred = decide_nli(state, ex["hypothesis"])
            correct += int(pred == ex.get("label"))
            total += 1
    return correct, total


def eval_mask(jsonl_path: str, tol: float = 1e-6) -> Tuple[int, int]:
    correct = 0
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            state = build_world_state(ex["premise"]) 
            pred = predict_masked_number(state, ex["masked"]) 
            if pred is not None and abs(float(pred) - float(ex["target"])) <= tol:
                correct += 1
            total += 1
    return correct, total


