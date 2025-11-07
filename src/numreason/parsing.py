import re
from typing import Dict, List, Optional, Tuple


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


EVENT_VERBS = {
    # verb -> delta sign from subject perspective
    # positive means acquisition, negative means giving/losing
    "gave": -1,
    "give": -1,
    "gave away": -1,
    "lost": -1,
    "spent": -1,
    "used": -1,
    "sold": -1,
    "bought": +1,
    "got": +1,
    "received": +1,
    "found": +1,
    "gained": +1,
    "added": +1,
}


def normalize_number(token: str) -> Optional[float]:
    token_l = token.lower().strip().strip(".,")
    if token_l in NUMBER_WORDS:
        return float(NUMBER_WORDS[token_l])
    # digits with optional decimal
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token_l):
        try:
            return float(token_l)
        except ValueError:
            return None
    return None


def tokenize_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extract_subject(tokens: List[str]) -> Optional[str]:
    # naive: first capitalized token is subject (e.g., John)
    for tok in tokens:
        if len(tok) > 0 and tok[0].isupper():
            return tok.strip(".,")
    return None


def extract_items(tokens: List[str]) -> Optional[str]:
    # choose the last plural/singular noun-looking token; naive heuristic
    # here we approximate by taking last alphabetic token
    nouns = [t.strip(".,").lower() for t in tokens if re.fullmatch(r"[A-Za-z][A-Za-z-]*", t)]
    return nouns[-1] if nouns else None


def extract_number(tokens: List[str]) -> Optional[float]:
    for t in tokens:
        val = normalize_number(t)
        if val is not None:
            return val
    return None


def classify_event(sentence: str) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[int]]:
    """
    Returns (subject, item_noun, number, sign)
    - sign in {+1, -1} for events; None if statement (e.g., had X)
    """
    tokens = sentence.split()
    subject = extract_subject(tokens)
    item = extract_items(tokens)
    num = extract_number(tokens)

    s_lower = sentence.lower()
    sign: Optional[int] = None
    for verb, v_sign in EVENT_VERBS.items():
        if verb in s_lower:
            sign = v_sign
            break

    # statements like "had 10 apples" (set/initialize)
    if sign is None and (" had " in s_lower or s_lower.startswith("had ")):
        sign = 0

    return subject, item, num, sign


def extract_claim(sentence: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Used for hypothesis claims like "John now has 7 apples"."""
    tokens = sentence.split()
    subject = extract_subject(tokens)
    item = extract_items(tokens)
    num = extract_number(tokens)
    return subject, item, num


