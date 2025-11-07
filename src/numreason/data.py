from typing import Dict, List, Optional, Tuple

from datasets import load_dataset, DatasetDict
from datasets import Dataset
from huggingface_hub.utils import RepositoryNotFoundError
try:
    from datasets import DatasetNotFoundError
except ImportError:
    class DatasetNotFoundError(Exception):
        pass
from rich import print
from pathlib import Path
import json
import random


def _safe_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        try:
            return float(str(value).strip())
        except Exception:
            return 0.0


def _load_synthetic_nli() -> DatasetDict:
    print("[yellow]Falling back to synthetic NLI dataset (local JSONL).[/yellow]")
    ds = load_dataset("json", data_files={
        "train": "data/synthetic_nli.jsonl",
        "validation": "data/synthetic_nli.jsonl",
        "test": "data/synthetic_nli.jsonl",
    })
    label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
    return ds.map(lambda ex: {"label": label_map.get(str(ex.get("label", "neutral")), 2)})


def _load_local_json_records(path: Path) -> List[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # single object; wrap into list if it has "data" key
                if "data" in data and isinstance(data["data"], list):
                    data = data["data"]
                else:
                    data = [data]
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        # Fallback: assume JSON lines
        records: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
        if records:
            return records
    raise ValueError(f"Could not parse JSON records from {path}")


def _extract_answer_info(record: Dict) -> Optional[Tuple[str, bool, Optional[float]]]:
    ans = record.get("answer")
    if isinstance(ans, dict):
        number = ans.get("number") if isinstance(ans.get("number"), str) else ans.get("number")
        spans = ans.get("spans")
        if number:
            try:
                return str(number), True, float(number)
            except (TypeError, ValueError):
                pass
        if spans and isinstance(spans, list) and len(spans) == 1:
            span = spans[0]
            if span:
                try:
                    return str(span), True, float(span)
                except (TypeError, ValueError):
                    return str(span), False, None
        return None

    if isinstance(ans, (int, float)):
        return str(ans), True, float(ans)

    if isinstance(ans, str):
        stripped = ans.strip()
        if not stripped:
            return None
        try:
            return stripped, True, float(stripped)
        except ValueError:
            return stripped, False, None

    return None


def _compose_premise(record: Dict) -> str:
    passage = record.get("passage", "") or ""
    question = record.get("question", "") or ""
    if passage and question:
        return f"{passage.strip()} Question: {question.strip()}"
    return question.strip() or passage.strip()


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    txt = f"{value:.6f}"
    return txt.rstrip("0").rstrip(".")


def _generate_negative_text(answer_text: str, candidates: List[str], rng: random.Random) -> Optional[str]:
    filtered = [c for c in candidates if c.lower() != answer_text.lower()]
    if not filtered:
        return None
    return rng.choice(filtered)


def _generate_numeric_negatives(value: float, rng: random.Random, max_samples: int = 2) -> List[str]:
    candidates: List[float] = []
    offsets = [1.0, -1.0, 2.0, -2.0]
    magnitude = max(1.0, abs(value) * 0.1)
    offsets.extend([magnitude, -magnitude])
    if value != 0:
        offsets.extend([value * 0.5, value * -0.5, value * 1.5, value * -1.5])

    seen: set = set()
    for off in offsets:
        candidate = value + off
        if abs(candidate - value) < 1e-9:
            continue
        key = round(candidate, 9)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
        if len(candidates) >= max_samples:
            break

    while len(candidates) < max_samples:
        noise = rng.uniform(-10, 10) or 1.0
        candidate = value + noise
        if abs(candidate - value) < 1e-9:
            continue
        key = round(candidate, 9)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    return [_format_number(c) for c in candidates[:max_samples]]


def _generate_text_negatives(answer_text: str, pool: List[str], rng: random.Random, max_samples: int = 2) -> List[str]:
    filtered = [c for c in pool if c.lower() != answer_text.lower()]
    if not filtered:
        return []
    rng.shuffle(filtered)
    return filtered[:max_samples]


def _build_nli_split(records: List[Dict], rng: random.Random) -> List[Dict]:
    prepared: List[Tuple[Dict, str, bool, Optional[float]]] = []
    text_pool: List[str] = []
    for rec in records:
        info = _extract_answer_info(rec)
        if info is None:
            continue
        answer_text, is_numeric, numeric_val = info
        prepared.append((rec, answer_text, is_numeric, numeric_val))
        if not is_numeric:
            text_pool.append(answer_text)

    examples: List[Dict] = []
    negative_per_example = 2
    for rec, answer_text, is_numeric, numeric_val in prepared:
        premise = _compose_premise(rec)
        if not premise:
            continue
        hypothesis_pos = f"The answer is {answer_text}."
        examples.append({"premise": premise, "hypothesis": hypothesis_pos, "label": 0})

        negative_texts: List[str] = []
        if is_numeric and numeric_val is not None:
            negative_texts = _generate_numeric_negatives(numeric_val, rng, negative_per_example)
        else:
            negative_texts = _generate_text_negatives(answer_text, text_pool, rng, negative_per_example)

        for neg in negative_texts:
            if not neg or neg == answer_text:
                continue
            hypothesis_neg = f"The answer is {neg}."
            examples.append({"premise": premise, "hypothesis": hypothesis_neg, "label": 1})

    return examples


def load_numeric_nli(
    name: str = "equate",
    split_sample: int = 200,
    local_paths: Optional[Dict[str, str]] = None,
) -> DatasetDict:
    """
    Loads a numeric NLI dataset with small subsampling.
    name in {"equate", "numglue"}
    Returns DatasetDict with train/validation/test where available.
    """
    label_map = {
        "entailment": 0,
        "contradiction": 1,
        "neutral": 2,
        0: 0,
        1: 1,
        2: 2,
        "0": 0,
        "1": 1,
        "2": 2,
    }

    rng = random.Random(42)

    if local_paths:
        resolved = {split: Path(path) for split, path in local_paths.items() if path}
        for split, p in resolved.items():
            if not p.exists():
                raise FileNotFoundError(f"Local dataset file not found: {p}")
        splits: Dict[str, Dataset] = {}
        if "train" in resolved:
            records = _load_local_json_records(resolved["train"])
            splits["train"] = Dataset.from_list(_build_nli_split(records, rng))
        if "validation" in resolved:
            records = _load_local_json_records(resolved["validation"])
            splits["validation"] = Dataset.from_list(_build_nli_split(records, rng))
        elif "dev" in resolved:
            records = _load_local_json_records(resolved["dev"])
            splits["validation"] = Dataset.from_list(_build_nli_split(records, rng))
        if "test" in resolved:
            records = _load_local_json_records(resolved["test"])
            splits["test"] = Dataset.from_list(_build_nli_split(records, rng))
        if not splits:
            raise ValueError("No valid splits provided in local_paths for NLI dataset")
        ds = DatasetDict(splits)
    else:
        try:
            if name == "equate":
                try:
                    ds = load_dataset("equate", "eqnli")
                except (DatasetNotFoundError, FileNotFoundError, RepositoryNotFoundError):
                    ds = load_dataset("equate_nli")

                ds = ds.map(
                    lambda ex: {
                        "premise": ex.get("premise", ""),
                        "hypothesis": ex.get("hypothesis", ""),
                        "label": label_map.get(ex.get("label", "neutral"), 2),
                    },
                    remove_columns=[
                        c
                        for c in ds[list(ds.keys())[0]].column_names
                        if c not in {"premise", "hypothesis", "label"}
                    ],
                )
            elif name == "numglue":
                ds = load_dataset("num_glue", "num_guage_v1")
                ds = ds.map(
                    lambda ex: {
                        "premise": ex.get("premise", ""),
                        "hypothesis": ex.get("hypothesis", ""),
                        "label": label_map.get(ex.get("label", 2), 2),
                    }
                )
            else:
                raise ValueError("Unsupported NLI dataset name")
        except (DatasetNotFoundError, FileNotFoundError, RepositoryNotFoundError, ValueError) as err:
            print(f"[red]Warning: {err}. Using synthetic fallback.[/red]")
            ds = _load_synthetic_nli()

    def subsample(split):
        if split in ds and split_sample and len(ds[split]) > split_sample:
            return ds[split].select(range(split_sample))
        return ds[split] if split in ds else None

    out = DatasetDict()
    for split in ["train", "validation", "test"]:
        if split in ds:
            out[split] = subsample(split)
    return out


def _load_synthetic_mask() -> DatasetDict:
    print("[yellow]Falling back to synthetic masked-number dataset (local JSONL).[/yellow]")
    ds = load_dataset("json", data_files={
        "train": "data/synthetic_mask.jsonl",
        "validation": "data/synthetic_mask.jsonl",
        "test": "data/synthetic_mask.jsonl",
    })
    return ds.map(lambda ex: {"target": _safe_float(ex.get("target", ex.get("label", 0)))})


def load_masked_numbers(
    source: str = "numeracy600k",
    split_sample: int = 500,
    local_paths: Optional[Dict[str, str]] = None,
) -> DatasetDict:
    """
    Loads a masked-number cloze dataset.
    source in {"numeracy600k", "drop_cloze", "svamp_cloze"}.
    """
    if local_paths:
        resolved = {split: Path(path) for split, path in local_paths.items() if path}
        for split, p in resolved.items():
            if not p.exists():
                raise FileNotFoundError(f"Local dataset file not found: {p}")
        splits: Dict[str, Dataset] = {}
        if "train" in resolved:
            splits["train"] = Dataset.from_list(_load_local_json_records(resolved["train"]))
        if "validation" in resolved:
            splits["validation"] = Dataset.from_list(_load_local_json_records(resolved["validation"]))
        elif "dev" in resolved:
            splits["validation"] = Dataset.from_list(_load_local_json_records(resolved["dev"]))
        if "test" in resolved:
            splits["test"] = Dataset.from_list(_load_local_json_records(resolved["test"]))
        if not splits:
            raise ValueError("No valid splits provided in local_paths for masked dataset")
        ds = DatasetDict(splits)
        ds = ds.map(
            lambda ex: {
                "masked": ex.get("masked", ex.get("text", ex.get("question", ""))),
                "target": _safe_float(ex.get("target", ex.get("answer", ex.get("label", 0)))),
            }
        )
    else:
        try:
            if source == "numeracy600k":
                try:
                    ds = load_dataset("ai4finance-foundation/numeracy600k")
                except (DatasetNotFoundError, FileNotFoundError, RepositoryNotFoundError):
                    ds = load_dataset("numeracy600k")

                def map_ex(ex):
                    return {
                        "masked": ex.get("question", ex.get("text", "")),
                        "target": _safe_float(ex.get("answer", ex.get("label", 0))),
                    }

                ds = ds.map(
                    map_ex,
                    remove_columns=[
                        c for c in ds[list(ds.keys())[0]].column_names if c not in {"masked", "target"}
                    ],
                )
            elif source in {"drop_cloze", "svamp_cloze"}:
                return _load_synthetic_mask()
            else:
                raise ValueError("Unsupported masked dataset source")
        except (DatasetNotFoundError, FileNotFoundError, RepositoryNotFoundError, ValueError) as err:
            print(f"[red]Warning: {err}. Using synthetic fallback.[/red]")
            ds = _load_synthetic_mask()

    def subsample(split):
        if split in ds and split_sample and len(ds[split]) > split_sample:
            return ds[split].select(range(split_sample))
        return ds[split] if split in ds else None

    out = DatasetDict()
    for split in ["train", "validation", "test"]:
        if split in ds:
            out[split] = subsample(split)
    return out


