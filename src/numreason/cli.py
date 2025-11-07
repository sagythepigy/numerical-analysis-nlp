import sys
from typing import Optional

import typer
from rich import print

from .reasoning import build_world_state
from .nli import decide_nli
from .mask_pred import predict_masked_number
from .data import load_numeric_nli, load_masked_numbers
from .train import TrainConfig, train_nli as train_nli_fn, eval_nli as eval_nli_fn, train_mask as train_mask_fn, eval_mask as eval_mask_fn
from transformers import AutoTokenizer
import json
import torch


app = typer.Typer(add_completion=False)


@app.command()
def nli(premise: str = typer.Option(..., help="Premise text"),
        hypothesis: str = typer.Option(..., help="Hypothesis text")) -> None:
    state = build_world_state(premise)
    label = decide_nli(state, hypothesis)
    print({"label": label})


@app.command()
def mask(premise: str = typer.Option(..., help="Premise text"),
         masked: str = typer.Option(..., help="Masked sentence with [MASK]")) -> None:
    state = build_world_state(premise)
    pred = predict_masked_number(state, masked)
    if pred is None:
        print({"prediction": None})
        raise typer.Exit(code=1)
    # print integers as ints when close to integer
    if abs(pred - round(pred)) < 1e-9:
        pred_out: Optional[float | int] = int(round(pred))
    else:
        pred_out = pred
    print({"prediction": pred_out})


@app.command()
def train_nli(dataset: str = typer.Option("equate", help="equate|numglue"),
              sample: int = typer.Option(500, help="per-split subsample size"),
              epochs: int = typer.Option(1),
              batch_size: int = typer.Option(8),
              model_name: str = typer.Option("distilbert-base-uncased"),
              metrics_path: str = typer.Option("", help="optional JSONL file path to log metrics"),
              local_train_path: str = typer.Option("", help="optional local train JSON file"),
              local_val_path: str = typer.Option("", help="optional local validation/dev JSON file"),
              local_test_path: str = typer.Option("", help="optional local test JSON file")) -> None:
    local_paths = None
    if any([local_train_path, local_val_path, local_test_path]):
        local_paths = {
            "train": local_train_path or None,
            "validation": local_val_path or None,
            "dev": local_val_path or None,
            "test": local_test_path or None,
        }
    ds = load_numeric_nli(dataset, split_sample=sample, local_paths=local_paths)
    cfg = TrainConfig(model_name=model_name, epochs=epochs, batch_size=batch_size)
    model = train_nli_fn(ds.get("train"), ds.get("validation"), cfg, metrics_path or None)
    # simple eval on test when available
    if "test" in ds and ds["test"] is not None:
        tok = AutoTokenizer.from_pretrained(model_name)
        acc = eval_nli_fn(model, tok, ds["test"], cfg)
        payload = {"phase": "test", "task": "nli", "accuracy": acc}
        print(payload)
        from .train import _maybe_write_metrics
        _maybe_write_metrics(metrics_path or None, payload)


@app.command()
def train_masked(dataset: str = typer.Option("numeracy600k", help="numeracy600k|drop_cloze|svamp_cloze"),
                 sample: int = typer.Option(1000),
                 epochs: int = typer.Option(1),
                 batch_size: int = typer.Option(8),
                 model_name: str = typer.Option("distilbert-base-uncased"),
                 metrics_path: str = typer.Option("", help="optional JSONL file path to log metrics"),
                 save_path: str = typer.Option("", help="optional path to save model checkpoint (*.pt)"),
                 local_train_path: str = typer.Option("", help="optional local train JSON file"),
                 local_val_path: str = typer.Option("", help="optional local validation/dev JSON file"),
                 local_test_path: str = typer.Option("", help="optional local test JSON file")) -> None:
    local_paths = None
    if any([local_train_path, local_val_path, local_test_path]):
        local_paths = {
            "train": local_train_path or None,
            "validation": local_val_path or None,
            "dev": local_val_path or None,
            "test": local_test_path or None,
        }
    ds = load_masked_numbers(dataset, split_sample=sample, local_paths=local_paths)
    cfg = TrainConfig(model_name=model_name, epochs=epochs, batch_size=batch_size)
    model = train_mask_fn(ds.get("train"), ds.get("validation"), cfg, metrics_path or None, save_path or None)
    if "test" in ds and ds["test"] is not None:
        tok = AutoTokenizer.from_pretrained(model_name)
        mae = eval_mask_fn(model, tok, ds["test"], cfg)
        payload = {"phase": "test", "task": "masked", "mae": mae}
        print(payload)
        from .train import _maybe_write_metrics
        _maybe_write_metrics(metrics_path or None, payload)


@app.command()
def predict_masked(model_name: str = typer.Option("distilbert-base-uncased"),
                   checkpoint_path: str = typer.Option(..., help="path to saved checkpoint (*.pt)"),
                   input_file: str = typer.Option(..., help="path to txt file with one masked sentence per line"),
                   output_file: str = typer.Option(..., help="path to write JSONL predictions"),
                   batch_size: int = typer.Option(16),
                   round_int: bool = typer.Option(False, help="round predictions to nearest integer"),
                   snap_min: int = typer.Option(-10**9, help="optional clamp lower bound after rounding/snap"),
                   snap_max: int = typer.Option(10**9, help="optional clamp upper bound after rounding/snap"),
                   snap_small_range: str = typer.Option("", help="optional inclusive range like '0,12' to snap to nearest integer in range")) -> None:
    from .models import MaskRegressor
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = MaskRegressor(model_name).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    # Load lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    preds: list[float] = []
    for i in tqdm(range(0, len(lines), batch_size), desc="predict"):
        batch = lines[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=160, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        preds.extend(out.detach().cpu().tolist())

    # Write JSONL
    # optional snapping
    snap_set = None
    if snap_small_range:
        try:
            lo_s, hi_s = [s.strip() for s in snap_small_range.split(",", 1)]
            lo_i, hi_i = int(lo_s), int(hi_s)
            snap_set = list(range(lo_i, hi_i + 1))
        except Exception:
            snap_set = None

    def post_process(x: float) -> float | int:
        y = x
        if round_int:
            y = round(y)
        if snap_set is not None:
            # snap to nearest integer in set
            y = min(snap_set, key=lambda v: abs(v - y))
        # clamp
        y = max(snap_min, min(snap_max, y))
        # pretty print ints
        return int(y) if isinstance(y, (int,)) or abs(y - round(y)) < 1e-9 else float(y)

    with open(output_file, "w", encoding="utf-8") as g:
        for text, pred in zip(lines, preds):
            val = post_process(float(pred))
            g.write(json.dumps({"masked": text, "prediction": val}, ensure_ascii=False) + "\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()


