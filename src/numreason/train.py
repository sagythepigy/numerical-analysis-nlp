from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

from .models import NliModel, MaskRegressor


@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    lr: float = 3e-5
    batch_size: int = 8
    epochs: int = 1
    max_len: int = 160
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _collate_pair(batch, tokenizer, max_len):
    premises = [b["premise"] for b in batch]
    hyps = [b["hypothesis"] for b in batch]
    enc = tokenizer(premises, hyps, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return enc, labels


def _collate_text_number(batch, tokenizer, max_len):
    texts = [b["masked"] for b in batch]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    targets = torch.tensor([float(b["target"]) for b in batch], dtype=torch.float)
    return enc, targets


def _maybe_write_metrics(metrics_path: Optional[str], payload: Dict[str, Any]) -> None:
    if not metrics_path:
        return
    p = Path(metrics_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def train_nli(train_ds: Dataset, val_ds: Optional[Dataset], cfg: TrainConfig, metrics_path: Optional[str] = None) -> NliModel:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = NliModel(cfg.model_name).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: _collate_pair(b, tokenizer, cfg.max_len))

    model.train()
    for epoch in range(cfg.epochs):
        running_loss = 0.0
        step = 0
        for enc, labels in tqdm(train_loader, desc="train nli"):
            enc = {k: v.to(cfg.device) for k, v in enc.items()}
            labels = labels.to(cfg.device)
            opt.zero_grad()
            logits = model(**enc)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            step += 1

        train_loss = running_loss / max(1, step)
        payload = {"phase": "train", "task": "nli", "epoch": epoch + 1, "loss": train_loss}
        print(payload)
        _maybe_write_metrics(metrics_path, payload)

    if val_ds is not None:
        acc = eval_nli(model, tokenizer, val_ds, cfg)
        payload = {"phase": "validation", "task": "nli", "epoch": cfg.epochs, "accuracy": acc}
        print(payload)
        _maybe_write_metrics(metrics_path, payload)

    return model


def eval_nli(model: NliModel, tokenizer, ds: Dataset, cfg: TrainConfig) -> float:
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=lambda b: _collate_pair(b, tokenizer, cfg.max_len))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for enc, labels in tqdm(loader, desc="eval nli"):
            enc = {k: v.to(cfg.device) for k, v in enc.items()}
            labels = labels.to(cfg.device)
            logits = model(**enc)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total)


def train_mask(
    train_ds: Dataset,
    val_ds: Optional[Dataset],
    cfg: TrainConfig,
    metrics_path: Optional[str] = None,
    save_path: Optional[str] = None,
) -> MaskRegressor:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = MaskRegressor(cfg.model_name).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.SmoothL1Loss()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: _collate_text_number(b, tokenizer, cfg.max_len))

    model.train()
    for epoch in range(cfg.epochs):
        running_loss = 0.0
        step = 0
        for enc, targets in tqdm(train_loader, desc="train mask"):
            enc = {k: v.to(cfg.device) for k, v in enc.items()}
            targets = targets.to(cfg.device)
            opt.zero_grad()
            pred = model(**enc)
            loss = criterion(pred, targets)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            step += 1

        train_loss = running_loss / max(1, step)
        payload = {"phase": "train", "task": "masked", "epoch": epoch + 1, "loss": train_loss}
        print(payload)
        _maybe_write_metrics(metrics_path, payload)

    if val_ds is not None:
        mae = eval_mask(model, tokenizer, val_ds, cfg)
        payload = {"phase": "validation", "task": "masked", "epoch": cfg.epochs, "mae": mae}
        print(payload)
        _maybe_write_metrics(metrics_path, payload)

    # Save checkpoint if requested
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "model_name": cfg.model_name,
        }, p)

    return model


def eval_mask(model: MaskRegressor, tokenizer, ds: Dataset, cfg: TrainConfig) -> float:
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=lambda b: _collate_text_number(b, tokenizer, cfg.max_len))
    model.eval()
    total_abs = 0.0
    total = 0
    with torch.no_grad():
        for enc, targets in tqdm(loader, desc="eval mask"):
            enc = {k: v.to(cfg.device) for k, v in enc.items()}
            targets = targets.to(cfg.device)
            pred = model(**enc)
            total_abs += (pred - targets).abs().sum().item()
            total += targets.size(0)
    return total_abs / max(1, total)


