from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class NliHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 3):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled)
        return self.classifier(x)


class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled)
        return self.mlp(x).squeeze(-1)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size

    def forward(self, **enc):
        outputs = self.encoder(**enc)
        # Use [CLS] substitute (DistilBERT: first token)
        pooled = outputs.last_hidden_state[:, 0]
        return pooled


class NliModel(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 3):
        super().__init__()
        self.encoder = TextEncoder(model_name)
        self.head = NliHead(self.encoder.hidden_size, num_labels)

    def forward(self, **enc):
        pooled = self.encoder(**enc)
        logits = self.head(pooled)
        return logits


class MaskRegressor(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = TextEncoder(model_name)
        self.head = RegressionHead(self.encoder.hidden_size)

    def forward(self, **enc):
        pooled = self.encoder(**enc)
        value = self.head(pooled)
        return value


