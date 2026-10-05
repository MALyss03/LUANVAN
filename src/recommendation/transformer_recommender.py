
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import math

import torch
from torch import nn


class CourseTransformer(nn.Module):
    """Small sequential recommender.

    Input: sequence of course IDs.
    Output: logits over the course vocabulary for the next-course prediction.
    """

    def __init__(
        self,
        num_courses: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 32,
    ):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(num_courses, d_model, padding_idx=0)
        self.position = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_courses)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        if seq_len > self.max_len:
            tokens = tokens[:, -self.max_len:]
            seq_len = self.max_len

        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.position(pos)
        padding_mask = tokens.eq(0)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1
        last = x[torch.arange(bsz, device=tokens.device), lengths]
        return self.head(self.norm(last))


class TransformerRanker:
    """Inference adapter for the sequential Transformer."""

    def __init__(
        self,
        checkpoint: str | Path,
        course_to_idx: Dict[str, int],
        device: str = "cpu",
        **model_kwargs,
    ):
        self.device = torch.device(device)
        self.course_to_idx = course_to_idx
        self.idx_to_course = {v: k for k, v in course_to_idx.items()}
        self.model = CourseTransformer(
            num_courses=len(course_to_idx),
            **model_kwargs,
        ).to(self.device)

        state = torch.load(checkpoint, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.no_grad()
    def score_candidates(self, history: List[str], candidate_ids: List[str]) -> Dict[str, float]:
        tokens = [self.course_to_idx.get(x, 0) for x in history]
        if not tokens:
            tokens = [0]
        tokens = tokens[-self.model.max_len:]
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        logits = self.model(x)[0]
        probs = torch.softmax(logits, dim=-1)

        scores = {}
        for cid in candidate_ids:
            idx = self.course_to_idx.get(cid)
            scores[cid] = float(probs[idx].item()) if idx is not None else 0.0
        return scores
