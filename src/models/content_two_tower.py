import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sasrec import SASRecBlocK


class ContentTwoTower(nn.Module):
    def __init__(
        self,
        item_features: torch.Tensor,
        max_len: int = 50,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.2,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.max_len = max_len
        self.temperature = temperature

        self.register_buffer(
            "item_features",
            item_features.to(torch.float32),
            persistent=False,
        )

        feature_dim = item_features.shape[1]

        # The same tower for history items and candidate items
        self.item_tower = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, d_model),
        )

        self.position_embedding = nn.Embedding(
            max_len,
            d_model,
        )

        self.embedding_dropout = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList(
            [
                SASRecBlocK(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        nn.init.normal_(
            self.position_embedding.weight,
            std=0.02,
        )

    def item_vectors(self, item_ids: torch.Tensor):
        """
        item_ids: [...]

        returns: [..., d_model]
        """
        features = self.item_features[item_ids]
        vectors = self.item_tower(features)

        vectors = F.normalize(
            vectors,
            dim=-1,
        )

        # PAD should not have a meaningful embedding, zero it out
        vectors = vectors.masked_fill(
            item_ids.eq(0).unsqueeze(-1),
            0.0,
        )

        return vectors

    def encode(self, item_seq: torch.Tensor):
        """
        item_seq: [B, L], left-padded
        """
        batch_size, seq_len = item_seq.shape

        if seq_len > self.max_len:
            raise ValueError(f"seq_len={seq_len} > max_len={self.max_len}")

        padding_mask = item_seq.eq(0)

        positions = (
            torch.arange(
                seq_len,
                device=item_seq.device,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        x = self.item_vectors(item_seq) + self.position_embedding(positions)

        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = self.embedding_dropout(x)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=item_seq.device,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(
                x,
                causal_mask=causal_mask,
                padding_mask=padding_mask,
            )

        x = self.final_norm(x)

        user_repr = x[:, -1, :]

        return F.normalize(
            user_repr,
            dim=-1,
        )

    def forward(
        self,
        item_seq: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ):
        """
        item_seq: [B, L]
        positive_ids: [B]
        negative_ids: [B, N]
        """
        user_repr = self.encode(item_seq)

        candidate_ids = torch.cat(
            [
                positive_ids.unsqueeze(1),
                negative_ids,
            ],
            dim=1,
        )

        candidate_vectors = self.item_vectors(candidate_ids)

        logits = torch.einsum(
            "bd,bkd->bk",
            user_repr,
            candidate_vectors,
        )

        return logits / self.temperature
