import torch
import torch.nn as nn
import torch.nn.functional as F


def sasrec_loss(pos_logits, neg_logits, positive_items):
    mask = positive_items.ne(0)

    loss = F.softplus(-pos_logits) + F.softplus(neg_logits)

    return loss[mask].mean()


class SASRecBlocK(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_dropout = nn.Dropout(p=dropout)

        self.ffn_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        x,
        causal_mask,
        padding_mask,
    ):
        residual = x

        x_norm = self.attn_norm(x)

        # Left-padded queries otherwise have no allowed keys. Let them attend
        # to themselves to avoid NaNs, while real queries still ignore padding.
        attention_mask = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1)
        diagonal = torch.eye(x.size(1), dtype=torch.bool, device=x.device)
        attention_mask = attention_mask & ~diagonal.unsqueeze(0)
        attention_mask = attention_mask.repeat_interleave(
            self.attention.num_heads, dim=0
        )

        attn_out, _ = self.attention(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attention_mask,
            need_weights=False,
        )

        x = residual + self.attn_dropout(attn_out)

        x = x + self.ffn(self.ffn_norm(x))

        return x


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model

        # 0 = padding
        self.item_embedding = nn.Embedding(
            num_items + 1,
            d_model,
            padding_idx=0,
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

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(
            self.item_embedding.weight,
            std=0.02,
        )

        nn.init.normal_(
            self.position_embedding.weight,
            std=0.02,
        )

        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, item_seq: torch.Tensor):
        """
        item_seq: [batch, seq_len]

        0 = PAD
        """

        batch_size, seq_len = item_seq.shape

        if seq_len > self.max_len:
            raise ValueError(f"seq_len={seq_len} > max_len={self.max_len}")

        positions = torch.arange(seq_len, device=item_seq.device)

        positions = positions.unsqueeze(dim=0).expand(batch_size, -1)

        x = self.item_embedding(item_seq) + self.position_embedding(positions)

        x = self.embedding_dropout(x)

        # True = padding
        padding_mask = item_seq.eq(0)

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

        return self.final_norm(x)

    def forward(
        self,
        item_seq,
        positive_items,
        negative_items,
    ):
        hidden = self.encode(item_seq)

        positive_embeddings = self.item_embedding(positive_items)

        negative_embeddings = self.item_embedding(negative_items)

        positive_logits = (hidden * positive_embeddings).sum(dim=-1)

        negative_logits = (hidden * negative_embeddings).sum(dim=-1)

        return (positive_logits, negative_logits)

    @torch.no_grad()
    def score_items(
        self,
        item_seq: torch.Tensor,
        candidate_ids: torch.Tensor,
    ):
        """
        item_seq: [B, L]
        candidate_ids: [C]

        returns: [B, C]
        """
        hidden = self.encode(item_seq)

        # Last position - actual last interaction
        user_repr = hidden[:, -1, :]  # [B, 1, L]

        item_repr = self.item_embedding(candidate_ids)

        return user_repr @ item_repr.T
