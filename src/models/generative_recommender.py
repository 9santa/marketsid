import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerativeRecommender(nn.Module):
    def __init__(
        self,
        item_sids: torch.Tensor,
        vocab_sizes: list[int],
        max_len: int = 50,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.num_levels = len(vocab_sizes)
        self.vocab_sizes = vocab_sizes

        # Lookup table: ordinaty item ID -> item SID sequence
        # [num_items + 1, 4]
        # row 0 = padding item
        self.register_buffer(
            "item_sids",
            item_sids.long(),
            persistent=False,
        )

        # Trainable embeddings for SIDs.
        # Every SID-level has its own (potentially unrelated) semantic,
        # so we build an embedding space for each level.
        self.sid_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    vocab_size,
                    d_model,
                )
                for vocab_size in vocab_sizes
            ]
        )

        self.history_position_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
        )

        self.encoder_norm = nn.LayerNorm(d_model)

        self.bos_embedding = nn.Parameter(torch.empty(d_model))

        self.decoder_position_embedding = nn.Embedding(
            self.num_levels,
            d_model,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
        )

        self.decoder_norm = nn.LayerNorm(d_model)

        # We have `vocab_sizes` heads because every level has a different classification problem
        # i.e. the first head gives scores for: P(c1=0), P(c1=1), ..., P(c1=vocab_size)
        self.output_heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size) for vocab_size in vocab_sizes]
        )

        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(
            self.bos_embedding,
            std=0.02,
        )

        nn.init.normal_(
            self.history_position_embedding.weight,
            std=0.02,
        )

        nn.init.normal_(
            self.decoder_position_embedding.weight,
            std=0.02,
        )

        for embedding in self.sid_embeddings:
            nn.init.normal_(
                embedding.weight,
                std=0.02,
            )

    def embed_items(
        self,
        item_ids: torch.Tensor,
    ):
        """
        Embeds catalog item IDs into `d_model`-dim vectors
        by summing embeddings from all SID levels.

        item_ids: [B, L]

        return: [B, L, D]
        """

        # Lookup SIDs
        sids = self.item_sids[item_ids]

        result = torch.zeros(
            *item_ids.shape,
            self.d_model,
            dtype=torch.float32,
            device=item_ids.device,
        )

        for level, embedding in enumerate(self.sid_embeddings):
            codes = sids[..., level]

            result = result + embedding(codes)

        # Divide so variance of the sum stays roughly consistent
        result = result / (self.num_levels**0.5)

        result = result.masked_fill(
            item_ids.eq(0).unsqueeze(-1),
            0.0,
        )

        return result

    def encode_history(
        self,
        item_seq: torch.Tensor,
    ):
        """
        item_seq: [B, L], left-padded

        returns: memory [B, L, D]
        """

        batch_size, seq_len = item_seq.shape

        if seq_len > self.max_len:
            raise ValueError(f"{seq_len=} > {self.max_len=}")

        padding_mask = item_seq.eq(0)

        positions = torch.arange(
            seq_len,
            device=item_seq.device,
        ).unsqueeze(0)

        x = self.embed_items(item_seq) + self.history_position_embedding(positions)

        x = self.dropout(x)

        # Causal mask is not needed here because
        # `item_seq` is already known user's history,
        # which doesn't include the target.
        memory = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        memory = self.encoder_norm(memory)

        # memory = everything the model currently understands about this user's interactions
        return memory, padding_mask

    def make_decoder_input(
        self,
        target_sids: torch.Tensor,
    ):
        """
        Constructs the input for Generative part with teacher-forcing shift.
        INPUT             PREDICT
        --------------------------------
        BOS         →       c1
        c1          →       c2
        c2          →       c3
        c3          →       c4

        target_sids: [B, 4]

        decoder input: [BOS, c1, c2, c3]

        return: [B, 4, D]
        """

        batch_size = target_sids.shape[0]

        decoder_input = torch.zeros(
            batch_size,
            self.num_levels,
            self.d_model,
            device=target_sids.device,
        )  # [B, 4, D]

        # First token is BOS
        decoder_input[:, 0] = self.bos_embedding

        # Fill the rest of the tokens: c1, c2, c3, ...
        for position in range(1, self.num_levels):
            prev_level = position - 1

            decoder_input[:, position] = self.sid_embeddings[prev_level](
                target_sids[:, prev_level]
            )

        positions = torch.arange(
            self.num_levels,
            device=target_sids.device,
        ).unsqueeze(0)

        decoder_input = decoder_input + self.decoder_position_embedding(positions)

        return decoder_input

    def decode(
        self,
        memory,
        memory_padding_mask,
        target_sids,
    ):
        decoder_input = self.make_decoder_input(target_sids)

        length = self.num_levels

        #         BOS   c1   c2   c3
        #
        # BOS      ✓     X    X    X
        # c1       ✓     ✓    X    X
        # c2       ✓     ✓    ✓    X
        # c3       ✓     ✓    ✓    ✓
        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=decoder_input.device,
            ),
            diagonal=1,  # allow to look at itself
        )

        # Decoder uses two sources of information:
        # 1) Self-attention over generated SID codes.
        # 2) Cross-attention over user history.
        #                     user history
        #                         ↓
        #                       encoder
        #                         ↓
        #                       memory
        #                         ↓
        #                         │
        # BOS → c1 → c2 ────── decoder
        #                         ↓
        #                     predict c3
        hidden = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        hidden = self.decoder_norm(hidden)

        logits = [
            head(hidden[:, level]) for level, head in enumerate(self.output_heads)
        ]

        return logits

    def forward(
        self,
        item_seq,
        target_item_ids,
    ):
        memory, padding_mask = self.encode_history(item_seq)

        target_sids = self.item_sids[target_item_ids]

        logits = self.decode(
            memory,
            padding_mask,
            target_sids,
        )

        return logits, target_sids


def generative_loss(logits, target_sids):
    """
    Average over Cross-Entropies per level.
    """
    losses = []

    for level, level_logits in enumerate(logits):
        losses.append(
            F.cross_entropy(
                level_logits.float(),
                target_sids[:, level],
            )
        )

    return torch.stack(losses).mean()
