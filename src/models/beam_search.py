import torch
import torch.nn.functional as F


@torch.inference_mode()
def constrained_beam_search(
    model,
    trie,
    item_seq,
    beam_size=200,
):
    """
    item_seq: [1, max_len]

    returns: list[(item_id, log_probability)]
    """
    model.eval()

    memory, padding_mask = model.encode_history(item_seq)

    beams = [([], 0.0)]

    for level in range(model.num_levels):
        candidates = []

        for prefix, score in beams:
            logits = model.next_token_logits(
                memory,
                padding_mask,
                prefix,
            )

            log_probs = F.log_softmax(
                logits[0].float(),
                dim=-1,
            )

            allowed = trie.allowed_tokens(prefix)

            for token in allowed:
                candidates.append(
                    (
                        prefix + [token],
                        score + float(log_probs[token]),
                    )
                )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        beams = candidates[:beam_size]

    results = [
        (
            trie.item_id(prefix),
            score,
        )
        for prefix, score in beams
    ]

    return results


def filter_seen_items(results, history, k=50):
    """Filter the full history before taking top-k, preserving beam scores/order.

    May return fewer than k items if the beam contains too few unseen items.
    """
    seen = set(history)
    return [(item_id, score) for item_id, score in results if item_id not in seen][:k]


@torch.inference_mode()
def batched_constrained_beam_search(
    model,
    trie,
    item_seq,
    beam_size=100,
):
    """
    item_seq: [B, L]

    returns:
        list[list[(item_id, score)]]
    """
    model.eval()

    memory, padding_mask = model.encode_history(item_seq)

    batch_size = item_seq.shape[0]

    # Для каждого пользователя отдельный beam.
    beams = [[([], 0.0)] for _ in range(batch_size)]

    for level in range(model.num_levels):
        next_beams = []

        for batch_idx in range(batch_size):
            user_candidates = []

            user_memory = memory[batch_idx : batch_idx + 1]

            user_padding_mask = padding_mask[batch_idx : batch_idx + 1]

            prefixes = [prefix for prefix, _ in beams[batch_idx]]

            scores = [score for _, score in beams[batch_idx]]

            # Группируем beams одной длины.
            # На данном уровне длина prefix у всех одинакова.
            n_beams = len(prefixes)

            repeated_memory = user_memory.expand(
                n_beams,
                -1,
                -1,
            )

            repeated_padding = user_padding_mask.expand(
                n_beams,
                -1,
            )

            decoder_input = torch.zeros(
                n_beams,
                level + 1,
                model.d_model,
                device=item_seq.device,
            )

            decoder_input[:, 0] = model.bos_embedding

            for position in range(1, level + 1):
                tokens = torch.tensor(
                    [prefix[position - 1] for prefix in prefixes],
                    dtype=torch.long,
                    device=item_seq.device,
                )

                decoder_input[:, position] = model.sid_embeddings[position - 1](tokens)

            positions = torch.arange(
                level + 1,
                device=item_seq.device,
            ).unsqueeze(0)

            decoder_input = decoder_input + model.decoder_position_embedding(positions)

            length = level + 1

            causal_mask = torch.triu(
                torch.ones(
                    length,
                    length,
                    dtype=torch.bool,
                    device=item_seq.device,
                ),
                diagonal=1,
            )

            hidden = model.decoder(
                tgt=decoder_input,
                memory=repeated_memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=repeated_padding,
            )

            hidden = model.decoder_norm(hidden)

            logits = model.output_heads[level](hidden[:, -1, :])

            log_probs = F.log_softmax(
                logits.float(),
                dim=-1,
            )

            for beam_idx, prefix in enumerate(prefixes):
                allowed = trie.allowed_tokens(prefix)

                base_score = scores[beam_idx]

                for token in allowed:
                    user_candidates.append(
                        (
                            prefix + [token],
                            base_score
                            + float(
                                log_probs[
                                    beam_idx,
                                    token,
                                ]
                            ),
                        )
                    )

            user_candidates.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            next_beams.append(user_candidates[:beam_size])

        beams = next_beams

    results = []

    for user_beams in beams:
        results.append(
            [
                (
                    trie.item_id(prefix),
                    score,
                )
                for prefix, score in user_beams
            ]
        )

    return results
