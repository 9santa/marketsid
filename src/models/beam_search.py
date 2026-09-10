"""Optimized catalog-constrained beam search.

Main optimizations:
- Keep beam state as GPU tensors instead of Python lists/tuples.
- Replace Python trie traversal with tensor transition tables.
- Replace nested Python candidate loops with tensor operations.
- Replace Python sorting with torch.topk().
- Keep scores on GPU until generation on all levels is finished.
"""

import torch
import torch.nn.functional as F


@torch.inference_mode()
def constrained_beam_search(
    model,
    trie,
    item_seq: torch.Tensor,
    beam_size: int = 100,
    *,
    decode_batch_size=2048,
) -> list:
    """Single-user wrapper around the batched implementation.

    Args:
        item_seq:
            Tensor [1, history_len].

        beam_size:
            Number of beams to keep.

        decode_batch_size:
            Maximum number of user/beam rows passed through the decoder
            at once. This limits GPU memory usage.

    Returns:
        list[(item_id, log_probability)]
    """
    if item_seq.ndim != 2 or item_seq.shape[0] != 1:
        raise ValueError("Single-user search expects item_seq shape [1, L]")

    return batched_constrained_beam_search(
        model=model,
        trie=trie,
        item_seq=item_seq,
        beam_size=beam_size,
        decode_batch_size=decode_batch_size,
    )[0]


def filter_seen_items(results, history, k=50):
    """
    Filter the full history before taking top-k, preserving beam scores/order.

    May return fewer than k items if the beam contains too few unseen items.
    """
    seen = set(history)
    return [(item_id, score) for item_id, score in results if item_id not in seen][:k]


@torch.inference_mode()
def batched_constrained_beam_search(
    model,
    trie,
    item_seq: torch.Tensor,
    beam_size: int = 100,
    *,
    decode_batch_size=2048,
) -> list[list]:
    """Catalog-constrained beam search for many users at once.

    Naive version conceptually did:

        for level:
            for user:
                decode all beams for this user

                for beam:
                    allowed = trie.allowed_tokens(prefix)

                    for token in allowed:
                        score = float(log_probs[beam, token])
                        candidates.append(...)

                candidates.sort(...)
                keep first beam_size

    This implementation does the same logic as tensors:

        decoder
            ↓
        [B, beam, vocab] log probabilities
            ↓
        tensor trie mask
            ↓
        add current beam scores
            ↓
        GPU topk
            ↓
        next beam state

    Args:
        model:
            Generative recommender.

        trie:
            SIDTrie supporting:
                trie.tensors(device, vocab_sizes)

        item_seq:
            [B, history_len]

        beam_size:
            Maximum beams per user.

        decode_batch_size:
            How many flattened user/beam rows to decode simultaneously.

    Returns:
        list[list[(item_id, log_probability)]]
    """
    if beam_size <= 0:
        raise ValueError("beam_size must be positive")

    if decode_batch_size <= 0:
        raise ValueError("decode_batch_size must be positive")

    if item_seq.ndim != 2 or item_seq.shape[0] == 0:
        raise ValueError("Expected a non-empty item_seq with shape [B, L]")

    model.eval()

    device = item_seq.device
    batch_size = item_seq.shape[0]

    # 1. Encode user history once
    memory, padding_mask = model.encode_history(item_seq)

    # 2. Get tensor representation of the trie.
    # transitions[level]: [number_of_parent_nodes, vocab_size_at_this_level]
    # transitions[level][node, token] contains next trie node or -1 if that token is not allowed
    transitions, leaves = trie.tensors(device, model.vocab_sizes)

    # 3. Initial beam state.
    # prefixes: [B, beam_width, current_prefix_length]
    # nodes: [B, beam_width]
    # scores: [B, beam_width]

    # Every user Initially has: prefix = [], trie node = root (0), score = 0
    prefixes = torch.empty(
        size=(batch_size, 1, 0),
        dtype=torch.long,
        device=device,
    )

    nodes = torch.zeros(
        size=(batch_size, 1),
        dtype=torch.long,
        device=device,
    )

    scores = torch.zeros(
        size=(batch_size, 1),
        dtype=torch.float32,
        device=device,
    )

    # Generate one SID token per level
    for level, transition_table in enumerate(transitions):
        beam_width = scores.shape[1]

        # 1. Decode all current beams
        flat_prefixes = prefixes.reshape(
            batch_size * beam_width,
            level,
        )

        logits_chunks = []

        for start in range(0, batch_size * beam_width, decode_batch_size):
            stop = min(start + decode_batch_size, batch_size * beam_width)

            owners = (
                torch.arange(
                    start,
                    stop,
                    device=device,
                )
                // beam_width
            )

            chunk_logits = model.next_token_logits(
                memory[owners],
                padding_mask[owners],
                flat_prefixes[start:stop],
            )

            logits_chunks.append(chunk_logits.float())

        # [B, beam_width, vocab]
        logits = torch.cat(
            logits_chunks,
            dim=0,
        ).reshape(
            batch_size,
            beam_width,
            -1,
        )

        # 2. Determine legal trie children
        children = transition_table[nodes]

        valid = children.ge(0)

        # 3. Compute token probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        log_probs = log_probs.masked_fill(~valid, -torch.inf)

        # 4. Expand all beams
        candidates = scores.unsqueeze(-1) + log_probs

        # 5. Select the best beams
        flat_candidates = candidates.flatten(1)

        next_width = min(beam_size, flat_candidates.shape[1])

        scores, flat_indices = flat_candidates.topk(
            next_width,
            dim=1,
        )

        vocab_size = logits.shape[-1]

        # 6. Recover the winning parent beam + token
        parent_beams = flat_indices // vocab_size
        next_tokens = flat_indices % vocab_size

        # 7. Get the prefix belonging to each winning parent beam
        parent_prefixes = prefixes.gather(
            1,
            parent_beams.unsqueeze(-1).expand(
                -1,
                -1,
                level,
            ),
        )

        # Append the winning token
        prefixes = torch.cat(
            (
                parent_prefixes,
                next_tokens.unsqueeze(-1),
            ),
            dim=-1,
        )

        nodes = (
            children.flatten(1)
            .gather(
                1,
                flat_indices,
            )
            .clamp_min(0)
        )

    item_ids = leaves[nodes].cpu()
    final_scores = scores.cpu()

    results = []

    for user_ids, user_scores in zip(item_ids, final_scores):
        finite = torch.isfinite(user_scores)

        results.append(
            list(
                zip(
                    user_ids[finite].tolist(),
                    user_scores[finite].tolist(),
                )
            )
        )

    return results


@torch.inference_mode()
def naive_constrained_beam_search(
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


@torch.inference_mode()
def naive_batched_constrained_beam_search(
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
