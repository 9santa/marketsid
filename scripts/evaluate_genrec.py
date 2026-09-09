"""Evaluate GenRec retrieval against the full catalog using constrained beams."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.data.sasrec_dataset import _pad_left
from src.models.beam_search import (
    batched_constrained_beam_search,
    filter_seen_items,
)
from src.models.generative_recommender import GenerativeRecommender
from src.models.sid_trie import SIDTrie


def make_eval_records(
    sequences,
    train_counts,
    split,
):
    records = []

    for items in sequences:
        items = np.asarray(
            items,
            dtype=np.int64,
        )

        if split == "valid":
            history = items[:-2]
            target = int(items[-2])

        elif split == "test":
            history = items[:-1]
            target = int(items[-1])

        else:
            raise ValueError(split)

        count = int(train_counts[target])

        if count == 0:
            group = "unseen"
        elif count <= 5:
            group = "tail"
        elif count <= 20:
            group = "mid"
        else:
            group = "head"

        records.append(
            {
                "history": history,
                "target": target,
                "group": group,
            }
        )

    return records


@torch.inference_mode()
def evaluate_genrec(
    model,
    trie,
    records,
    max_len,
    device,
    beam_size=200,
    k_values=(10, 20, 50),
    batch_size=64,
):
    model.eval()

    groups = [
        "all",
        "warm",
        "tail",
        "mid",
        "head",
        "unseen",
    ]

    totals = {
        group: {
            "n": 0,
            **{f"hits@{k}": 0 for k in k_values},
            **{f"ndcg@{k}": 0.0 for k in k_values},
        }
        for group in groups
    }

    max_k = max(k_values)
    short_recommendations = 0

    for start in range(
        0,
        len(records),
        batch_size,
    ):
        batch = records[start : start + batch_size]

        histories = np.stack(
            [
                _pad_left(
                    row["history"],
                    max_len,
                )
                for row in batch
            ]
        )

        item_seq = torch.from_numpy(histories).to(device)

        results = batched_constrained_beam_search(
            model=model,
            trie=trie,
            item_seq=item_seq,
            beam_size=beam_size,
        )

        for row, generated in zip(
            batch,
            results,
        ):
            recommendations = [
                item_id
                for item_id, _ in filter_seen_items(generated, row["history"], max_k)
            ]
            short_recommendations += len(recommendations) < max_k

            target = row["target"]
            group = row["group"]

            eval_groups = [
                "all",
                group,
            ]

            if group != "unseen":
                eval_groups.append("warm")

            for g in eval_groups:
                totals[g]["n"] += 1

            if target not in recommendations:
                continue

            rank = recommendations.index(target)

            for k in k_values:
                if rank >= k:
                    continue

                for g in eval_groups:
                    totals[g][f"hits@{k}"] += 1

                    totals[g][f"ndcg@{k}"] += 1.0 / math.log2(rank + 2)

        print(f"Evaluated {start + len(batch):,}/{len(records):,} users", flush=True)

    metrics = {}

    for group, values in totals.items():
        n = values["n"]

        if n == 0:
            continue

        metrics[group] = {"n": n}

        for k in k_values:
            metrics[group][f"recall@{k}"] = values[f"hits@{k}"] / n

            metrics[group][f"ndcg@{k}"] = values[f"ndcg@{k}"] / n

    print(f"Users with fewer than {max_k} recommendations: {short_recommendations}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--beam-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N users")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.beam_size < 50
        or args.batch_size <= 0
        or (args.limit is not None and args.limit <= 0)
    ):
        parser.error("Require beam-size >= 50, batch-size > 0 and limit > 0")

    data_dir = Path("data/processed")
    checkpoint_path = Path("checkpoints/genrec_content_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    vocab_sizes = checkpoint["vocab_sizes"]
    semantic_ids = np.load(data_dir / "item_semantic_ids_content.npy")
    catalog = pl.read_parquet(
        data_dir / "catalog_model.parquet", columns=["item_idx"]
    ).sort("item_idx")
    num_items = len(catalog)
    assert catalog["item_idx"].to_list() == list(range(1, num_items + 1))
    assert semantic_ids.shape == (num_items, len(vocab_sizes))
    assert np.all(semantic_ids >= 0) and np.all(semantic_ids < np.asarray(vocab_sizes))
    assert len(np.unique(semantic_ids, axis=0)) == num_items, "SID collisions"

    item_sids = np.zeros((num_items + 1, len(vocab_sizes)), dtype=np.int64)
    item_sids[1:] = semantic_ids
    trie = SIDTrie(item_sids)
    model = GenerativeRecommender(
        item_sids=torch.from_numpy(item_sids),
        vocab_sizes=vocab_sizes,
        max_len=config["max_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    interactions = pl.read_parquet(
        data_dir / "interactions_model.parquet", columns=["item_idx", "split"]
    )
    train_ids = interactions.filter(pl.col("split") == "train")["item_idx"].to_numpy()
    train_counts = np.bincount(train_ids.astype(np.int64), minlength=num_items + 1)
    users = pl.read_parquet(data_dir / "user_sequences.parquet").sort("user_idx")
    if args.limit is not None:
        users = users.head(args.limit)
    for row in users.iter_rows(named=True):
        assert row["split_sequence"] == ["train"] * (len(row["item_sequence"]) - 2) + [
            "valid",
            "test",
        ], f"Invalid split order for user_idx={row['user_idx']}"
    records = make_eval_records(
        users["item_sequence"].to_list(), train_counts, args.split
    )
    print(
        f"Split: {args.split} | users: {len(records):,} | full catalog: {num_items:,} items"
    )

    metrics = evaluate_genrec(
        model,
        trie,
        records,
        config["max_len"],
        device,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
    )
    print(json.dumps(metrics, indent=2))

    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    output = args.output or Path(f"artifacts/genrec_{args.split}{suffix}_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint["epoch"],
                "split": args.split,
                "candidate_catalog": "all",
                "num_users": len(records),
                "beam_size": args.beam_size,
                "batch_size": args.batch_size,
                "metrics": metrics,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
