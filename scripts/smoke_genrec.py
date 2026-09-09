"""Inspect constrained generation for a small validation sample."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
import torch

from src.data.sasrec_dataset import _pad_left
from src.models.beam_search import constrained_beam_search, filter_seen_items
from src.models.generative_recommender import GenerativeRecommender
from src.models.sid_trie import SIDTrie


def check_recommendations(results, history, item_sids, trie, k):
    ids = [item_id for item_id, _ in results]
    scores = [score for _, score in results]
    valid_ids = all(0 < item_id < len(item_sids) for item_id in ids)
    return {
        "all_sids_exist": valid_ids
        and all(trie.item_id(item_sids[item_id]) == item_id for item_id in ids),
        "no_duplicate_items": len(ids) == len(set(ids)),
        "no_pad": 0 not in ids,
        "no_seen_items": set(ids).isdisjoint(history),
        "exactly_k_recommendations": len(ids) == k,
        "finite_scores": all(np.isfinite(score) for score in scores),
        "scores_descending": all(a >= b for a, b in zip(scores, scores[1:])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/genrec_content_best.pt")
    )
    parser.add_argument("--num-users", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=200)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/genrec_smoke.json")
    )
    args = parser.parse_args()
    if args.num_users <= 0 or args.k <= 0 or args.beam_size < args.k:
        parser.error("Require num-users > 0 and beam-size >= k > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    catalog = pl.read_parquet(
        args.data_dir / "catalog_model.parquet",
        columns=["item_idx", "parent_asin", "title"],
    ).sort("item_idx")
    semantic_ids = np.load(args.data_dir / "item_semantic_ids_content.npy")
    vocab_sizes = checkpoint["vocab_sizes"]
    if catalog["item_idx"].to_list() != list(range(1, len(catalog) + 1)):
        raise ValueError("Catalog item IDs must be contiguous and start at 1")
    if semantic_ids.shape != (len(catalog), len(vocab_sizes)):
        raise ValueError("SID dimensions do not match catalog/checkpoint")
    if not np.issubdtype(semantic_ids.dtype, np.integer) or np.any(semantic_ids < 0):
        raise ValueError("SID tokens must be nonnegative integers")
    if np.any(semantic_ids >= np.asarray(vocab_sizes)):
        raise ValueError("SID tokens exceed checkpoint vocabulary sizes")
    if len(np.unique(semantic_ids, axis=0)) != len(catalog):
        raise ValueError("SID collisions: use the final IDs with collision tokens")

    item_sids = np.zeros((len(catalog) + 1, len(vocab_sizes)), dtype=np.int64)
    item_sids[1:] = semantic_ids
    trie = SIDTrie(item_sids)
    model = GenerativeRecommender(
        item_sids=torch.from_numpy(item_sids),
        vocab_sizes=vocab_sizes,
        **{
            key: config[key]
            for key in (
                "max_len",
                "d_model",
                "n_heads",
                "n_encoder_layers",
                "n_decoder_layers",
                "dropout",
            )
        },
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    users = (
        pl.read_parquet(
            args.data_dir / "user_sequences.parquet",
            columns=["user_idx", "item_sequence", "split_sequence"],
        )
        .sort("user_idx")
        .head(args.num_users)
    )
    if len(users) != args.num_users:
        raise ValueError(f"Requested {args.num_users} users, found {len(users)}")
    catalog_rows = {row["item_idx"]: row for row in catalog.iter_rows(named=True)}

    def describe(item_id):
        return {**catalog_rows[item_id], "sid": item_sids[item_id].tolist()}

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "device": str(device),
        "split": "valid",
        "candidate_catalog": "all",
        "beam_size": args.beam_size,
        "k": args.k,
        "num_users": len(users),
        "users": [],
    }
    lines = []

    def emit(line):
        lines.append(line)
        print(line, flush=True)

    emit(
        f"Device: {device} | validation users: {len(users)} | beam_size: {args.beam_size} | k: {args.k}"
    )
    emit(f"Candidate catalog: all {len(catalog)} items")
    for row in users.iter_rows(named=True):
        items, splits = row["item_sequence"], row["split_sequence"]
        if splits != ["train"] * (len(items) - 2) + ["valid", "test"]:
            raise ValueError(f"Invalid split order for user_idx={row['user_idx']}")
        history, target = items[:-2], items[-2]
        seen = set(history)
        if not history:
            raise ValueError(f"Empty history for user_idx={row['user_idx']}")
        item_seq = (
            torch.from_numpy(_pad_left(history, config["max_len"]))
            .unsqueeze(0)
            .to(device)
        )
        started = time.perf_counter()
        results = constrained_beam_search(
            model, trie, item_seq, beam_size=args.beam_size
        )
        recommendations = filter_seen_items(results, history, k=args.k)
        duration = time.perf_counter() - started
        checks = check_recommendations(
            recommendations, history, item_sids, trie, args.k
        )
        raw_checks = check_recommendations(results, [], item_sids, trie, len(results))
        checks["raw_beams_valid"] = all(raw_checks.values())
        generated = [
            {**describe(item_id), "score": score} for item_id, score in recommendations
        ]
        seen_count = sum(item_id in seen for item_id, _ in results)
        report["users"].append(
            {
                "user_idx": row["user_idx"],
                "history": [describe(item_id) for item_id in history],
                "target": describe(target),
                "generated": generated,
                "checks": checks,
                "num_beams": len(results),
                "num_seen_filtered": seen_count,
                "duration_seconds": duration,
            }
        )
        emit(
            f"\nUser {row['user_idx']} | {duration:.2f}s | beams={len(results)} | seen_filtered={seen_count}"
        )
        emit(f"history: {history}")
        target_row = describe(target)
        emit(f"target: item {target} | SID={target_row['sid']} | {target_row['title']}")
        emit("generated:")
        for rank, item in enumerate(generated, start=1):
            title = " ".join((item["title"] or "").split())
            emit(
                f"{rank}. item {item['item_idx']} | SID={item['sid']} | score={item['score']:.6f} | {title}"
            )
        emit(
            "checks: "
            + ", ".join(
                f"{name}={'PASS' if passed else 'FAIL'}"
                for name, passed in checks.items()
            )
        )

    report["all_checks_passed"] = all(
        all(user["checks"].values()) for user in report["users"]
    )
    report["total_generation_seconds"] = sum(
        user["duration_seconds"] for user in report["users"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    emit(f"\nAll checks passed: {report['all_checks_passed']} | JSON: {args.output}")
    args.output.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    if not report["all_checks_passed"]:
        raise SystemExit("Smoke checks failed; see per-user results in the report")


if __name__ == "__main__":
    main()
