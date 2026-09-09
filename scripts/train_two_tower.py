from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import random
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

from src.models.content_two_tower import ContentTwoTower
from src.data.sasrec_dataset import _pad_left
from src.data.two_tower_dataset import NextItemDataset
from src.data.embedding_cache import embedding_metadata, load_embedding_cache


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_history(history, path):
    # Keep the last complete snapshot if writing the next one is interrupted.
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_data():
    catalog = pl.read_parquet(DATA_DIR / "catalog_model.parquet").sort("item_idx")

    interactions = pl.read_parquet(DATA_DIR / "interactions_model.parquet")

    sequences_df = pl.read_parquet(DATA_DIR / "user_sequences.parquet").sort("user_idx")

    embeddings = load_embedding_cache(
        DATA_DIR / "item_text_embeddings.npy", embedding_metadata(catalog)
    )

    num_items = len(catalog)

    assert embeddings.shape == (num_items, 384)
    assert catalog["item_idx"].to_list() == list(range(1, num_items + 1))

    features = np.zeros(
        (num_items + 1, embeddings.shape[1]),
        dtype=np.float32,
    )
    features[1:] = embeddings

    # Popularity on train
    counts_df = (
        interactions.filter(pl.col("split") == "train")
        .group_by("item_idx")
        .agg(pl.len().alias("n"))
    )

    train_counts = np.zeros(
        num_items + 1,
        dtype=np.int64,
    )

    train_counts[counts_df["item_idx"].to_numpy()] = counts_df["n"].to_numpy()

    warm_ids = np.flatnonzero(train_counts > 0)
    all_ids = np.arange(1, num_items + 1)

    sequences = []

    for row in sequences_df.iter_rows(named=True):
        items = np.asarray(
            row["item_sequence"],
            dtype=np.int64,
        )

        splits = row["split_sequence"]

        assert splits == (["train"] * (len(items) - 2) + ["valid", "test"]), (
            f"Invalid split order for user_idx={row['user_idx']}: {splits}"
        )

        sequences.append(items)

    print("Users:", len(sequences))
    print("Items:", num_items)
    print("Train items:", len(warm_ids))
    print("Features:", features.shape)

    mapping_hash = hashlib.sha256(
        json.dumps(catalog.select("item_idx", "parent_asin").rows()).encode("utf-8")
    ).hexdigest()

    return (
        torch.from_numpy(features),
        sequences,
        train_counts,
        warm_ids,
        all_ids,
        mapping_hash,
    )


def make_eval_records(
    sequences,
    train_counts,
    split,
):
    records = []

    for items in sequences:
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
def evaluate(
    model,
    records,
    candidate_ids,
    max_len,
    device,
    k_values=(10, 20, 50),
    batch_size=64,
):
    model.eval()

    candidate_ids = np.asarray(
        candidate_ids,
        dtype=np.int64,
    )

    candidate_tensor = torch.as_tensor(
        candidate_ids,
        device=device,
    )

    vectors = []

    for chunk in candidate_tensor.split(8192):
        vectors.append(model.item_vectors(chunk))

    candidate_vectors = torch.cat(
        vectors,
        dim=0,
    )

    candidate_pos = np.full(
        model.item_features.shape[0],
        -1,
        dtype=np.int64,
    )

    candidate_pos[candidate_ids] = np.arange(len(candidate_ids))

    totals = {
        group: {
            "n": 0,
            **{f"hits@{k}": 0 for k in k_values},
            **{f"ndcg@{k}": 0.0 for k in k_values},
        }
        for group in ["all", "warm", "tail", "mid", "head", "unseen"]
    }

    max_k = max(k_values)

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]

        histories = np.stack([_pad_left(r["history"], max_len) for r in batch])

        item_seq = torch.from_numpy(histories).to(device)

        user_vectors = model.encode(item_seq)

        scores = user_vectors @ candidate_vectors.T

        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite recommendation scores during evaluation")

        rows = []
        cols = []

        for row_idx, record in enumerate(batch):
            positions = candidate_pos[np.asarray(record["history"], dtype=np.int64)]

            positions = positions[positions >= 0]

            rows.extend([row_idx] * len(positions))
            cols.extend(positions.tolist())

        if rows:
            scores[
                torch.as_tensor(rows, device=device),
                torch.as_tensor(cols, device=device),
            ] = -torch.inf

        topk_positions = torch.topk(scores, k=max_k, dim=1).indices

        topk_items = candidate_tensor[topk_positions].cpu().numpy()

        for row_idx, record in enumerate(batch):
            target = record["target"]
            group = record["group"]

            groups = ["all", group]

            if group != "unseen":
                groups.append("warm")

            for g in groups:
                totals[g]["n"] += 1

            matches = np.flatnonzero(topk_items[row_idx] == target)

            if len(matches) == 0:
                continue

            rank = int(matches[0])

            for k in k_values:
                if rank >= k:
                    continue

                for g in groups:
                    totals[g][f"hits@{k}"] += 1
                    totals[g][f"ndcg@{k}"] += 1.0 / math.log2(rank + 2)

    result = {}

    for group, values in totals.items():
        n = values["n"]

        if n == 0:
            continue

        result[group] = {"n": n}

        for k in k_values:
            result[group][f"recall@{k}"] = values[f"hits@{k}"] / n
            result[group][f"ndcg@{k}"] = values[f"ndcg@{k}"] / n

    return result


def train_one_epoch(
    model: ContentTwoTower,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device,
):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for item_seq, pos_items, neg_items in loader:
        item_seq = item_seq.to(device, non_blocking=True)
        pos_items = pos_items.to(device, non_blocking=True)
        neg_items = neg_items.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(
                item_seq,
                pos_items,
                neg_items,
            )

            # The model places the positive item in candidate column zero.
            loss = F.cross_entropy(
                logits,
                torch.zeros(logits.size(0), dtype=torch.long, device=device),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")

    args = parser.parse_args()

    CONFIG = {
        "max_len": 50,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 3,
        "dropout": 0.2,
        "temperature": 0.1,
        "batch_size": args.batch_size,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": 42,
        "training_strategy": "all_prefixes",
    }

    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    (
        features,
        sequences,
        train_counts,
        warm_ids,
        all_ids,
        mapping_hash,
    ) = load_data()

    dataset = NextItemDataset(
        sequences=sequences,
        warm_ids=warm_ids,
        max_len=CONFIG["max_len"],
        seed=CONFIG["seed"],
    )

    train_loader = DataLoader(
        Subset(dataset, range(min(len(dataset), 2048))) if args.smoke else dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=False,
        multiprocessing_context="spawn",
        pin_memory=(device.type == "cuda"),
    )

    print("Training examples:", len(train_loader.dataset))
    print("Batches:", len(train_loader))

    model = ContentTwoTower(
        item_features=features,
        max_len=CONFIG["max_len"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        dropout=CONFIG["dropout"],
        temperature=CONFIG["temperature"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    valid_records = make_eval_records(
        sequences,
        train_counts,
        "valid",
    )

    test_records = make_eval_records(
        sequences,
        train_counts,
        "test",
    )

    if args.smoke:
        valid_records = valid_records[:512]
        test_records = test_records[:512]

    checkpoint_path = CHECKPOINT_DIR / (
        "two_tower_smoke.pt" if args.smoke else "two_tower_best.pt"
    )

    best_metric = -1.0
    best_epoch = -1
    patience_counter = 0

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    history_path = CHECKPOINT_DIR / f"two_tower_history_{run_id}.json"
    history = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "config": dict(CONFIG),
        "device": str(device),
        "smoke": args.smoke,
        "mapping_hash": mapping_hash,
        "num_users": len(sequences),
        "num_items": len(all_ids),
        "num_train_items": len(warm_ids),
        "num_train_examples": len(train_loader.dataset),
        "num_valid_records": len(valid_records),
        "num_test_records": len(test_records),
        "selection_metric": "valid.warm.recall@20",
        "best_epoch": None,
        "best_metric": None,
        "stopped_early": False,
        "epochs": [],
        "test_metrics": None,
    }
    save_history(history, history_path)
    print("Training history:", history_path)

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_started = time.perf_counter()
        dataset.set_epoch(epoch)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
        )

        valid_metrics = evaluate(
            model,
            valid_records,
            warm_ids,
            max_len=CONFIG["max_len"],
            device=device,
        )

        metric = valid_metrics["warm"]["recall@20"]

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val warm R@20={metric:.6f} | "
            f"NDCG@20="
            f"{valid_metrics['warm']['ndcg@20']:.6f}"
        )

        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": CONFIG,
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "mapping_hash": mapping_hash,
                    "history_file": str(history_path),
                },
                checkpoint_path,
            )

            print("  Saved best checkpoint")
        else:
            patience_counter += 1

        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_seconds": time.perf_counter() - epoch_started,
                "valid_metrics": valid_metrics,
                "is_best": best_epoch == epoch,
            }
        )
        history["best_epoch"] = best_epoch
        history["best_metric"] = best_metric
        history["stopped_early"] = patience_counter >= CONFIG["patience"]
        save_history(history, history_path)

        if history["stopped_early"]:
            print("Early stopping")
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    assert checkpoint["mapping_hash"] == mapping_hash

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nBest epoch:", best_epoch)

    # 1. Сопоставимый с SASRec warm-catalog benchmark.
    warm_metrics = evaluate(
        model,
        test_records,
        warm_ids,
        max_len=CONFIG["max_len"],
        device=device,
    )

    # 2. Полный каталог: unseen items теперь доступны.
    all_metrics = evaluate(
        model,
        test_records,
        all_ids,
        max_len=CONFIG["max_len"],
        device=device,
    )

    history["test_metrics"] = {
        "warm_catalog": warm_metrics,
        "all_catalog": all_metrics,
    }
    history["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_history(history, history_path)

    print("\n=== WARM CATALOG ===")
    print(json.dumps(warm_metrics, indent=2))

    print("\n=== ALL CATALOG ===")
    print(json.dumps(all_metrics, indent=2))

    if not args.smoke:
        with open(
            CHECKPOINT_DIR / "two_tower_metrics.json",
            "w",
        ) as f:
            json.dump(
                {
                    "config": CONFIG,
                    "best_epoch": best_epoch,
                    "warm_catalog": warm_metrics,
                    "all_catalog": all_metrics,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
