from pathlib import Path
from datetime import datetime, timezone
import json
import math
import random
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.sasrec_dataset import (
    SASRecTrainDataset,
    _pad_left,
)

from src.models.sasrec import SASRec, sasrec_loss


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "max_len": 50,
    "d_model": 64,
    "n_heads": 8,
    "n_layers": 4,
    "dropout": 0.2,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 20,
    "patience": 3,
    "seed": 42,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_history(history, path):
    # Replace the previous snapshot only after the new JSON is fully written.
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


set_seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


interactions = pl.read_parquet(DATA_DIR / "interactions_model.parquet")

sequences_df = pl.read_parquet(DATA_DIR / "user_sequences.parquet").sort("user_idx")

sequences = sequences_df["item_sequence"].to_list()

num_items = interactions["item_idx"].max()

warm_item_ids = (
    interactions.filter(pl.col("split") == "train")["item_idx"]
    .unique()
    .sort()
    .to_numpy()
)

print("Users:", len(sequences))
print("All items:", num_items)
print("Train items:", len(warm_item_ids))


train_dataset = SASRecTrainDataset(
    sequences=sequences,
    warm_item_ids=warm_item_ids,
    max_len=CONFIG["max_len"],
    seed=CONFIG["seed"],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

print("Training examples:", len(train_dataset))
print("Batches:", len(train_loader))


model = SASRec(
    num_items=num_items,
    max_len=CONFIG["max_len"],
    d_model=CONFIG["d_model"],
    n_heads=CONFIG["n_heads"],
    n_layers=CONFIG["n_layers"],
    dropout=CONFIG["dropout"],
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


def train_one_epoch(
    model: SASRec,
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
            pos_logits, neg_logits = model(
                item_seq,
                pos_items,
                neg_items,
            )

            loss = sasrec_loss(
                pos_logits,
                neg_logits,
                pos_items,
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


def build_eval_records(interactions: pl.DataFrame, split: str):
    rows = (
        interactions.sort(["user_idx", "timestamp", "item_idx"])
        .group_by("user_idx", maintain_order=True)
        .agg(
            pl.col("item_idx").alias("items"),
            pl.col("split").alias("splits"),
            pl.col("train_popularity_group").alias("popularity_groups"),
        )
    )

    records = []

    for row in rows.iter_rows(named=True):
        items = row["items"]
        splits = row["splits"]
        groups = row["popularity_groups"]

        target_position = splits.index(split)

        history = items[:target_position]
        target = items[target_position]

        records.append(
            {
                "user_idx": row["user_idx"],
                "history": history,
                "target": target,
                "popularity_group": groups[target_position],
            }
        )

    return records


@torch.no_grad()
def evaluate(
    model: SASRec,
    records,
    warm_item_ids,
    max_len,
    device,
    k_values=(10, 20, 50),
    batch_size=256,
):
    model.eval()

    candidate_ids = torch.tensor(
        warm_item_ids,
        dtype=torch.long,
        device=device,
    )

    max_k = max(k_values)

    totals = {
        group: {
            "n": 0,
            **{f"hits@{k}": 0 for k in k_values},
            **{f"ndcg@{k}": 0.0 for k in k_values},
        }
        for group in ["all", "warm", "tail", "mid", "head", "unseen"]
    }

    warm_set = set(warm_item_ids.tolist())

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]

        histories = np.stack([_pad_left(r["history"], max_len) for r in batch])

        item_seq = torch.from_numpy(histories).to(device)

        scores = model.score_items(item_seq, candidate_ids)

        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite recommendation scores during evaluation")

        # Exclude already-seen items
        for row_idx, record in enumerate(batch):
            seen = set(record["history"])

            mask = torch.isin(
                candidate_ids,
                torch.as_tensor(
                    list(seen),
                    device=device,
                ),
            )

            scores[row_idx, mask] = -torch.inf

        topk_positions = torch.topk(
            scores,
            k=max_k,
            dim=1,
        ).indices

        topk_items = candidate_ids[topk_positions].cpu().numpy()

        for row_idx, record in enumerate(batch):
            target = record["target"]
            group = record["popularity_group"]

            groups = ["all", group]

            if target in warm_set:
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


valid_records = build_eval_records(interactions, split="valid")

test_records = build_eval_records(interactions, split="test")

best_metric = -1.0
best_epoch = -1
patience_counter = 0

started_at = datetime.now(timezone.utc)
run_id = started_at.strftime("%Y%m%dT%H%M%S%fZ")
history_path = CHECKPOINT_DIR / f"sasrec_history_{run_id}.json"
history = {
    "run_id": run_id,
    "started_at": started_at.isoformat(),
    "config": dict(CONFIG),
    "device": str(device),
    "num_users": len(sequences),
    "num_items": int(num_items),
    "num_train_items": len(warm_item_ids),
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
    train_dataset.set_epoch(epoch)

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
        warm_item_ids,
        max_len=CONFIG["max_len"],
        device=device,
    )

    metric = valid_metrics["warm"]["recall@20"]

    print(
        f"Epoch {epoch:02d} | "
        f"loss={train_loss:.4f} | "
        f"val R@20={metric:.6f} | "
        f"val NDCG@20="
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
                "history_file": str(history_path),
            },
            CHECKPOINT_DIR / "sasrec_best.pt",
        )

        print("Saved best checkpoint")

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
    CHECKPOINT_DIR / "sasrec_best.pt",
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])

test_metrics = evaluate(
    model,
    test_records,
    warm_item_ids,
    max_len=CONFIG["max_len"],
    device=device,
)

history["test_metrics"] = test_metrics
history["finished_at"] = datetime.now(timezone.utc).isoformat()
save_history(history, history_path)


print("\n=== TEST RESULTS ===")
print("Best epoch:", best_epoch)

for group, values in test_metrics.items():
    print(f"\n[{group}] n={values['n']}")

    for k in (10, 20, 50):
        print(f"R@{k}={values[f'recall@{k}']:.6f} NDCG@{k}={values[f'ndcg@{k}']:.6f}")
