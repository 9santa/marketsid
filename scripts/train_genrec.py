from pathlib import Path
import json
import random

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from src.models.generative_recommender import (
    GenerativeRecommender,
)
from src.data.genrec_dataset import (
    GenRecTrainDataset,
    _pad_left,
)


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")


CONFIG = {
    "max_len": 50,
    "d_model": 128,
    "n_heads": 4,
    "n_encoder_layers": 3,
    "n_decoder_layers": 2,
    "dropout": 0.2,
    "batch_size": 2048,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 20,
    "patience": 3,
    "seed": 42,
}


# ============================================================
# DATA
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAINING EPOCH
# ============================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for (
        item_seq,
        target_ids,
    ) in loader:
        item_seq = item_seq.to(
            device,
            non_blocking=True,
        )

        target_ids = target_ids.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits, target_sids = model(
                item_seq,
                target_ids,
            )

            losses = []

            for level, level_logits in enumerate(logits):
                losses.append(
                    F.cross_entropy(
                        level_logits.float(),
                        target_sids[:, level],
                    )
                )

            loss = torch.stack(losses).mean()

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        scaler.step(optimizer)

        scaler.update()

        batch_size = len(item_seq)

        total_loss += loss.item() * batch_size

        total_examples += batch_size

    return total_loss / total_examples


@torch.inference_mode()
def evaluate_teacher_forced(
    model,
    sequences,
    device,
    max_len,
    batch_size=512,
):
    model.eval()

    histories = []
    targets = []

    for sequence in sequences:
        sequence = np.asarray(
            sequence,
            dtype=np.int64,
        )

        history = sequence[:-2]
        target = sequence[-2]

        histories.append(
            _pad_left(
                history,
                max_len,
            )
        )

        targets.append(target)

    total = 0

    correct = np.zeros(
        model.num_levels,
        dtype=np.int64,
    )

    total_loss = 0.0

    for start in range(
        0,
        len(histories),
        batch_size,
    ):
        x = torch.from_numpy(np.stack(histories[start : start + batch_size])).to(device)

        target_ids = torch.as_tensor(
            targets[start : start + batch_size],
            device=device,
        )

        logits, target_sids = model(
            x,
            target_ids,
        )

        batch_loss = []

        for level, level_logits in enumerate(logits):
            batch_loss.append(
                F.cross_entropy(
                    level_logits.float(),
                    target_sids[:, level],
                )
            )

            predictions = level_logits.argmax(dim=-1)

            correct[level] += (predictions == target_sids[:, level]).sum().item()

        loss = torch.stack(batch_loss).mean()

        n = len(x)

        total_loss += loss.item() * n

        total += n

    return {
        "loss": (total_loss / total),
        "accuracy": (correct / total).tolist(),
    }


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "Device:",
        device,
    )

    catalog = pl.read_parquet(DATA_DIR / "catalog_model.parquet").sort("item_idx")

    sequences_df = pl.read_parquet(DATA_DIR / "user_sequences.parquet").sort("user_idx")

    sequences = sequences_df["item_sequence"].to_list()

    semantic_ids = np.load(DATA_DIR / "item_semantic_ids_content.npy")

    assert len(semantic_ids) == len(catalog)

    num_items = len(catalog)

    # row 0 = PAD item
    item_sids = np.zeros(
        (
            num_items + 1,
            semantic_ids.shape[1],
        ),
        dtype=np.int64,
    )

    item_sids[1:] = semantic_ids

    vocab_sizes = [
        int(semantic_ids[:, level].max()) + 1 for level in range(semantic_ids.shape[1])
    ]

    print(
        "Vocab sizes:",
        vocab_sizes,
    )

    # ============================================================
    # DATASET
    # ============================================================
    train_dataset = GenRecTrainDataset(
        sequences=sequences,
        max_len=CONFIG["max_len"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=(device.type == "cuda"),
    )

    print(
        "Training examples:",
        len(train_dataset),
    )

    # ============================================================
    # MODEL
    # ============================================================
    model = GenerativeRecommender(
        item_sids=torch.from_numpy(item_sids),
        vocab_sizes=vocab_sizes,
        max_len=CONFIG["max_len"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_encoder_layers=CONFIG["n_encoder_layers"],
        n_decoder_layers=CONFIG["n_decoder_layers"],
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

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(
        1,
        CONFIG["epochs"] + 1,
    ):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
        )

        val = evaluate_teacher_forced(
            model,
            sequences,
            device,
            CONFIG["max_len"],
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train={train_loss:.4f} | "
            f"val={val['loss']:.4f} | "
            f"acc=" + "/".join(f"{x:.3f}" for x in val["accuracy"])
        )

        if val["loss"] < best_loss:
            best_loss = val["loss"]
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": CONFIG,
                    "vocab_sizes": vocab_sizes,
                    "epoch": epoch,
                    "val_loss": best_loss,
                },
                CHECKPOINT_DIR / "genrec_content_best.pt",
            )

            print("  Saved best checkpoint")

        else:
            patience_counter += 1

            if patience_counter >= CONFIG["patience"]:
                print("Early stopping")
                break


if __name__ == "__main__":
    main()
