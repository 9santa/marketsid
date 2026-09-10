from pathlib import Path
from datetime import datetime, timezone
import json
import random
import time
import argparse

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
RQ_CHECKPOINT = CHECKPOINT_DIR / "rqvae_content_best.pt"


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
    "epochs": 50,
    "patience": 5,
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


def save_history(history, path):
    # Replace the previous snapshot only after the new JSON is fully written.
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_rq_codebooks(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["config"]
    state = checkpoint["model_state_dict"]

    num_codebooks = config["num_codebooks"]

    latent_dim = config["latent_dim"]

    codebooks = []

    for level in range(num_codebooks):
        key = f"quantizer.codebooks.{level}.weight"

        if key not in state:
            raise KeyError(f"{key} not found in RQ-VAE checkpoint")

        weight = state[key].detach().float().clone()

        if weight.shape[1] != latent_dim:
            raise ValueError(f"Unexpected shape {weight.shape}")

        codebooks.append(weight)

    return codebooks


# ============================================================
# TRAINING EPOCH
# ============================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    variant_config: dict,
):
    model.train()

    total_rec_loss = 0.0
    total_anchor_loss = 0.0
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

            recommendation_loss = torch.stack(losses).mean()

            anchor_loss = model.anchor_loss()

            loss = recommendation_loss + variant_config["anchor_lambda"] * anchor_loss

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        scaler.step(optimizer)

        scaler.update()

        batch_size = len(item_seq)

        total_rec_loss += recommendation_loss.item() * batch_size
        total_anchor_loss += anchor_loss.item() * batch_size
        total_loss += loss.item() * batch_size

        total_examples += batch_size

    return {
        "loss": (total_loss / total_examples),
        "rec_loss": (total_rec_loss / total_examples),
        "anchor_loss": (total_anchor_loss / total_examples),
    }


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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        choices=["A", "B", "C", "D"],
        required=True,
    )

    parser.add_argument(
        "--anchor-lambda",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()

    VARIANTS = {
        "A": {
            "embedding_mode": "scratch",
            "tie_semantic_output": False,
            "anchor_lambda": 0.0,
        },
        "B": {
            "embedding_mode": "rq_init",
            "tie_semantic_output": False,
            "anchor_lambda": 0.0,
        },
        "C": {
            "embedding_mode": "rq_init",
            "tie_semantic_output": True,
            "anchor_lambda": 0.0,
        },
        "D": {
            "embedding_mode": "rq_anchor",
            "tie_semantic_output": True,
            "anchor_lambda": args.anchor_lambda,
        },
    }

    variant_config = VARIANTS[args.variant]

    print("Experiment variant:", args.variant)
    print(variant_config)

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

    print("Users:", len(sequences))
    print("Items:", num_items)

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

    rq_codebooks = load_rq_codebooks(RQ_CHECKPOINT)

    if args.variant == "A":
        rq_codebooks_for_model = None
    else:
        rq_codebooks_for_model = rq_codebooks

    for i, cb in enumerate(rq_codebooks):
        print(
            f"RQ codebook {i}:",
            tuple(cb.shape),
            "mean norm=",
            cb.norm(dim=1).mean().item(),
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
    print("Batches:", len(train_loader))

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
        embedding_mode=variant_config["embedding_mode"],
        tie_semantic_output=variant_config["tie_semantic_output"],
        rq_codebooks=(rq_codebooks_for_model),
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
    best_epoch = -1
    patience_counter = 0

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    history_path = CHECKPOINT_DIR / f"genrec_history_{run_id}.json"
    history = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "config": dict(CONFIG),
        "device": str(device),
        "vocab_sizes": vocab_sizes,
        "num_users": len(sequences),
        "num_items": num_items,
        "num_train_examples": len(train_dataset),
        "selection_metric": "valid.loss",
        "best_epoch": None,
        "best_metric": None,
        "stopped_early": False,
        "epochs": [],
    }
    save_history(history, history_path)
    print("Training history:", history_path)

    for epoch in range(
        1,
        CONFIG["epochs"] + 1,
    ):
        epoch_started = time.perf_counter()
        losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            variant_config,
        )

        val = evaluate_teacher_forced(
            model,
            sequences,
            device,
            CONFIG["max_len"],
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train={losses["loss"]:.4f} | "
            f"val={val['loss']:.4f} | "
            f"acc=" + "/".join(f"{x:.3f}" for x in val["accuracy"])
        )

        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_epoch = epoch
            patience_counter = 0

            CHECKPOINT_PATH = CHECKPOINT_DIR / f"genrec_content_{args.variant}_best.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": CONFIG,
                    "variant": args.variant,
                    "variant_config": variant_config,
                    "vocab_sizes": vocab_sizes,
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "semantic_drift": model.semantic_drift(),
                    "history_file": str(history_path),
                },
                CHECKPOINT_PATH,
            )

            print("  Saved best checkpoint")

        else:
            patience_counter += 1

        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": losses["loss"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_seconds": time.perf_counter() - epoch_started,
                "valid_metrics": val,
                "is_best": best_epoch == epoch,
            }
        )
        history["best_epoch"] = best_epoch
        history["best_metric"] = best_loss
        history["stopped_early"] = patience_counter >= CONFIG["patience"]
        save_history(history, history_path)

        if history["stopped_early"]:
            print("Early stopping")
            break

    history["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_history(history, history_path)
    print("\nBest epoch:", best_epoch)


if __name__ == "__main__":
    main()
