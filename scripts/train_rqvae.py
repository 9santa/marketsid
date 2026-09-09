from pathlib import Path
import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# from models.rqvae import RQVAE

from src.models.rqvae import RQVAE


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings = np.load(DATA_DIR / "item_text_embeddings.npy").astype(np.float32)

    print("Embeddings:", embeddings.shape)
    print("Device:", device)

    dataset = TensorDataset(torch.from_numpy(embeddings))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    config = {
        "input_dim": embeddings.shape[1],
        "hidden_dim": 256,
        "latent_dim": 128,
        "num_codebooks": 3,
        "codebook_size": 256,
        "beta": 0.25,
    }

    model = RQVAE(**config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_reconstruction = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_reconstruction = 0.0
        total_vq = 0.0
        total_items = 0

        for (x,) in loader:
            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(x)
            loss = output["loss"]

            loss.backward()
            optimizer.step()

            n = len(x)
            total_loss += loss.item() * n
            total_reconstruction += output["reconstruction_loss"].item() * n
            total_vq += output["vq_loss"].item() * n
            total_items += n

        mean_loss = total_loss / total_items
        mean_reconstruction = total_reconstruction / total_items
        mean_vq = total_vq / total_items

        print(
            f"Epoch {epoch:02d} | "
            f"loss={mean_loss:.6f} | "
            f"recon={mean_reconstruction:.6f} | "
            f"vq={mean_vq:.6f}"
        )

        if mean_reconstruction < best_reconstruction:
            best_reconstruction = mean_reconstruction

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "reconstruction_loss": best_reconstruction,
                },
                CHECKPOINT_DIR / "rqvae_content_best.pt",
            )

    print("\nBest reconstruction:", best_reconstruction)


if __name__ == "__main__":
    main()
