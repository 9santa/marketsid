from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import polars as pl
import torch
from scipy.stats import entropy

from src.models.rqvae import RQVAE


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    CHECKPOINT_DIR / "rqvae_content_best.pt",
    map_location=device,
    weights_only=False,
)

model = RQVAE(**checkpoint["config"]).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

embeddings = np.load(DATA_DIR / "item_text_embeddings.npy").astype(np.float32)

catalog = pl.read_parquet(DATA_DIR / "catalog_model.parquet").sort("item_idx")

assert len(embeddings) == len(catalog)

all_codes = []

with torch.inference_mode():
    for start in range(0, len(embeddings), 4096):
        x = torch.from_numpy(embeddings[start : start + 4096]).to(device)

        codes = model.encode_codes(x)

        all_codes.append(codes.cpu().numpy())

codes = np.concatenate(all_codes, axis=0)

print("Codes shape:", codes.shape)

np.save(
    DATA_DIR / "item_semantic_ids.npy",
    codes.astype(np.int16),
)

num_items, num_codebooks = codes.shape
codebook_size = checkpoint["config"]["codebook_size"]

unique_codes = np.unique(codes, axis=0)

print("\n=== SID DIAGNOSTICS ===")
print("Items:", num_items)
print("Unique SIDs:", len(unique_codes))
print("Collision rate:", 1 - len(unique_codes) / num_items)

for level in range(num_codebooks):
    counts = np.bincount(
        codes[:, level],
        minlength=codebook_size,
    )

    used = np.count_nonzero(counts)

    probs = counts / counts.sum()

    print(
        f"Codebook {level + 1}: "
        f"used={used}/{codebook_size}, "
        f"entropy={entropy(probs):.3f}, "
        f"max_entropy={np.log(codebook_size):.3f}"
    )


counter = Counter(map(tuple, codes.tolist()))

print("\nMost common SIDs:")

for sid, count in counter.most_common(10):
    print(sid, count)


groups = defaultdict(list)

for row_idx, sid in enumerate(codes):
    groups[tuple(sid)].append(row_idx)

collision_tokens = np.zeros(len(codes), dtype=np.int16)

for sid, row_indices in groups.items():
    for collision_id, row_idx in enumerate(row_indices):
        collision_tokens[row_idx] = collision_id

max_collision_group = max(len(indices) for indices in groups.values())

print("Max collision group:", max_collision_group)


final_codes = np.column_stack(
    [
        codes,
        collision_tokens,
    ]
).astype(np.int16)

assert final_codes.shape == (len(codes), 4)

assert len(np.unique(final_codes, axis=0)) == len(final_codes)

np.save(DATA_DIR / "item_semantic_ids_content.npy", final_codes)

print("Final SID shape:", final_codes.shape)
print("Collision vocab:", final_codes[:, 3].max() + 1)
