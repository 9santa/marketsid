from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl


DATA_DIR = Path("data/processed")

embeddings = np.load(DATA_DIR / "item_text_embeddings.npy")
codes = np.load(DATA_DIR / "item_semantic_ids_content.npy")
catalog = pl.read_parquet(DATA_DIR / "catalog_model.parquet").sort("item_idx")

assert len(embeddings) == len(codes) == len(catalog)

SEED = 42
rng = np.random.default_rng(SEED)


def sample_prefix_pairs(
    codes,
    prefix_len,
    n_pairs=20_000,
):
    rng = np.random.default_rng(SEED)

    groups = defaultdict(list)

    for idx, code in enumerate(codes):
        key = tuple(code[:prefix_len])
        groups[key].append(idx)

    usable = [indices for indices in groups.values() if len(indices) >= 2]

    similarities = []

    while len(similarities) < n_pairs:
        group = usable[rng.integers(len(usable))]

        i, j = rng.choice(group, size=2, replace=False)

        similarities.append(float(embeddings[i] @ embeddings[j]))

    return np.asarray(similarities)


def sample_random_pairs(
    n_items,
    n_pairs=20_000,
    seed=SEED,
):
    rng = np.random.default_rng(seed)

    i = rng.integers(
        0,
        n_items,
        size=n_pairs,
    )

    j = rng.integers(
        0,
        n_items,
        size=n_pairs,
    )

    mask = i == j

    while mask.any():
        j[mask] = rng.integers(
            0,
            n_items,
            size=mask.sum(),
        )
        mask = i == j

    return np.sum(
        embeddings[i] * embeddings[j],
        axis=1,
    )


random_sim = sample_random_pairs(len(codes))

print("\n=== SEMANTIC LOCALITY ===")

print(
    "random:",
    f"{random_sim.mean():.4f}",
)

for prefix_len in (1, 2, 3):
    sims = sample_prefix_pairs(
        codes,
        prefix_len,
    )

    print(
        f"prefix={prefix_len}: "
        f"mean={sims.mean():.4f} "
        f"median={np.median(sims):.4f} "
        f"p10={np.quantile(sims, 0.1):.4f} "
        f"p90={np.quantile(sims, 0.9):.4f}"
    )


semantic_codes = codes[:, :3]

groups = defaultdict(list)

for idx, sid in enumerate(semantic_codes):
    groups[tuple(sid)].append(idx)

largest_groups = sorted(
    groups.items(),
    key=lambda x: len(x[1]),
    reverse=True,
)[:10]

print("\n=== LARGEST COLLISION GROUPS ===")

for sid, indices in largest_groups[:5]:
    print()
    print("=" * 80)
    print(
        "SID:",
        sid,
        "size:",
        len(indices),
    )

    for idx in indices[:8]:
        row = catalog.row(
            idx,
            named=True,
        )

        print(
            "-",
            row["item_text"][:160].replace("\n", " "),
        )


group_sizes = np.asarray([len(v) for v in groups.values()])

print("\n=== COLLISION STRUCTURE ===")

print(
    "Unique semantic SIDs:",
    len(groups),
)

print(
    "Singleton groups:",
    np.sum(group_sizes == 1),
)

print(
    "Groups with collision:",
    np.sum(group_sizes > 1),
)

print(
    "Items in collision groups:",
    group_sizes[group_sizes > 1].sum(),
)

print(
    "Max group size:",
    group_sizes.max(),
)

print(
    "Mean collided group size:",
    group_sizes[group_sizes > 1].mean(),
)
