from pathlib import Path

import math

import polars as pl


DATA_DIR = Path("data/processed")

K_VALUES = [10, 20, 50]


interactions = pl.read_parquet(DATA_DIR / "interactions_model.parquet")


# ============================================================
# TRAIN POPULARITY
# ============================================================

train = interactions.filter(pl.col("split") == "train")

popularity = (
    train.group_by("item_idx")
    .agg(pl.len().alias("count"))
    .sort(
        "count",
        descending=True,
    )
)


popular_items = popularity["item_idx"].to_list()


print("Train candidate items:", len(popular_items))


# ============================================================
# HISTORIES
# ============================================================

# Validation interactions are also observed before the test target.
test_histories = {
    row["user_idx"]: set(row["history"])
    for row in (
        interactions.filter(pl.col("split").is_in(["train", "valid"]))
        .group_by("user_idx")
        .agg(pl.col("item_idx").alias("history"))
        .iter_rows(named=True)
    )
}


# ============================================================
# TEST TARGETS
# ============================================================

test = interactions.filter(pl.col("split") == "test").select(
    "user_idx",
    "item_idx",
    "train_interactions",
    "train_popularity_group",
)


# ============================================================
# EVALUATION
# ============================================================

max_k = max(K_VALUES)

metrics = {
    k: {
        "hits": 0,
        "ndcg": 0.0,
    }
    for k in K_VALUES
}

warm_targets = 0
unseen_targets = 0

for row in test.iter_rows(named=True):
    user = row["user_idx"]
    target = row["item_idx"]

    if row["train_interactions"] == 0:
        unseen_targets += 1
        continue

    warm_targets += 1

    seen = test_histories[user]

    recommendations = []

    for item in popular_items:
        if item in seen:
            continue

        recommendations.append(item)

        if len(recommendations) == max_k:
            break

    for k in K_VALUES:
        topk = recommendations[:k]

        if target in topk:
            rank = topk.index(target)

            metrics[k]["hits"] += 1

            metrics[k]["ndcg"] += 1.0 / math.log2(rank + 2)  # +2 cos 0-indexed


print("\n=== POPULARITY BASELINE ===")

print(
    "Warm test targets:",
    warm_targets,
)

print(
    "Unseen test targets:",
    unseen_targets,
)

print(
    "Unseen fraction:",
    unseen_targets / (warm_targets + unseen_targets),
)


for k in K_VALUES:
    recall = metrics[k]["hits"] / warm_targets

    ndcg = metrics[k]["ndcg"] / warm_targets

    print(f"Recall@{k}: {recall:.6f}")

    print(f"NDCG@{k}: {ndcg:.6f}")
