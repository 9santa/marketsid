from pathlib import Path
import polars as pl


DATA_DIR = Path("data/processed")

interactions = pl.read_parquet(DATA_DIR / "interactions.parquet")

catalog = pl.read_parquet(DATA_DIR / "catalog_metadata.parquet")


# ============================================================
# ITEM IDS
# ============================================================

# 0 is reserved for padding
item_mapping = (
    catalog.select("parent_asin")
    .sort("parent_asin")
    .with_row_index("item_idx", offset=1)
)

print("Items:", len(item_mapping))
print(item_mapping.head())


# ============================================================
# USER IDS
# ============================================================

user_mapping = (
    interactions.select("user_id").unique().sort("user_id").with_row_index("user_idx")
)

print("Users:", len(user_mapping))
print(user_mapping.head())


# ============================================================
# JOIN IDS
# ============================================================

interactions = interactions.join(
    item_mapping,
    on="parent_asin",
    how="inner",
).join(
    user_mapping,
    on="user_id",
    how="inner",
)

catalog = catalog.join(
    item_mapping,
    on="parent_asin",
    how="inner",
)


# ============================================================
# SANITY CHECKS
# ============================================================

assert interactions["item_idx"].min() >= 1
assert interactions["item_idx"].max() == len(item_mapping)

assert interactions["user_idx"].min() == 0
assert interactions["user_idx"].max() == len(user_mapping) - 1

print("\nInteractions:")
print(interactions.shape)

print("\nItem index range:")
print(
    interactions["item_idx"].min(),
    interactions["item_idx"].max(),
)


# ============================================================
# ITEM SEQUENCES FOR USERS
# ============================================================

sequences = (
    interactions.sort(
        [
            "user_idx",
            "timestamp",
            "item_idx",
        ]
    )
    .group_by(
        "user_idx",
        maintain_order=True,
    )
    .agg(
        pl.col("item_idx").alias("item_sequence"),
        pl.col("split").alias("split_sequence"),
        pl.col("rating").alias("rating_sequence"),
        pl.col("timestamp").alias("timestamp_sequence"),
    )
)

print("\n=== SEQUENCES ===")
print(sequences)

print("\nSequence lengths:")

print(
    sequences.select(
        pl.col("item_sequence").list.len().min().alias("min"),
        pl.col("item_sequence").list.len().median().alias("median"),
        pl.col("item_sequence").list.len().mean().alias("mean"),
        pl.col("item_sequence").list.len().max().alias("max"),
    )
)


# ============================================================
# SAVE
# ============================================================

interactions.write_parquet(DATA_DIR / "interactions_model.parquet")

catalog.write_parquet(DATA_DIR / "catalog_model.parquet")

item_mapping.write_parquet(DATA_DIR / "item_mapping.parquet")

user_mapping.write_parquet(DATA_DIR / "user_mapping.parquet")

sequences.write_parquet(DATA_DIR / "user_sequences.parquet")

print("\nSaved model-prepared data.")
