from operator import le
from datasets import load_dataset
import polars as pl
from pathlib import Path


# Категории специально выбраны связанные,
# чтобы можно было оценить behaviour-aware Semantic IDs:
# substitutes: Sony headphones <-> Bose headphones
# complements: Phone -> charger, case, earbuds; Laptop -> mouse, monitor, mousepad

CATEGORIES = [
    "Electronics",
    "Cell_Phones_and_Accessories",
    "Office_Products",
]

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_category(category: str) -> pl.DataFrame:
    print(f"Loading {category}...")

    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"5core_rating_only_{category}",
        split="full",
        trust_remote_code=True,
    )

    df = pl.from_arrow(ds.data.table)

    df = df.select(
        "user_id",
        "parent_asin",
        "rating",
        "timestamp",
    ).with_columns(
        pl.lit(category).alias("source_category"),
        pl.col("rating").cast(pl.Float32),
        pl.col("timestamp").cast(pl.Int64),
    )

    print(
        category,
        "rows =",
        len(df),
        "users =",
        df["user_id"].n_unique(),
        "items =",
        df["parent_asin"].n_unique(),
    )

    return df


frames = [load_category(c) for c in CATEGORIES]

interactions = pl.concat(frames)

interactions = interactions.unique(
    subset=["user_id", "parent_asin", "timestamp"],
    keep="first",
)

print("\nCombined:")
print("rows:", len(interactions))
print("users:", interactions["user_id"].n_unique())
print("items:", interactions["parent_asin"].n_unique())


print("\n=== CATEGORY STATS ===")

stats = interactions.group_by("source_category").agg(
    pl.len().alias("n_interactions"),
    pl.col("user_id").n_unique().alias("n_unique_users"),
    pl.col("parent_asin").n_unique().alias("n_unique_items"),
)

print(stats)

user_categories = interactions.group_by("user_id").agg(
    pl.col("source_category").n_unique().alias("n_categories")
)

print("\n=== USER CATEGORY COVERAGE ===")

print(
    user_categories.group_by("n_categories")
    .agg(pl.len().alias("n_users"))
    .sort("n_categories")
)


# =========================== PAIRWISE CATEGORY OVERLAPS ===========================

users_by_cat = {
    category: set(
        interactions.filter(pl.col("source_category") == category)["user_id"].to_list()
    )
    for category in CATEGORIES
}

print("\n=== USER OVERLAP ===")

for i, c1 in enumerate(CATEGORIES):
    for c2 in CATEGORIES[i + 1 :]:
        overlap = users_by_cat[c1] & users_by_cat[c2]

        print(f"{c1} <-> {c2}: {len(overlap):,} shared users")


# =========================== ELIGIBLE USERS FILTERING ===========================

user_stats = interactions.group_by("user_id").agg(
    pl.len().alias("n_interactions"),
    pl.col("parent_asin").n_unique().alias("n_items"),
    pl.col("source_category").n_unique().alias("n_categories"),
)

print("\n=== USER INTERACTIONS STATS ===")

print(
    user_stats.select(
        pl.col("n_interactions").min().alias("min"),
        pl.col("n_interactions").median().alias("median"),
        pl.col("n_interactions").mean().alias("mean"),
        pl.col("n_interactions").quantile(0.90).alias("p90"),
        pl.col("n_interactions").quantile(0.99).alias("p99"),
        pl.col("n_interactions").max().alias("max"),
    )
)

MIN_USER_INTERACTIONS = 5
MAX_USER_INTERACTIONS = 200

eligible_users = user_stats.filter(
    (pl.col("n_interactions") >= MIN_USER_INTERACTIONS)
    & (pl.col("n_interactions") <= MAX_USER_INTERACTIONS)
)

print("\nEligible users:", len(eligible_users))


# =========================== USER SAMPLING (aiming for 750k-1.5M interactions) ===========================

TARGET_USERS = 100_000
SEED = 42

sampled_users = eligible_users.sample(
    n=TARGET_USERS,
    seed=SEED,
    shuffle=True,
)["user_id"]

# Wrap the sampled list/Series in lit() and implode() to safely pass it to .is_in()
sampled = interactions.filter(pl.col("user_id").is_in(pl.lit(sampled_users).implode()))

print("\n=== SAMPLED DATASET ===")
print("interactions:", len(sampled))
print("users:", sampled["user_id"].n_unique())
print("items:", sampled["parent_asin"].n_unique())
# Got:
# interactions: 1057432
# users: 100000
# items: 296108

print(
    sampled.group_by("source_category").agg(
        pl.len().alias("interactions"),
        pl.col("user_id").n_unique().alias("users"),
        pl.col("parent_asin").n_unique().alias("items"),
    )
)


# =========================== POPULARITY DISTRIBUTION ===========================

item_stats = sampled.group_by("parent_asin").agg(
    pl.len().alias("n_interactions"),
    pl.col("user_id").n_unique().alias("n_users"),
    pl.col("source_category").first().alias("category"),
)

print("\n=== ITEM STATS ===")

print(
    item_stats.select(
        pl.col("n_interactions").min().alias("min"),
        pl.col("n_interactions").median().alias("median"),
        pl.col("n_interactions").mean().alias("mean"),
        pl.col("n_interactions").quantile(0.90).alias("p90"),
        pl.col("n_interactions").quantile(0.99).alias("p99"),
        pl.col("n_interactions").max().alias("max"),
    )
)

item_stats = item_stats.with_columns(
    pl.when(pl.col("n_interactions") <= 5)
    .then(pl.lit("tail"))
    .when(pl.col("n_interactions") <= 20)
    .then(pl.lit("mid"))
    .otherwise(pl.lit("head"))
    .alias("popularity_group")
)

print(
    item_stats.group_by("popularity_group").agg(
        pl.len().alias("items"),
        pl.col("n_interactions").sum().alias("interactions"),
    )
)


# =========================== FINAL K-CORE FILTERING ===========================

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 2


def iterative_k_core(
    df: pl.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pl.DataFrame:
    iteration = 0

    while True:
        iteration += 1
        old_rows = len(df)

        valid_users = (
            df.group_by("user_id")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_user_interactions)
            .select("user_id")
        )

        df = df.join(
            valid_users,
            on="user_id",
            how="semi",
        )

        valid_items = (
            df.group_by("parent_asin")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_item_interactions)
            .select("parent_asin")
        )

        df = df.join(
            valid_items,
            on="parent_asin",
            how="semi",
        )

        print(f"K-core iteration {iteration}: {old_rows:,} -> {len(df):,} interactions")

        if len(df) == old_rows:
            break

    return df


sampled = iterative_k_core(
    sampled,
    min_user_interactions=MIN_USER_INTERACTIONS,
    min_item_interactions=MIN_ITEM_INTERACTIONS,
)

print("\n=== FINAL DATASET ===")
print("interactions:", len(sampled))
print("users:", sampled["user_id"].n_unique())
print("items:", sampled["parent_asin"].n_unique())

print(sampled.head())


n_user_item_pairs = sampled.select(["user_id", "parent_asin"]).n_unique()

n_repeated = len(sampled) - n_user_item_pairs

print("\n=== REPEATED USER-ITEM INTERACTIONS ===")
print("unique pairs:", n_user_item_pairs)
print("repeated interactions:", n_repeated)
print("fraction:", n_repeated / len(sampled))


# =========================== TEMPORAL SPLIT ===========================
# User-wise Chronological leave-two-out

# i1, i2, i3, i4, i5, i6, i7
#
# train: i1 i2 i3 i4 i5
# valid:                i6
# test:                    i7


sampled = sampled.sort(["user_id", "timestamp"])

sampled = sampled.with_columns(
    pl.int_range(pl.len()).over("user_id").alias("position"),
    pl.len().over("user_id").alias("sequence_length"),  # no. of interactions per user
)

sampled = sampled.with_columns(
    pl.when(pl.col("position") == pl.col("sequence_length") - 1)
    .then(pl.lit("test"))
    .when(pl.col("position") == pl.col("sequence_length") - 2)
    .then(pl.lit("valid"))
    .otherwise(pl.lit("train"))
    .alias("split")
)


print(
    sampled.group_by("split").agg(
        pl.len().alias("interactions"),
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("parent_asin").n_unique().alias("unique_items"),
    )
)


# sampled = sampled.with_columns(
#     pl.col("source_category").shift(1).over("user_id").alias("prev_category")
# )
#
# sampled = sampled.with_columns(
#     (pl.col("source_category") != pl.col("prev_category"))
#     .fill_null(False)
#     .alias("is_cross_category")
# )
#
# targets = sampled.filter(pl.col("split").is_in(["valid", "test"]))
#
# print(targets.group_by(["split", "is_cross_category"]).agg(pl.len().alias("n")))

# POPULARITY SLICE ONLY ON TRAIN SPLIT

train = sampled.filter(pl.col("split") == "train")

train_item_stats = train.group_by("parent_asin").agg(
    pl.len().alias("train_interactions"),
    pl.col("user_id").n_unique().alias("train_users"),
)

sampled = sampled.join(
    train_item_stats,
    on="parent_asin",
    how="left",
).with_columns(
    pl.col("train_interactions").fill_null(0).cast(pl.UInt32),
    pl.col("train_users").fill_null(0).cast(pl.UInt32),
)

sampled = sampled.with_columns(
    pl.when(pl.col("train_interactions") == 0)
    .then(pl.lit("unseen"))
    .when(pl.col("train_interactions") <= 5)
    .then(pl.lit("tail"))
    .when(pl.col("train_interactions") <= 20)
    .then(pl.lit("mid"))
    .otherwise(pl.lit("head"))
    .alias("train_popularity_group")
)


targets = sampled.filter(pl.col("split").is_in(["valid", "test"]))

print("\n=== TARGET POPULARITY ===")

print(
    targets.group_by(["split", "train_popularity_group"])
    .agg(pl.len().alias("targets"))
    .sort(["split", "train_popularity_group"])
)

print("\n=== WARM / UNSEEN TARGETS ===")

print(
    targets.group_by("split").agg(
        pl.len().alias("all_targets"),
        (pl.col("train_interactions") > 0).sum().alias("warm_targets"),
        (pl.col("train_interactions") == 0).sum().alias("unseen_targets"),
        ((pl.col("train_interactions") == 0).mean()).alias("unseen_fraction"),
    )
)

# =========================== BUILD ITEM CATALOG ===========================

catalog = sampled.group_by("parent_asin").agg(
    pl.col("source_category").first().alias("source_category"),
    pl.col("source_category").n_unique().alias("n_source_categories"),
    pl.col("train_interactions").first(),
    pl.col("train_users").first(),
)

print("\n=== CATALOG ===")
print(catalog)

print(
    "Items appearing in multiple source categories:",
    catalog.filter(pl.col("n_source_categories") > 1).height,
)


# =========================== SAVE ===========================

sampled.write_parquet(OUTPUT_DIR / "interactions.parquet")

train_item_stats.write_parquet(OUTPUT_DIR / "train_item_stats.parquet")

catalog.write_parquet(OUTPUT_DIR / "catalog.parquet")

print("\nSaved final interaction dataset.")
