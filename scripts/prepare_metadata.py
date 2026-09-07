from huggingface_hub import snapshot_download
import polars as pl
from pathlib import Path
import json
import re
import html


DATA_DIR = Path("data/processed")

CATEGORIES = [
    "Electronics",
    "Cell_Phones_and_Accessories",
    "Office_Products",
]

catalog = pl.read_parquet(DATA_DIR / "catalog.parquet")

wanted_items = set(catalog["parent_asin"].to_list())

print("Need metadata for:", len(wanted_items), "items")


COLUMNS = [
    "parent_asin",
    "title",
    "main_category",
    "store",
    "categories",
    "features",
    "description",
    "details",
    "price",
]


PARQUET_REVISIONS = {
    # 'Office_Products' metadata not currently present as parquet on main,
    # but available at this repository revision
    "Office_Products": "b7ac094bf775b1ae467e61a98d343207e826c4e6",
}


def load_metadata(category: str) -> pl.DataFrame:
    print(f"\nLoading metadata: {category}")

    category_items = (
        catalog.filter(pl.col("source_category") == category)
        .select("parent_asin")
        .unique()
    )

    revision = PARQUET_REVISIONS.get(category, "main")

    # Download metadata files into local HF cache
    repo_path = snapshot_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        repo_type="dataset",
        revision=revision,
        allow_patterns=[
            f"raw_meta_{category}/*.parquet",
        ],
    )

    # Path to downloaded Parquet files
    parquet_glob = Path(repo_path) / f"raw_meta_{category}" / "*.parquet"

    result = (
        pl.scan_parquet(parquet_glob)
        .select(COLUMNS)
        .join(
            category_items.lazy(),
            on="parent_asin",
            how="semi",
        )
        .collect(engine="streaming")
    )

    print(f"Found {len(result):,} / {len(category_items):,}")

    return result


metadata = pl.concat(
    [load_metadata(category) for category in CATEGORIES],
    how="vertical",
)

metadata = metadata.unique(
    subset=["parent_asin"],
    keep="first",
)

print("\nMetadata rows:", len(metadata))


# BUILD ITEM TEXT

WHITESPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"<[^>]+>")

IGNORE_DETAIL_KEYS = {
    "ASIN",
    "UPC",
    "EAN",
    "ISBN",
    "Best Sellers Rank",
    "Customer Reviews",
    "Date First Available",
}


def clean(value) -> str:
    if value is None:
        return ""

    value = str(value)
    value = html.unescape(value)
    value = HTML_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value)

    return value.strip()


def clean_list(values, limit=None) -> list[str]:
    if not values:
        return []

    result = [clean(v) for v in values if clean(v)]

    if limit is not None:
        result = result[:limit]

    return result


def format_details(raw_details) -> str:
    if not raw_details:
        return ""

    try:
        if isinstance(raw_details, str):
            details = json.loads(raw_details)
        else:
            details = raw_details
    except (json.JSONDecodeError, TypeError):
        return ""

    if not isinstance(details, dict):
        return ""

    parts = []

    for key, value in details.items():
        if key in IGNORE_DETAIL_KEYS:
            continue

        key = clean(key)
        value = clean(value)

        if not key or not value:
            continue

        # Guard against mega-long fields
        value = value[:200]

        parts.append(f"{key}: {value}")

        if len(parts) >= 12:
            break

    return "; ".join(parts)


def build_item_text(row: dict) -> str:
    parts = []

    title = clean(row.get("title"))
    if title:
        parts.append(f"Title: {title}")

    store = clean(row.get("store"))
    if store:
        parts.append(f"Brand or store: {store}")

    categories = clean_list(
        row.get("categories"),
        limit=8,
    )
    if categories:
        parts.append("Categories: " + " > ".join(categories))

    features = clean_list(
        row.get("features"),
        limit=10,
    )
    if features:
        parts.append("Features: " + "; ".join(features))

    descriptions = clean_list(
        row.get("description"),
        limit=3,
    )
    if descriptions:
        parts.append("Description: " + " ".join(descriptions))

    details = format_details(row.get("details"))
    if details:
        parts.append("Details: " + details)

    text = "\n".join(parts)

    return text[:4000]


metadata = metadata.with_columns(
    pl.struct(metadata.columns)
    .map_elements(
        build_item_text,
        return_dtype=pl.String,
    )
    .alias("item_text")
)


# CHECK METADATA COVERAGE

catalog_with_meta = catalog.join(
    metadata,
    on="parent_asin",
    how="left",
)

catalog_with_meta = catalog_with_meta.with_columns(
    pl.col("item_text").is_not_null().alias("has_metadata")
)

print("\n=== METADATA COVERAGE ===")

print(
    catalog_with_meta.group_by("source_category").agg(
        pl.len().alias("items"),
        pl.col("has_metadata").sum().alias("with_metadata"),
        pl.col("has_metadata").mean().alias("coverage"),
    )
)

print("\n=== TEXT LENGTH ===")

print(
    catalog_with_meta.filter(pl.col("has_metadata")).select(
        pl.col("item_text").str.len_chars().median().alias("median_chars"),
        pl.col("item_text").str.len_chars().quantile(0.90).alias("p90_chars"),
        pl.col("item_text").str.len_chars().max().alias("max_chars"),
    )
)


print(
    catalog_with_meta.filter(pl.col("has_metadata"))
    .select(
        "parent_asin",
        "source_category",
        "item_text",
    )
    .sample(
        n=5,
        seed=42,
    )
)


catalog_with_meta.write_parquet(DATA_DIR / "catalog_metadata.parquet")

print("\nSaved catalog metadata.")
