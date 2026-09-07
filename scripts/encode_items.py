from pathlib import Path

import numpy as np
import polars as pl
import torch

from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data/processed")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 256
MAX_SEQ_LENGTH = 512


catalog = pl.read_parquet(DATA_DIR / "catalog_model.parquet").sort("item_idx")


print("Items:", len(catalog))

assert catalog["item_idx"][0] == 1
assert catalog["item_idx"][-1] == len(catalog)


# ============================================================
# TEXTS
# ============================================================

texts = catalog["item_text"].fill_null("").to_list()

assert len(texts) == len(catalog)


# ============================================================
# MODEL
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


model = SentenceTransformer(
    model_name_or_path=MODEL_NAME,
    device=device,
)

model.max_seq_length = MAX_SEQ_LENGTH

print("Embedding dimension:", model.get_embedding_dimension())

print("Max sequence length:", model.max_seq_length)


# ============================================================
# ENCODE
# ============================================================

embedding_path = DATA_DIR / "item_text_embeddings.npy"

if embedding_path.exists():
    print("Embeddings already exist. Loading from disk...")
    embeddings = np.load(embedding_path)
else:
    print("Encoding items...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

print("\nEmbeddings:")
print("shape:", embeddings.shape)
print("dtype:", embeddings.dtype)


# ============================================================
# CHECK NORMALIZATION
# ============================================================

norms = np.linalg.norm(embeddings, axis=1)

print("Norm mean:", norms.mean())
print("Norm std:", norms.std())


# ============================================================
# SAVE
# ============================================================

np.save(
    DATA_DIR / "item_text_embeddings.npy",
    embeddings.astype(np.float32),
)

print("\nSaved embeddings.")


# ============================================================
# SANITY CHECK
# ============================================================
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(
    n_neighbors=6,
    metric="cosine",
)

nn.fit(embeddings)

rng = np.random.default_rng(42)

query_indices = rng.choice(
    len(catalog),
    size=10,
    replace=False,
)

for query_index in query_indices:
    distances, indices = nn.kneighbors(embeddings[query_index : query_index + 1])

    query = catalog.row(
        query_index,
        named=True,
    )

    print("\n")
    print("=" * 50)

    print(
        "QUERY:",
        query["parent_asin"],
    )

    print(query["item_text"][:300])

    print("\nNEIGHBORS")

    for dist, idx in zip(distances[0][1:], indices[0][1:]):
        neighbor = catalog.row(
            int(idx),
            named=True,
        )

        print(
            f"\ncos={1 - dist:.3f}",
            neighbor["parent_asin"],
        )

        print(neighbor["item_text"][:100])
