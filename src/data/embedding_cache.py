"""Bind cached text embeddings to their ordered catalog and encoder settings."""

import hashlib
import json
from pathlib import Path

import numpy as np


ENCODER_CONFIG = {
    "model_name": "BAAI/bge-small-en-v1.5",
    # Pin the checkpoint already used in the local Hugging Face cache.
    "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "max_seq_length": 512,
    "normalize_embeddings": True,
    "embedding_dim": 384,
}


class EmbeddingCacheError(ValueError):
    pass


def embedding_metadata(catalog, encoder_config=None):
    digest = hashlib.sha256()
    for item_idx, asin, text in catalog.select(
        "item_idx", "parent_asin", "item_text"
    ).iter_rows():
        digest.update(json.dumps([item_idx, asin, text or ""]).encode("utf-8"))
        digest.update(b"\n")
    return {
        "version": 1,
        "catalog_hash": digest.hexdigest(),
        "num_items": len(catalog),
        "encoder": dict(ENCODER_CONFIG if encoder_config is None else encoder_config),
    }


def _file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_embedding_cache(path, expected_metadata):
    path = Path(path)
    try:
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("invalid metadata")
        file_hash = metadata.pop("embeddings_sha256", None)
        if metadata != expected_metadata:
            raise ValueError("catalog or encoder settings changed")
        if file_hash != _file_hash(path):
            raise ValueError("embedding file checksum mismatch")
        embeddings = np.load(path, allow_pickle=False)
        expected_shape = (
            expected_metadata["num_items"],
            expected_metadata["encoder"]["embedding_dim"],
        )
        if embeddings.shape != expected_shape or embeddings.dtype != np.float32:
            raise ValueError("embedding shape or dtype mismatch")
        if not np.isfinite(embeddings).all():
            raise ValueError("non-finite embeddings")
        return embeddings
    except (OSError, ValueError) as error:
        raise EmbeddingCacheError(
            f"Cannot validate {path}: {error}. "
            "Regenerate with: python -m scripts.encode_items"
        ) from error


def save_embedding_cache(path, embeddings, metadata):
    path = Path(path)
    temporary_path = path.with_suffix(".npy.tmp")
    metadata_path = path.with_suffix(".json")
    temporary_metadata_path = metadata_path.with_suffix(".json.tmp")
    with temporary_path.open("wb") as stream:
        np.save(stream, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
    temporary_metadata_path.write_text(
        json.dumps(
            {**metadata, "embeddings_sha256": _file_hash(temporary_path)}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    # Publish metadata last. An interrupted update fails checksum validation.
    temporary_path.replace(path)
    temporary_metadata_path.replace(metadata_path)
