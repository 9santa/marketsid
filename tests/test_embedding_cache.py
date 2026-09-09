import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from src.data.embedding_cache import (
    ENCODER_CONFIG,
    EmbeddingCacheError,
    embedding_metadata,
    load_embedding_cache,
    save_embedding_cache,
)


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "embeddings.npy"
        self.catalog = pl.DataFrame({
            "item_idx": [1, 2], "parent_asin": ["a", "b"], "item_text": ["A", None],
        })
        self.metadata = embedding_metadata(self.catalog)
        self.embeddings = np.ones((2, 384), dtype=np.float32)
        save_embedding_cache(self.path, self.embeddings, self.metadata)

    def test_matching_cache_round_trips(self):
        np.testing.assert_array_equal(
            load_embedding_cache(self.path, self.metadata), self.embeddings
        )

    def test_catalog_and_encoder_changes_invalidate_cache(self):
        catalogs = [
            self.catalog.reverse(),
            self.catalog.with_columns(pl.Series("parent_asin", ["b", "a"])),
            self.catalog.with_columns(pl.lit("new text").alias("item_text")),
        ]
        expected_metadata = [embedding_metadata(c) for c in catalogs]
        for field, value in (
            ("model_name", "another-model"), ("revision", "another-revision"),
            ("max_seq_length", 256), ("normalize_embeddings", False),
            ("embedding_dim", 768),
        ):
            expected_metadata.append(embedding_metadata(
                self.catalog, {**ENCODER_CONFIG, field: value}
            ))
        for metadata in expected_metadata:
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(EmbeddingCacheError, "Regenerate"):
                    load_embedding_cache(self.path, metadata)

    def test_missing_or_invalid_manifest_is_not_trusted(self):
        metadata_path = self.path.with_suffix(".json")
        metadata_path.unlink()
        with self.assertRaises(EmbeddingCacheError):
            load_embedding_cache(self.path, self.metadata)
        for value in ("{", "[]", json.dumps(self.metadata)):
            metadata_path.write_text(value)
            with self.subTest(value=value), self.assertRaises(EmbeddingCacheError):
                load_embedding_cache(self.path, self.metadata)

    def test_replaced_embedding_file_fails_checksum_validation(self):
        np.save(self.path, self.embeddings * 2)
        with self.assertRaisesRegex(EmbeddingCacheError, "checksum"):
            load_embedding_cache(self.path, self.metadata)

    def test_wrong_shape_and_nonfinite_embeddings_are_rejected(self):
        for values in (np.ones((3, 384)), np.full((2, 384), np.nan)):
            save_embedding_cache(self.path, values, self.metadata)
            with self.assertRaises(EmbeddingCacheError):
                load_embedding_cache(self.path, self.metadata)


if __name__ == "__main__":
    unittest.main()
