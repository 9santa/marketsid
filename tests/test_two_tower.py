from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from scripts import train_two_tower
from src.data.two_tower_dataset import NextItemDataset
from src.models.content_two_tower import ContentTwoTower
from src.data.embedding_cache import (
    EmbeddingCacheError, embedding_metadata, save_embedding_cache,
)


class TwoTowerTests(unittest.TestCase):
    def test_all_training_prefixes_are_covered_without_held_out_items(self):
        dataset = NextItemDataset(
            [[8, 90, 91], [1, 2, 3, 4, 90, 91], [90, 91], [5, 6, 7, 92, 93]],
            np.arange(1, 30), max_len=2,
        )
        expected = [([0, 1], 2), ([1, 2], 3), ([2, 3], 4), ([0, 5], 6), ([5, 6], 7)]
        self.assertEqual(len(dataset), len(expected))
        for epoch in (0, 1):
            dataset.set_epoch(epoch)
            for idx, (prefix, target) in enumerate(expected):
                actual_prefix, actual_target, negatives = dataset[idx]
                self.assertEqual(actual_prefix.tolist(), prefix)
                self.assertEqual(actual_target.item(), target)
                self.assertEqual(negatives.shape, (64,))
                history = [1, 2, 3, 4] if idx < 3 else [5, 6, 7]
                self.assertFalse(np.isin(negatives.numpy(), history).any())
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_epoch_changes_negatives_but_not_prefixes_or_targets(self):
        dataset = NextItemDataset([[1, 2, 3, 4, 5]], np.arange(1, 100))
        first = dataset[0]
        for original, repeated in zip(first, dataset[0]):
            torch.testing.assert_close(original, repeated)
        dataset.set_epoch(1)
        second = dataset[0]
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        self.assertFalse(torch.equal(first[2], second[2]))

    def test_load_preserves_split_order_when_timestamps_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pl.DataFrame({
                "item_idx": [1, 2, 3, 4, 5],
                "parent_asin": ["a", "b", "c", "d", "e"],
                "item_text": ["A", "B", "C", "D", "E"],
            }).write_parquet(path / "catalog_model.parquet")
            interactions = pl.DataFrame({
                "user_idx": [0] * 5,
                "position": [0, 1, 2, 3, 4],
                "timestamp": [1, 2, 3, 3, 4],
                "item_idx": [4, 3, 2, 1, 5],
                "split": ["train", "train", "train", "valid", "test"],
            })
            # Store in the problematic timestamp/item-ID order.
            interactions.sort(["timestamp", "item_idx"]).write_parquet(
                path / "interactions_model.parquet"
            )
            pl.DataFrame({
                "user_idx": [0],
                "item_sequence": [[4, 3, 2, 1, 5]],
                "split_sequence": [["train", "train", "train", "valid", "test"]],
            }).write_parquet(path / "user_sequences.parquet")
            catalog = pl.read_parquet(path / "catalog_model.parquet")
            save_embedding_cache(
                path / "item_text_embeddings.npy", np.ones((5, 384), dtype=np.float32),
                embedding_metadata(catalog),
            )
            with patch.object(train_two_tower, "DATA_DIR", path), redirect_stdout(io.StringIO()):
                features, sequences, counts, warm, all_ids, mapping_hash = train_two_tower.load_data()
            self.assertEqual(sequences[0].tolist(), [4, 3, 2, 1, 5])
            self.assertEqual(warm.tolist(), [2, 3, 4])
            self.assertEqual(features.shape, (6, 384))
            self.assertEqual(len(mapping_hash), 64)
            record = train_two_tower.make_eval_records(sequences, counts, "valid")[0]
            self.assertEqual(record["history"].tolist(), [4, 3, 2])
            self.assertEqual(record["target"], 1)

            catalog.with_columns(pl.lit("changed").alias("item_text")).write_parquet(
                path / "catalog_model.parquet"
            )
            with patch.object(train_two_tower, "DATA_DIR", path):
                with self.assertRaises(EmbeddingCacheError):
                    train_two_tower.load_data()
            catalog.write_parquet(path / "catalog_model.parquet")

            pl.DataFrame({
                "user_idx": [0],
                "item_sequence": [[4, 3, 1, 2, 5]],
                "split_sequence": [["train", "train", "valid", "train", "test"]],
            }).write_parquet(path / "user_sequences.parquet")
            with patch.object(train_two_tower, "DATA_DIR", path):
                with self.assertRaisesRegex(AssertionError, "Invalid split order for user_idx=0"):
                    train_two_tower.load_data()

    def test_training_updates_model_and_evaluation_is_finite(self):
        torch.manual_seed(42)
        model = ContentTwoTower(
            torch.randn(9, 12), max_len=5, d_model=8,
            n_heads=2, n_layers=2, dropout=0,
        )
        dataset = NextItemDataset([[1, 2, 3, 4, 5]], np.arange(1, 9), max_len=5)
        loader = DataLoader(dataset, batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.item_tower[0].weight.detach().clone()
        loss = train_two_tower.train_one_epoch(
            model, loader, optimizer, torch.amp.GradScaler("cuda", enabled=False),
            torch.device("cpu"),
        )
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.item_tower[0].weight))
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        metrics = train_two_tower.evaluate(
            model, [{"history": [1, 2, 3], "target": 4, "group": "tail"}],
            np.arange(1, 9), max_len=5, device=torch.device("cpu"), k_values=(1, 2),
        )
        self.assertEqual(metrics["warm"]["n"], 1)
        self.assertTrue(np.isfinite(metrics["warm"]["ndcg@2"]))


if __name__ == "__main__":
    unittest.main()
