from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from scripts import train_sasrec
from src.data.sasrec_dataset import SASRecTrainDataset
from src.models.sasrec import SASRec, sasrec_loss


class SASRecMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = SASRec(
            num_items=20,
            max_len=5,
            d_model=8,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )
        self.items = torch.tensor(
            [
                [0, 0, 1, 2, 3],
                [1, 2, 3, 4, 5],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )

    def test_eval_is_finite_and_matches_training(self):
        self.model.train()
        with torch.no_grad():
            expected = self.model.encode(self.items)
        self.model.eval()
        fastpath = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(True)
            with torch.no_grad():
                actual = self.model.encode(self.items)
                scores = self.model.score_items(self.items, torch.arange(1, 21))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.isfinite(scores).all())
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)

    def test_future_items_and_padding_do_not_affect_real_context(self):
        self.model.eval()
        changed = self.items.clone()
        changed[0, -1] = 9
        with torch.no_grad():
            expected = self.model.encode(self.items)
            actual = self.model.encode(changed)
            torch.testing.assert_close(actual[0, 2:4], expected[0, 2:4])
            self.model.item_embedding.weight[0].fill_(100.0)
            actual = self.model.encode(self.items)
            torch.testing.assert_close(
                actual[self.items.ne(0)],
                expected[self.items.ne(0)],
            )

    def test_loss_and_gradients_are_finite_with_left_padding(self):
        positives = torch.tensor([4, 6, 2, 0])
        negatives = torch.full((4, 64), 20)
        pos_logits, neg_logits = self.model(self.items, positives, negatives)
        self.assertEqual(pos_logits.shape, (4,))
        self.assertEqual(neg_logits.shape, (4, 64))
        loss = sasrec_loss(pos_logits, neg_logits, positives)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_training_scores_match_next_item_inference_scores(self):
        self.model.eval()
        positives = torch.tensor([4, 6, 2, 1])
        negatives = torch.tensor([[8, 9], [9, 8], [10, 11], [11, 10]])
        with torch.no_grad():
            positive_scores, negative_scores = self.model(self.items, positives, negatives)
            catalog_scores = self.model.score_items(self.items, torch.arange(21))
        torch.testing.assert_close(
            positive_scores, catalog_scores.gather(1, positives[:, None]).squeeze(1)
        )
        torch.testing.assert_close(negative_scores, catalog_scores.gather(1, negatives))


class SASRecTrainingTests(unittest.TestCase):
    def test_all_prefixes_include_later_users_and_skip_short_histories(self):
        dataset = SASRecTrainDataset(
            [[8, 90, 91], [1, 2, 3, 4, 90, 91], [90, 91], [5, 6, 7, 92, 93]],
            np.arange(1, 30), max_len=2,
        )
        expected = [([0, 1], 2), ([1, 2], 3), ([2, 3], 4), ([0, 5], 6), ([5, 6], 7)]
        self.assertEqual(len(dataset), len(expected))
        first_negatives = dataset[0][2].clone()
        for epoch in (0, 1):
            dataset.set_epoch(epoch)
            for idx, (prefix, target) in enumerate(expected):
                actual_prefix, actual_target, negatives = dataset[idx]
                self.assertEqual(actual_prefix.tolist(), prefix)
                self.assertEqual(actual_target.item(), target)
                history = [1, 2, 3, 4] if idx < 3 else [5, 6, 7]
                self.assertFalse(np.isin(negatives.numpy(), history).any())
        self.assertFalse(torch.equal(first_negatives, dataset[0][2]))
        for idx in (-1, len(dataset)):
            with self.assertRaises(IndexError):
                dataset[idx]
        self.assertEqual(len(SASRecTrainDataset([[1, 2, 3]], [4])), 0)

    def test_dataset_trains_through_real_training_loop(self):
        torch.manual_seed(42)
        dataset = SASRecTrainDataset(
            [[1, 2, 3, 4, 18, 19], [5, 6, 7, 18, 19]],
            np.arange(1, 21), max_len=7,
        )
        # Includes a final batch of size one; B, L and N are different.
        loader = DataLoader(dataset, batch_size=4)
        model = SASRec(20, max_len=7, d_model=8, n_heads=2, n_layers=1, dropout=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.item_embedding.weight.detach().clone()
        loss = train_sasrec.train_one_epoch(
            model, loader, optimizer, torch.amp.GradScaler("cuda", enabled=False),
            torch.device("cpu"),
        )
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.item_embedding.weight))
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_loss_balances_positives_and_multiple_negatives(self):
        positives = torch.tensor([2., -1., 100.])
        negatives = torch.tensor([[0., 3.], [-2., 1.], [100., 100.]])
        labels = torch.tensor([1, 2, 0])
        expected = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                positives[:2], torch.ones(2)
            )
            + torch.nn.functional.binary_cross_entropy_with_logits(
                negatives[:2], torch.zeros(2, 2)
            )
        )
        torch.testing.assert_close(sasrec_loss(positives, negatives, labels), expected)

    def test_evaluation_masks_seen_items_with_noncontiguous_candidates(self):
        model = SimpleNamespace(
            num_items=10, eval=Mock(),
            score_items=lambda sequences, candidates: candidates.float().expand(
                sequences.size(0), -1
            ).clone(),
        )
        records = [
            {"history": [8, 8, 9, 0], "target": 6, "popularity_group": "tail"},
            {"history": [], "target": 10, "popularity_group": "head"},
            {"history": [10, 2], "target": 8, "popularity_group": "mid"},
            {"history": [8], "target": 9, "popularity_group": "unseen"},
        ]
        metrics = train_sasrec.evaluate(
            model, records, np.array([8, 2, 6, 4, 10]), max_len=3,
            device=torch.device("cpu"), k_values=(1, 2), batch_size=2,
        )
        self.assertEqual(metrics["all"]["recall@1"], 0.5)
        self.assertEqual(metrics["all"]["recall@2"], 0.75)
        self.assertEqual(metrics["warm"]["n"], 3)
        self.assertAlmostEqual(metrics["tail"]["ndcg@2"], 1 / np.log2(3))
        self.assertEqual(metrics["unseen"]["recall@2"], 0)

    def test_main_trains_saves_and_evaluates_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sequences = []
            rows = []
            for user in range(12):
                other = (user + 1) % 12
                items = list(range(user * 5 + 1, user * 5 + 6)) + [other * 5 + 1, other * 5 + 2]
                sequences.append(items)
                for position, (item, split) in enumerate(zip(
                    items, ["train"] * 5 + ["valid", "test"]
                )):
                    rows.append({
                        "user_idx": user, "position": position, "item_idx": item,
                        "split": split, "train_popularity_group": "tail",
                    })
            pl.DataFrame(rows).write_parquet(path / "interactions_model.parquet")
            pl.DataFrame({"user_idx": list(range(12)), "item_sequence": sequences}).write_parquet(
                path / "user_sequences.parquet"
            )
            config = {
                **train_sasrec.CONFIG, "epochs": 1, "batch_size": 4, "max_len": 7,
                "d_model": 8, "n_heads": 2, "n_layers": 1,
            }
            with (
                patch.object(train_sasrec, "DATA_DIR", path),
                patch.object(train_sasrec, "CHECKPOINT_DIR", path),
                patch.object(train_sasrec, "CONFIG", config),
                patch("torch.cuda.is_available", return_value=False),
                redirect_stdout(io.StringIO()),
            ):
                train_sasrec.main()
            checkpoint = torch.load(path / "sasrec_best.pt", weights_only=True)
            history = json.loads(Path(checkpoint["history_file"]).read_text())
            self.assertEqual(history["num_train_examples"], 48)
            self.assertEqual(len(history["epochs"]), 1)
            self.assertEqual(history["best_epoch"], 1)
            self.assertEqual(history["test_metrics"]["all"]["n"], 12)


if __name__ == "__main__":
    unittest.main()
