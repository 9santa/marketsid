import unittest

import torch

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
        positives = torch.where(self.items.ne(0), self.items + 1, 0)
        negatives = torch.where(self.items.ne(0), 20, 0)
        pos_logits, neg_logits = self.model(self.items, positives, negatives)
        loss = sasrec_loss(pos_logits, neg_logits, positives)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
