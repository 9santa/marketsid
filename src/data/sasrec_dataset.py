import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SASRecTrainDataset(Dataset):
    """
    One training example = one user's train sequence.

    Example:
        history = [A, B, C, D, E]

        input    = [A, B, C, D]
        positive = [B, C, D, E]
        negative = [X, Y, Z, W]

    All sequences are left-padded with 0.

    During training, the model will compute scores for all positive and negative items (per position)
    and apply a pairwise loss to push the positive score higher than negatives.
    The random prefix ensures the model learns to predict the next item given any prefix of the user’s behaviour.
    """

    def __init__(
        self,
        sequences,
        warm_item_ids,
        max_len=50,
        seed=42,
    ):
        self.histories = [np.asarray(seq[:-2], dtype=np.int64) for seq in sequences]

        self.warm_item_ids = np.asarray(warm_item_ids, dtype=np.int64)

        self.max_len = max_len
        self.seed = seed
        self.epoch = 0

        self.histories = [h for h in self.histories if len(h) >= 2]

    def __len__(self):
        return len(self.histories)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + idx)

        history = self.histories[idx]

        window = history[-(self.max_len + 1) :]

        input_items = window[:-1]  # all but last -> the context
        positive_items = window[
            1:
        ]  # all but first -> the next items to predict for each subsequence

        # Negatives - only train catalog, one per positive
        negative_items = rng.choice(
            self.warm_item_ids,
            size=len(positive_items),
            replace=True,
        )

        # Remove bad-negatives - items that the user has actually interacted with in the training data
        bad = np.isin(negative_items, history)
        while bad.any():
            negative_items[bad] = rng.choice(
                self.warm_item_ids,
                size=bad.sum(),
                replace=True,
            )
            bad = np.isin(negative_items, history)

        return (
            torch.from_numpy(_pad_left(input_items, self.max_len)),
            torch.from_numpy(_pad_left(positive_items, self.max_len)),
            torch.from_numpy(_pad_left(negative_items, self.max_len)),
        )


def _pad_left(values, max_len, dtype=np.int64):
    result = np.zeros(max_len, dtype=dtype)
    values = np.asarray(values, dtype=dtype)[-max_len:]

    if len(values) > 0:
        result[-len(values) :] = values

    return result
