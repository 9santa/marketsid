import numpy as np
import torch
from torch.utils.data import Dataset


class SASRecTrainDataset(Dataset):
    """One example per training prefix, with one next item and 64 negatives.

    The final two interactions are held out. Inputs are left-padded with 0
    and retain at most max_len items. Negatives are resampled each epoch.
    """

    def __init__(
        self,
        sequences,
        warm_item_ids,
        max_len=50,
        seed=42,
    ):
        self.histories = [np.asarray(seq[:-2], dtype=np.int64) for seq in sequences]

        # Map example indices to users without storing copies of every prefix.
        self.prefix_offsets = np.cumsum(
            [0] + [max(len(history) - 1, 0) for history in self.histories],
            dtype=np.int64,
        )

        self.warm_item_ids = np.asarray(warm_item_ids, dtype=np.int64)

        self.max_len = max_len
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return int(self.prefix_offsets[-1])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        user_idx = int(np.searchsorted(self.prefix_offsets, idx, side="right") - 1)
        target_position = int(idx - self.prefix_offsets[user_idx] + 1)

        history = self.histories[user_idx]

        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + idx)

        prefix = history[:target_position]
        target = int(history[target_position])

        negatives = rng.choice(
            self.warm_item_ids,
            size=64,
            replace=True,
        )

        # Remove bad-negatives - items that the user has actually interacted with in the training data
        bad = np.isin(negatives, history)
        while bad.any():
            negatives[bad] = rng.choice(
                self.warm_item_ids,
                size=int(bad.sum()),
                replace=True,
            )
            bad = np.isin(negatives, history)

        return (
            torch.from_numpy(_pad_left(prefix, self.max_len)),
            torch.tensor(target, dtype=torch.long),
            torch.from_numpy(negatives.astype(np.int64)),
        )


def _pad_left(values, max_len, dtype=np.int64):
    result = np.zeros(max_len, dtype=dtype)
    values = np.asarray(values, dtype=dtype)[-max_len:]

    if len(values) > 0:
        result[-len(values) :] = values

    return result
