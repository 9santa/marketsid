import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.sasrec_dataset import _pad_left


class GenRecTrainDataset(Dataset):
    def __init__(
        self,
        sequences,
        max_len=50,
    ):
        self.sequences = [
            np.asarray(
                seq[:-2],
                dtype=np.int64,
            )
            for seq in sequences
        ]

        self.max_len = max_len

        self.examples = []

        for user_idx, sequence in enumerate(self.sequences):
            for target_position in range(1, len(sequence)):
                self.examples.append(
                    (
                        user_idx,
                        target_position,
                    )
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        user_idx, target_position = self.examples[idx]

        sequence = self.sequences[user_idx]

        history = sequence[:target_position]

        target = sequence[target_position]

        history = _pad_left(history, self.max_len)

        return (
            torch.from_numpy(history),
            torch.tensor(target, dtype=torch.long),
        )
