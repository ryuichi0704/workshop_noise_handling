import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import common
from dataset.base_dataset import BaseDataset


class DistillationDataset(BaseDataset):
    def __init__(self, mode):
        super().__init__(mode)

        if mode == "test":
            self.labels = pd.read_pickle("../input/quickdraw/test_labels.pkl.gz")
            self.images = pd.read_pickle("../input/quickdraw/test_images.pkl.gz")
            self.oofs = None
        else:  # train or valid
            self.labels = pd.read_pickle("../input/quickdraw/train_labels.pkl.gz")
            self.images = pd.read_pickle("../input/quickdraw/train_images.pkl.gz")
            assert os.path.isfile(
                "../input/quickdraw/train_oofs.pkl.gz"
            ), "you need to put oof logits by yourself."
            self.oofs = pd.read_pickle("../input/quickdraw/train_oofs.pkl.gz")
            self.oofs = 1 / (1 + np.exp(-self.oofs))  # sigmoid

            if mode == "train":  # train-mode
                self.images, _, self.labels, _, self.oofs, _ = train_test_split(
                    self.images,
                    self.labels,
                    self.oofs,
                    test_size=(common.args.valid_rate),
                    random_state=42,
                )
            else:  # valid-mode
                _, self.images, _, self.labels, _, self.oofs = train_test_split(
                    self.images,
                    self.labels,
                    self.oofs,
                    test_size=(common.args.valid_rate),
                    random_state=42,
                )

        self.images = self.images[:, :, :, np.newaxis]

        assert len(self.images) == len(self.labels), "{} vs {}".format(
            len(self.images), len(self.labels)
        )

    def __getitem__(self, idx):
        img = self.images[idx]

        img = self._augmentation(img)
        img = img.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)
        label = np.eye(common.yml["train"]["n_classes"])[self.labels[idx]]

        if self.mode == "train":
            oof = F.softmax(torch.Tensor(self.oofs[idx]))

            return (torch.Tensor(img), torch.FloatTensor(label), torch.FloatTensor(oof))
        else:
            return (torch.Tensor(img), torch.FloatTensor(label))
