import albumentations as alb
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import common


class BaseDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode

        if mode == "test":
            self.labels = pd.read_pickle("../input/quickdraw/test_labels.pkl.gz")
            self.images = pd.read_pickle("../input/quickdraw/test_images.pkl.gz")
        else:  # train or valid
            self.labels = pd.read_pickle("../input/quickdraw/train_labels.pkl.gz")
            self.images = pd.read_pickle("../input/quickdraw/train_images.pkl.gz")

            if mode == "train":  # train-mode
                self.images, _, self.labels, _ = train_test_split(
                    self.images,
                    self.labels,
                    test_size=(common.args.valid_rate),
                    random_state=42,
                )
            else:  # valid-mode
                _, self.images, _, self.labels = train_test_split(
                    self.images,
                    self.labels,
                    test_size=(common.args.valid_rate),
                    random_state=42,
                )

        self.images = self.images[:, :, :, np.newaxis]

        assert len(self.images) == len(self.labels), "{} vs {}".format(
            len(self.images), len(self.labels)
        )

    def _augmentation(self, img):
        # -------
        def _albumentations(mode):
            aug_list = []

            aug_list.append(
                alb.Resize(
                    common.args.image_size,
                    common.args.image_size,
                    interpolation=cv2.INTER_CUBIC,
                    p=1.0,
                )
            )

            if mode == "train":  # use data augmentation only with train mode
                aug_list.append(alb.HorizontalFlip(p=0.5))

            aug_list.append(alb.Normalize(p=1.0, mean=(0.1093), std=(0.2921)))

            return alb.Compose(aug_list, p=1.0)

        img = _albumentations(self.mode)(image=img)["image"]

        return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        img = self._augmentation(img)
        img = img.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)

        return (torch.Tensor(img), torch.tensor(self.labels[idx]))
