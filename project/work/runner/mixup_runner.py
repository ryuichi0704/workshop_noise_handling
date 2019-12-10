import warnings

import torch
from tqdm import tqdm

import common
from runner.base_runner import BaseRunner

warnings.filterwarnings("ignore")


class MixupRunner(BaseRunner):
    def __init__(self):
        super().__init__()

    def _train_loop(self, loader):
        def mixup_criterion(pred, labels_a, labels_b, lam):
            return lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
                pred, labels_b
            )

        self.model.train()
        running_loss = 0

        train_preds, train_labels = [], []

        self.optimizer.zero_grad()

        for i, (images, labels) in enumerate(tqdm(loader, leave=False, desc="TRAIN")):
            images, labels = images.to(common.DEVICE), labels.to(common.DEVICE)

            # ---Mixup---
            outputs, labels_a, labels_b, lam = self.model.forward(
                images, labels, mode="train"
            )
            train_loss = mixup_criterion(outputs, labels_a, labels_b, lam)
            # -----------

            train_loss.backward()

            if (i + 1) % common.args.accumulation_steps == 0:
                self.optimizer.step()
                self.model.zero_grad()

            running_loss += train_loss.item()

            _, predicted = torch.max(outputs.data, 1)

            train_preds.append(predicted.cpu())
            train_labels.append(labels.cpu())

        train_loss = running_loss / len(loader)

        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_accuracy = self._calc_accuracy(train_preds, train_labels)

        return train_loss, train_accuracy
