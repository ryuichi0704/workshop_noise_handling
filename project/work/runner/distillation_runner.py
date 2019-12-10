import warnings
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import common
from dataset.distillation_dataset import DistillationDataset
from runner.base_runner import BaseRunner

warnings.filterwarnings("ignore")


class DistillationRunner(BaseRunner):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def _build_loader(self, mode):
        dataset = DistillationDataset(mode=mode)

        if mode == "train":
            sampler = torch.utils.data.sampler.RandomSampler(data_source=dataset.images)

        else:  # valid, test
            sampler = torch.utils.data.sampler.SequentialSampler(
                data_source=dataset.images
            )

        loader = DataLoader(
            dataset,
            batch_size=common.args.batch_size,
            sampler=sampler,
            num_workers=cpu_count(),
            worker_init_fn=lambda x: np.random.seed(),
            drop_last=True if mode == "train" else False,
            pin_memory=True,
        )

        return loader

    def _train_loop(self, loader):
        self.model.train()
        running_loss = 0

        train_preds, train_labels = [], []

        self.optimizer.zero_grad()

        for i, (images, labels, oofs) in enumerate(
            tqdm(loader, leave=False, desc="TRAIN")
        ):
            images, labels, oofs = (
                images.to(common.DEVICE),
                labels.to(common.DEVICE),
                oofs.to(common.DEVICE),
            )

            outputs = self.model.forward(images, labels, mode="train")

            # ----------------------------------
            # calculate hard loss and soft loss
            train_loss_hard = (
                self.criterion(outputs, labels).sum() / labels.size(0)
            ) / common.args.accumulation_steps
            train_loss_soft = (
                self.criterion(outputs, oofs).sum() / labels.size(0)
            ) / common.args.accumulation_steps

            # merge hard loss and soft loss
            train_loss = (
                train_loss_hard + (common.args.soft_loss_weight * train_loss_soft)
            ) / (1 + common.args.soft_loss_weight)
            # ----------------------------------

            train_loss.backward()

            if (i + 1) % common.args.accumulation_steps == 0:
                self.optimizer.step()
                self.model.zero_grad()

            running_loss += train_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)

            train_preds.append(predicted.cpu())
            train_labels.append(labels.cpu())

        train_loss = running_loss / len(loader)

        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_accuracy = self._calc_accuracy(train_preds, train_labels)

        return train_loss, train_accuracy

    def _eval_loop(self, loader, mode):
        self.model.eval()
        running_loss = 0

        eval_preds, eval_labels, eval_logits = [], [], []

        with torch.no_grad():
            for (images, labels) in tqdm(loader, leave=False, desc=mode.upper()):
                images, labels = images.to(common.DEVICE), labels.to(common.DEVICE)
                outputs = self.model.forward(images, labels, mode)
                eval_loss = self.criterion(outputs, labels).sum() / labels.size(0)
                running_loss += eval_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels.data, 1)

                eval_preds.append(predicted.cpu())
                eval_labels.append(labels.cpu())
                eval_logits.append(outputs.data.cpu())

            eval_loss = running_loss / len(loader)

            eval_preds = torch.cat(eval_preds)
            eval_labels = torch.cat(eval_labels)
            eval_logits = torch.cat(eval_logits)
            eval_accuracy = self._calc_accuracy(eval_preds, eval_labels)

        return eval_loss, eval_accuracy, eval_logits
