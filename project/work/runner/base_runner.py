import gzip
import os
import pickle
import time
import warnings
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import common
from dataset.base_dataset import BaseDataset
from model import Network

warnings.filterwarnings("ignore")


class BaseRunner(object):
    def __init__(self):
        self.model = Network()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=common.args.lr,
            momentum=common.args.momentum,
            weight_decay=common.args.weight_decay,
            nesterov=True,
        )

        if common.DEVICE.type == "cuda":
            devices = [int(n) for n in common.args.gpu.split(",")]
            self.model = torch.nn.DataParallel(self.model, device_ids=devices).to(
                common.DEVICE
            )  # for MultiGPU

        self.train_loss_history = []
        self.train_accuracy_history = []
        self.valid_loss_history = []
        self.valid_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []

    # -------------

    def _build_loader(self, mode):
        dataset = BaseDataset(mode=mode)

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

    def _calc_accuracy(self, preds, labels):
        score = 0
        total = 0

        for (pred, label) in zip(preds, labels):
            if pred == label:
                score += 1
            total += 1

        return score / total

    def _train_loop(self, loader):
        self.model.train()
        running_loss = 0

        train_preds, train_labels = [], []

        self.optimizer.zero_grad()

        for i, (images, labels) in enumerate(tqdm(loader, leave=False, desc="TRAIN")):
            images, labels = images.to(common.DEVICE), labels.to(common.DEVICE)

            outputs = self.model.forward(images, labels, mode="train")

            train_loss = (
                self.criterion(outputs, labels) / common.args.accumulation_steps
            )

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

    def _eval_loop(self, loader, mode):
        self.model.eval()
        running_loss = 0

        eval_preds, eval_labels, eval_logits = [], [], []

        with torch.no_grad():
            for (images, labels) in tqdm(loader, leave=False, desc=mode.upper()):
                images, labels = images.to(common.DEVICE), labels.to(common.DEVICE)
                outputs = self.model.forward(images, labels, mode)
                eval_loss = self.criterion(outputs, labels)
                running_loss += eval_loss.item()

                _, predicted = torch.max(outputs.data, 1)

                eval_preds.append(predicted.cpu())
                eval_labels.append(labels.cpu())
                eval_logits.append(outputs.data.cpu())

            eval_loss = running_loss / len(loader)

            eval_preds = torch.cat(eval_preds)
            eval_labels = torch.cat(eval_labels)
            eval_logits = torch.cat(eval_logits)
            eval_accuracy = self._calc_accuracy(eval_preds, eval_labels)

        return eval_loss, eval_accuracy, eval_logits

    def _save_logits(self, valid_logits, test_logits):
        with gzip.open(
            os.path.join(common.OUTPUT_DIR, "valid_logits.pkl.gz"), mode="wb"
        ) as f:
            pickle.dump(valid_logits.numpy(), f)

        with gzip.open(
            os.path.join(common.OUTPUT_DIR, "test_logits.pkl.gz"), mode="wb"
        ) as f:
            pickle.dump(test_logits.numpy(), f)

    # -------

    def train_model(self):
        train_loader = self._build_loader(mode="train")
        valid_loader = self._build_loader(mode="valid")
        test_loader = self._build_loader(mode="test")

        scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(common.args.epoch * 0.8), int(common.args.epoch * 0.9)],
            gamma=0.1,
        )

        best_accuracy = -1

        for current_epoch in range(1, common.args.epoch + 1, 1):
            start_time = time.time()
            train_loss, train_accuracy = self._train_loop(train_loader)
            valid_loss, valid_accuracy, valid_logits = self._eval_loop(
                valid_loader, mode="valid"
            )
            test_loss, test_accuracy, test_logits = self._eval_loop(
                test_loader, mode="test"
            )

            if best_accuracy < valid_accuracy:
                self._save_logits(valid_logits, test_logits)

            print(
                "epoch: {} / ".format(current_epoch)
                + "train loss: {:.5f} / ".format(train_loss)
                + "train acc: {:.5f} / ".format(train_accuracy)
                + "valid loss: {:.5f} / ".format(valid_loss)
                + "valid acc: {:.5f} / ".format(valid_accuracy)
                + "test loss: {:.5f} / ".format(test_loss)
                + "test acc: {:.5f} / ".format(test_accuracy)
                + "lr: {:.5f} / ".format(self.optimizer.param_groups[0]["lr"])
                + "time: {}sec".format(int(time.time() - start_time))
            )

            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_accuracy)
            self.valid_loss_history.append(valid_loss)
            self.valid_accuracy_history.append(valid_accuracy)
            self.test_loss_history.append(test_loss)
            self.test_accuracy_history.append(test_accuracy)

            scheduler.step()
