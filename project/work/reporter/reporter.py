import glob
import os

import gspread
import matplotlib.pyplot as plt
import numpy as np
import requests
from oauth2client.service_account import ServiceAccountCredentials

import common


class Reporter(object):
    def __init__(self, runner):
        self.train_loss_history = runner.train_loss_history
        self.train_accuracy_history = runner.train_accuracy_history
        self.valid_loss_history = runner.valid_loss_history
        self.valid_accuracy_history = runner.valid_accuracy_history
        self.test_loss_history = runner.test_loss_history
        self.test_accuracy_history = runner.test_accuracy_history
        self.best_epoch = np.where(
            self.valid_accuracy_history == np.nanmax(self.valid_accuracy_history)
        )[0][0]

        self.best_train_accuracy = self.train_accuracy_history[self.best_epoch]
        self.best_valid_accuracy = self.valid_accuracy_history[self.best_epoch]
        self.best_test_accuracy = self.test_accuracy_history[self.best_epoch]

    def _slack_notify_image(self, filename):
        if (
            len(common.yml["slack"]["token"]) == 0
            or len(common.yml["slack"]["channels"]) == 0
        ):
            print(
                "for sending slack notification, you need to set token in settings.yaml."
            )
        else:
            image = {"file": open(filename, "rb")}
            param = {
                "token": common.yml["slack"]["token"],
                "channels": common.yml["slack"]["channels"],
                "filename": filename,
                "title": filename,
            }
            requests.post(
                url="https://slack.com/api/files.upload", params=param, files=image
            )

    def plot_loss_curve(self):
        plt.figure(figsize=(20, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(
            np.arange(common.args.epoch) + 1,
            self.train_loss_history,
            label="train loss",
            color="red",
            linestyle="dotted",
            linewidth=1.0,
        )
        plt.plot(
            np.arange(common.args.epoch) + 1,
            self.valid_loss_history,
            label="valid loss",
            color="red",
            linestyle="dashed",
            linewidth=2.0,
            marker="^",
        )
        plt.plot(
            np.arange(common.args.epoch) + 1,
            self.test_loss_history,
            label="test loss",
            color="red",
            linestyle="solid",
            linewidth=2.0,
            marker="o",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(
            np.arange(common.args.epoch) + 1,
            self.train_accuracy_history,
            label="train acc",
            color="red",
            linestyle="dotted",
            linewidth=1.0,
        )
        plt.plot(
            np.arange(common.args.epoch) + 1,
            self.valid_accuracy_history,
            label="valid acc",
            color="red",
            linestyle="dashed",
            linewidth=2.0,
            marker="^",
        )
        plt.plot(
            np.arange(common.args.epoch) + 1,
            self.test_accuracy_history,
            label="test acc",
            color="red",
            linestyle="solid",
            linewidth=2.0,
            marker="o",
        )
        plt.title("Acc={:.5f}".format(self.best_test_accuracy))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid()

        plt.suptitle(
            "lr={lr}, batch_size={batch_size}, mixup={mixup}, manifold-mixup={manifold_mixup}, mixup-alpha={mixup_alpha}, seed={seed}, momentum={momentum}, weight_decay={weight_decay}, distillation={distillation}, soft_loss_weight={soft_loss_weight}, softmax_temperature={softmax_temperature}".format(
                lr=common.args.lr,
                batch_size=common.args.batch_size,
                mixup=common.args.mixup,
                manifold_mixup=common.args.manifold_mixup,
                mixup_alpha=common.args.mixup_alpha,
                seed=common.args.seed,
                momentum=common.args.momentum,
                weight_decay=common.args.weight_decay,
                distillation=common.args.distillation,
                soft_loss_weight=common.args.soft_loss_weight,
                softmax_temperature=common.args.softmax_temperature,
            )
        )

        plt.show()

        figname = os.path.join(common.OUTPUT_DIR, "{}.png".format(common.args.uuid))

        plt.savefig(figname, dpi=100)
        self._slack_notify_image(filename=figname)

    def write_spreadsheet(self):
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]

        # need to put json in the same directory
        credential_json_name = glob.glob("./reporter/*.json")[0]

        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            credential_json_name, scope
        )
        gc = gspread.authorize(credentials)
        worksheet = gc.open("Experiment").sheet1

        worksheet.append_row(
            [
                common.args.uuid,
                common.args.batch_size,
                common.args.lr,
                common.args.mixup,
                common.args.manifold_mixup,
                common.args.mixup_alpha,
                common.args.epoch,
                common.args.seed,
                common.args.momentum,
                common.args.weight_decay,
                common.args.distillation,
                common.args.soft_loss_weight,
                common.args.softmax_temperature,
                self.best_train_accuracy,
                self.best_valid_accuracy,
                self.best_test_accuracy,
            ]
        )
