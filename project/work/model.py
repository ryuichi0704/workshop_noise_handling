import numpy as np
import torch
import torch.nn as nn
import torchvision

import common


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        model = torchvision.models.resnet18(pretrained=False)

        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # for arbital image size
        self.fc = nn.Linear(model.fc.in_features, common.yml["train"]["n_classes"])

    def _get_layer_mix(self, mode):
        if common.args.mixup and mode == "train":
            if common.args.manifold_mixup:
                layer_mix = np.random.choice([0, 1, 2])
            else:
                layer_mix = 0
        else:
            layer_mix = None

        return layer_mix

    def _mixup_data(self, inputs, labels):
        if common.args.mixup_alpha > 0:
            lam = np.random.beta(common.args.mixup_alpha, common.args.mixup_alpha)
        else:
            lam = 1

        batch_size = inputs.size()[0]
        index = torch.randperm(batch_size).to(common.DEVICE)

        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        labels_a, labels_b = labels, labels[index]

        return mixed_inputs, labels_a, labels_b, lam

    def forward(self, x, y, mode):
        layer_mix = self._get_layer_mix(mode)

        if layer_mix == 0:  # normal mixup
            x, y_a, y_b, lam = self._mixup_data(x, y)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if layer_mix == 1:
            x, y_a, y_b, lam = self._mixup_data(x, y)

        x = self.layer2(x)

        if layer_mix == 2:
            x, y_a, y_b, lam = self._mixup_data(x, y)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x.squeeze())

        if common.args.mixup and mode == "train":
            return x, y_a, y_b, lam
        else:
            return x
