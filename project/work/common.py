import argparse
import os
import random
import uuid
import yaml

import numpy as np
import torch

# Load setting yaml
with open("./settings.yaml") as file:
    yml = yaml.safe_load(file)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--image-size", type=int, default=32)
parser.add_argument("--valid-rate", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--uuid", type=str, default=str(uuid.uuid4())[0:8])
parser.add_argument("-mixup", "--mixup", action="store_true")
parser.add_argument("-manifold-mixup", "--manifold-mixup", action="store_true")
parser.add_argument("--mixup-alpha", type=float, default=0.2)
parser.add_argument("-distillation", "--distillation", action="store_true")
parser.add_argument(
    "--soft-loss-weight", type=float, default=1.0
)  # hard : soft = 1 : soft-loss-weight
parser.add_argument("--softmax-temperature", type=float, default=1.0)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--accumulation-steps", type=int, default=1)

args = parser.parse_args()

if args.manifold_mixup:
    args.mixup = True

# GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Assertion
gpus = [int(n) for n in args.gpu.split(",")]

assert not (
    len(gpus) > 1 and (args.mixup or args.manifold_mixup)
), "(manifold) mixup in this repository does not work with dataparallel (for simple implementation). See https://github.com/vikasverma1077/manifold_mixup/issues/4"
assert (
    args.distillation is False or args.mixup is False
), "simultaneous use of mixup and distillation is not implemented yet."
assert (
    args.softmax_temperature==1
), "not implemented."


# Seed fix
random.seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Set output dir
OUTPUT_DIR = "../output/lr{lr}_bs{bs}_epoch{epoch}_seed{seed}".format(
    lr=args.lr, bs=args.batch_size, epoch=args.epoch, seed=args.seed
)

if args.mixup:
    if args.manifold_mixup:
        OUTPUT_DIR += "_manifoldmixup_alpha{}".format(args.mixup_alpha)
    else:
        OUTPUT_DIR += "_mixup_alpha{}".format(args.mixup_alpha)

if args.distillation:
    OUTPUT_DIR += "_distillation_weight{}_temperature{}".format(
        args.soft_loss_weight, args.softmax_temperature
    )

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
