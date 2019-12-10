import ast
import glob
import gzip
import os
import pickle
import random
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SIZE = 32
SAMPLE_PER_CLASS = 300
BASE_SIZE = 256
TRAIN_RATIO = 0.8

CSVS = glob.glob("./quickdraw_original/train_simplified/*.csv")


def draw_cv2(raw_strokes, SIZE=256, lw=6):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)

    for stroke in raw_strokes:
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(
                img,
                (stroke[0][i], stroke[1][i]),
                (stroke[0][i + 1], stroke[1][i + 1]),
                255,
                lw,
            )

    if SIZE != BASE_SIZE:
        return cv2.resize(img, (SIZE, SIZE))
    else:
        return img


def image_generator(filename):
    while True:
        for df in pd.read_csv(filename, chunksize=SAMPLE_PER_CLASS):
            df["drawing"] = df["drawing"].apply(ast.literal_eval)
            x = np.zeros((len(df), SIZE, SIZE))

            for i, raw_strokes in enumerate(df.drawing.values):
                x[i] = draw_cv2(raw_strokes, SIZE=SIZE, lw=6)
            x = x.reshape((len(df), SIZE, SIZE, 1)).astype(np.uint8)
            yield x


if __name__ == "__main__":
    if os.path.exists("../input/quickdraw"):
        shutil.rmtree("../input/quickdraw")
    os.mkdir("../input/quickdraw")

    imgs = []
    labels = []
    split_keys = []

    for i, f in enumerate(tqdm(CSVS)):
        train_datagen = image_generator(filename=f)
        x = next(train_datagen)

        for j, img in enumerate(x[:, :, :, 0]):
            imgs.append(img)
            labels.append(i)

    random.seed(0)
    train_index = np.sort(
        random.sample(np.arange(len(labels)).tolist(), int(len(labels) * TRAIN_RATIO))
    )
    test_index = np.sort(np.delete(np.arange(len(labels)), train_index))

    with gzip.open("../input/quickdraw/train_images.pkl.gz", mode="wb") as pk:
        pickle.dump(np.array(imgs)[train_index], pk)

    with gzip.open("../input/quickdraw/train_labels.pkl.gz", mode="wb") as pk:
        pickle.dump(np.array(labels)[train_index], pk)

    with gzip.open("../input/quickdraw/test_images.pkl.gz", mode="wb") as pk:
        pickle.dump(np.array(imgs)[test_index], pk)

    with gzip.open("../input/quickdraw/test_labels.pkl.gz", mode="wb") as pk:
        pickle.dump(np.array(labels)[test_index], pk)
