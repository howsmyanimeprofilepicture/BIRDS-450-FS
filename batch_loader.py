import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from typing import List, Dict, NamedTuple
from protonet import ProtoNet
import torch.nn.functional as F
from tqdm import tqdm


def img2arr(image_path: str,
            size: int = 32) -> torch.Tensor:
    image_tensor = Image.open(image_path)
    image_tensor = image_tensor.resize(size=(32, 32))
    image_tensor = np.asarray(image_tensor)/255
    image_tensor = image_tensor.transpose(-1, 0, 1)
    image_tensor = (
        torch.from_numpy(image_tensor)
        .to(torch.float32)
    )
    return image_tensor


def get_images_and_labels(data_path="./BIRDS-450-FS"):
    images: Dict[str, List[torch.Tensor]] = {
        path.name: [img2arr(image_path)
                    for image_path in path.glob("*")]
        for path in Path(data_path).glob("*")
    }
    labels: List[str] = [*images.keys()]

    return images, labels


def split_labels(labels: List[str], split_ratio=0.8):
    assert split_ratio <= 1 and split_ratio > 0
    random.shuffle(labels)
    train_labels = labels[:int(len(labels) * split_ratio)]
    test_labels = labels[int(len(labels) * split_ratio):]

    return (train_labels, test_labels)


class Batch(NamedTuple):
    X_spt: torch.Tensor
    X_qry: torch.Tensor
    y_spt: torch.LongTensor
    y_qry: torch.LongTensor


class SamplingResult(NamedTuple):
    batch: Batch
    id2cls: List[str]


class BatchLoader:
    def __init__(self,
                 num_way: int,
                 num_spt: int,
                 device,
                 num_qry: int = 1,
                 data_path="./BIRDS-450-FS",
                 split_ratio=0.8) -> None:
        (self.images,
         self.labels) = get_images_and_labels(data_path)
        (self.train_labels,
         self.test_labels) = split_labels(self.labels)

        self.num_way = num_way
        self.num_spt = num_spt
        self.device = device
        self.num_qry = num_qry
        self.split_ratio = split_ratio

    def sample(self):
        classes: List[str] = random.sample(self.labels,
                                           self.num_way)
        c, h, w = self.images[classes[0]][0].size()

        X_spt: List[torch.Tensor] = []
        X_qry: List[torch.Tensor] = []
        y_spt: List[int] = []
        y_qry: List[int] = []

        for i, cls in enumerate(classes):
            image_tensors = random.sample(self.images[cls],
                                          self.num_spt + self.num_qry)
            X_spt += image_tensors[:self.num_spt]
            X_qry += image_tensors[self.num_spt:]
            y_spt += [i] * self.num_spt
            y_qry += [i] * self.num_qry

        X_spt = torch.stack(X_spt).to(device=self.device)
        X_qry = torch.stack(X_qry).to(device=self.device)
        y_spt = torch.tensor(y_spt).to(device=self.device)
        y_qry = torch.tensor(y_qry).to(device=self.device)

        assert X_spt.size() == (self.num_way*self.num_spt,
                                c, h, w)
        assert X_qry.size() == (self.num_way*self.num_qry,
                                c, h, w)

        return SamplingResult(batch=Batch(X_spt,
                                          X_qry,
                                          y_spt,
                                          y_qry),
                              id2cls=classes)


def visualize_batch(batch: Batch,
                    id2cls: List[str]):
    (X_spt,
     X_qry,
     y_spt,
     y_qry) = batch

    num_way: int = torch.unique(y_spt).size(0)
    num_spt: int = y_spt.size(0)//num_way
    num_qry: int = y_qry.size(0)//num_way

    fig, axes = plt.subplots(
        ncols=num_way,
        nrows=num_spt,
        figsize=(num_way*3, 1 + num_spt*3,),
        subplot_kw={
            "xticks": [],
            "yticks": []
        }
    )
    fig.suptitle("Support Set", fontsize=20)
    axes = axes.T.reshape(-1)
    for i, ax in enumerate(axes):
        ax.imshow(X_spt[i, :, :, :].permute(1, 2, 0))
        ax.set_title(id2cls[y_spt[i]])

    fig, axes = plt.subplots(
        ncols=num_way,
        nrows=num_qry,
        figsize=(num_way*3, 1 + num_qry*3),
        subplot_kw={
            "xticks": [],
            "yticks": []
        }
    )
    fig.suptitle("Query Set", fontsize=20)
    axes = axes.T.reshape(-1)
    for i, ax in enumerate(axes):
        ax.imshow(X_qry[i, :, :, :].permute(1, 2, 0))
        ax.set_title(id2cls[y_qry[i]])

    plt.show()
