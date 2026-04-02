from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Resize

from .utils import quad_composition, quad_concept_learning

data_root = str(Path(__file__).resolve().parents[1] / "expt" / "data")


class NoLabelImageFolder(ImageFolder):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)  # Discard label
        return image

    @classmethod
    def to_tensor_dataset(cls, root, transform=None):
        """Load entire folder to memory as TensorDataset"""
        folder = cls(root=root, transform=transform or transforms.ToTensor())
        images = torch.stack([folder[i] for i in range(len(folder))])
        return TensorDataset(images)


def mnist_loader(holdout=False):
    labels = ("obs", "scaled", "shear", "shift", "swel", "thic", "thin")
    h = "holdout_" if holdout else ""
    csvs = [
        pd.read_csv(
            f"{data_root}/mnist/{h}normalized_mnist_{lab}.csv",
            header=None,
        )
        for lab in labels
    ]
    uf = nn.Unflatten(-1, (1, 28, 28))
    ds = (
        TensorDataset(uf(torch.tensor(d.values[:, :-2], dtype=torch.float32)))
        for d in csvs
    )
    concepts = labels
    datasets = {k: v for k, v in zip(concepts, ds)}

    mnist_channels = 1
    mnist_height = 28
    mnist_width = 28

    return labels, datasets, mnist_channels, mnist_height, mnist_width


def ident3d_loader(holdout=False):
    conc_lrn = ["obs", "bg", "obj", "sl"]
    compo = ["bg-obj", "bg-sl", "obj-sl"]
    labels = conc_lrn + (compo if holdout else [])
    data_root = str(Path(__file__).resolve().parents[1] / "expt" / "data" / "3dident")

    datasets = {}

    transform = transforms.Compose(
        [
            Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    for label in labels:
        if label in compo:
            subdir = f"holdout_{label}"
        else:
            subdir = f"holdout_{label}" if holdout else label

        full_dataset = NoLabelImageFolder(
            root=data_root + f"/{subdir}/",
            transform=transform,
        )
        datasets[label] = full_dataset

    channels = 3
    height = 64
    width = 64

    return labels, datasets, channels, height, width


def quad_loader(holdout=False):
    if holdout:
        labels = quad_concept_learning + quad_composition
        csvs = [
            np.load(f"{data_root}/quad/holdout/{lab}_holdout/dataset.npz")["images"]
            for lab in labels
        ]
    else:
        labels = (
            "obs1",
            "ivn_quad1",
            "ivn_quad2",
            "ivn_quad3",
            "ivn_quad4",
            "ivn_size",
            "ivn_orientation",
        )  # temp folder names
        csvs = [
            np.load(f"{data_root}/quad/{lab}/dataset.npz")["images"] for lab in labels
        ]
        labels = quad_concept_learning
    ds = (TensorDataset(torch.tensor(d) / 255.0) for d in csvs)
    concepts = labels
    datasets = {k: v for k, v in zip(concepts, ds)}

    quad_channels = 3
    quad_height = 64
    quad_width = 64

    return labels, datasets, quad_channels, quad_height, quad_width


def quad_causal_loader(holdout=False):
    labels = (
        quad_concept_learning + quad_composition if holdout else quad_concept_learning
    )
    pre = "holdout" if holdout else "train"
    datasets = {
        label: NoLabelImageFolder.to_tensor_dataset(
            root=f"{data_root}/quad_causal/{pre}_{label}/",
            transform=transforms.ToTensor(),
        )
        for label in labels
    }

    channels = 3
    height = 64
    width = 64

    return labels, datasets, channels, height, width
