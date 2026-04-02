import random
from typing import Tuple

import lightning as pl
import numpy as np
import torch
from conceptualizer.loader import (
    ident3d_loader,
    mnist_loader,
    quad_causal_loader,
    quad_loader,
)
from conceptualizer.utils import InterfaceDisentDataset
from disent.dataset import DisentDataset
from disent.dataset.sampling import RandomSampler
from disent.dataset.sampling._base import BaseDisentSampler
from disent.frameworks.vae import AdaVae, BetaTcVae, TripletVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, EncoderConv64
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import ConcatDataset, DataLoader, random_split

from disent_patch import DecoderConv28, EncoderConv28

# performance boost
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda")

# snakemake
smw = snakemake.wildcards
smp = snakemake.params
smo = snakemake.output[0]
batch_size = int(smp["batch_size"])

# seed
seed = int(smw["seed"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
pl.seed_everything(seed, workers=True)

# load and split data
dataset_name = smw["dataset"]
if dataset_name == "quad":
    loader = quad_loader
elif dataset_name == "quad_causal":
    loader = quad_causal_loader
elif dataset_name == "mnist":
    loader = mnist_loader
elif dataset_name == "3dident":
    loader = ident3d_loader
else:
    raise NotImplementedError

_, datasets, C, H, W = loader()
shape = (C, H, W)

train_datasets = {}
val_datasets = {}
generator = torch.Generator(device="cpu")
for label, dataset in datasets.items():
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_datasets[label], val_datasets[label] = random_split(
        dataset, [train_size, val_size], generator=generator
    )
train_dataset = InterfaceDisentDataset(ConcatDataset(list(train_datasets.values())))
val_dataset = InterfaceDisentDataset(ConcatDataset(list(val_datasets.values())))


# prepare the config and dataset supervision
class SupervisedSampler(BaseDisentSampler):
    """
    Sampler for G groups each of size M (concatenated).
    - num_samples == 3 -> return (i, j, k) where j is another member in i's group and k as specified.
    - num_samples == 2 -> return (i, k) omitting j (k chosen the same way).
    """

    def __init__(self, G: int, M: int, num_samples: int):
        if num_samples not in (2, 3):
            raise ValueError("num_samples must be 2 or 3")
        super().__init__(num_samples=num_samples)
        if G < 1 or M < 1:
            raise ValueError("G and M must be >= 1")
        self._G = int(G)
        self._M = int(M)
        self._rng = random.Random()

    def _init(self, dataset):
        if dataset is not None:
            N = len(dataset)
            if N != self._G * self._M:
                raise RuntimeError(
                    f"dataset length {N} does not match G*M = {self._G * self._M}"
                )

    def _sample_idx(self, i: int) -> Tuple[int, ...]:
        G = self._G
        M = self._M
        N = G * M
        if not (0 <= i < N):
            raise IndexError("i out of range")

        group_i = i // M

        # pick k (depends only on group_i)
        if group_i == 0:
            if G < 2:
                raise RuntimeError(
                    "Need at least 2 groups to pick k from another group when i is in group 0"
                )
            gk = self._rng.randrange(1, G)
        else:
            gk = 0
        k = gk * M + self._rng.randrange(M)

        if self.num_samples == 2:
            return (i, k)

        # else num_samples == 3: pick j (different member in same group as i)
        base = group_i * M
        offset_i = i - base
        r = self._rng.randrange(M - 1)  # 0..M-2
        if r >= offset_i:
            r += 1
        j = base + r

        return (i, j, k)


method = smw["arch_flag"]
if method == "Ada-GVAE":
    VAE = AdaVae
    config = VAE.cfg(
        optimizer="adam",
        optimizer_kwargs=dict(lr=1e-3),
        loss_reduction="mean_sum",
        beta=4,
        ada_average_mode="gvae",
        ada_thresh_mode="kl",
    )
    # weakly supervised
    G = len(train_datasets)
    M_train = len(train_dataset) // G
    M_val = len(val_dataset) // G
    train = DisentDataset(
        train_dataset,
        sampler=SupervisedSampler(G, M_train, 2),
    )

    val = DisentDataset(
        val_dataset,
        sampler=SupervisedSampler(G, M_val, 2),
    )
elif method == "BetaTCVAE":
    VAE = BetaTcVae
    config = VAE.cfg(
        optimizer="adam",
        optimizer_kwargs=dict(lr=1e-3),
        loss_reduction="mean_sum",
        beta=1,
    )
    # unsupervised
    train = DisentDataset(
        train_dataset,
        sampler=RandomSampler(num_samples=1),
    )
    val = DisentDataset(
        val_dataset,
        sampler=RandomSampler(num_samples=1),
    )
elif method == "TVAE":
    VAE = TripletVae
    config = VAE.cfg(
        optimizer="adam",
        optimizer_kwargs=dict(lr=1e-3),
        loss_reduction="mean_sum",
        beta=1,
    )
    # supervised
    G = len(train_datasets)
    M_train = len(train_dataset) // G
    M_val = len(val_dataset) // G
    train = DisentDataset(
        train_dataset,
        sampler=SupervisedSampler(G, M_train, 3),
    )

    val = DisentDataset(
        val_dataset,
        sampler=SupervisedSampler(G, M_val, 3),
    )
else:
    raise NotImplementedError

# prepare dataloaders
train_dataloader = DataLoader(
    train,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
)

# logger
csv_logger = CSVLogger(
    save_dir=f"results/dataset={smw['dataset']}/arch=disent/method={smw['arch_flag']}/seed={smw['seed']}/",
    name="",
    version=0,
)


# create the pytorch lightning system
class VAEWithValLogging(VAE):
    def validation_step(self, batch, batch_idx):
        # Disable gradient tracking (validation doesn't need gradients)
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx)

        # Log as validation loss
        self.log("val_loss", loss, prog_bar=True)

        return loss


if dataset_name == "mnist":
    model = AutoEncoder(
        encoder=EncoderConv28(x_shape=shape, z_size=smp["latent_dim"], z_multiplier=2),
        decoder=DecoderConv28(x_shape=shape, z_size=smp["latent_dim"]),
    )
else:
    model = AutoEncoder(
        encoder=EncoderConv64(x_shape=shape, z_size=smp["latent_dim"], z_multiplier=2),
        decoder=DecoderConv64(x_shape=shape, z_size=smp["latent_dim"]),
    )

module = VAEWithValLogging(model, cfg=config)

# add early stopping
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    min_delta=0.0,
    verbose=True,
)


# train
trainer = pl.Trainer(
    logger=csv_logger, max_epochs=smp["epochs"], callbacks=[early_stop_callback]
)
trainer.fit(module, train_dataloader, val_dataloader)

# save
# "results/dataset={dataset}/arch=disent/method={arch_flag}/seed={seed}/model.ckpt",
trainer.save_checkpoint(smo)
