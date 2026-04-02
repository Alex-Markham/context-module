import random

import lightning as pl
import numpy as np
import pandas as pd
import torch
from conceptualizer.loader import (
    ident3d_loader,
    mnist_loader,
    quad_causal_loader,
    quad_loader,
)
from disent.frameworks.vae import AdaVae, BetaTcVae, TripletVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, EncoderConv64
from ot.sliced import sliced_wasserstein_distance
from torch import nn
from torch.utils.data import DataLoader

from disent_patch import DecoderConv28, EncoderConv28

# performance boost
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# snakemake
smi = snakemake.input
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

# prepare the data
dataset_name = smw["dataset"]
sep = "_"
if dataset_name == "quad":
    loader = quad_loader
elif dataset_name == "quad_causal":
    loader = quad_causal_loader
elif dataset_name == "mnist":
    loader = mnist_loader
elif dataset_name == "3dident":
    loader = ident3d_loader
    sep = "-"
else:
    raise NotImplementedError

_, datasets, C, H, W = loader(holdout=True)
shape = (C, H, W)

# load the trained model
method = smw["arch_flag"]
if method == "Ada-GVAE":
    VAE = AdaVae
elif method == "BetaTCVAE":
    VAE = BetaTcVae
elif method == "TVAE":
    VAE = TripletVae
else:
    raise NotImplementedError

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

module = VAE.load_from_checkpoint(smi["module"], model=model)
module.eval()


# evaluate
def _compute_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


elbos = {}
ots = {}
num_ot_batchs = 20
with torch.no_grad():
    for concept, dataset in datasets.items():
        val_lb = 0.0
        normalizer = 0.0
        ot = 0.0
        orig_batches = []
        gen_batches = []
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
        )
        for idx, batch in enumerate(dataloader):
            # normalize batch to x = images tensor
            if isinstance(batch, (list, tuple)):
                x = batch[0]  # first element is images: shape [B, C, H, W]
            elif isinstance(batch, dict):
                # choose key that holds images
                x = batch.get("x", next(iter(batch.values())))
            else:
                # batch itself is the image tensor
                x = batch
            bsz = len(x)
            recon_x = torch.sigmoid(module(x))
            mu, log_var = module._model._encoder(x)
            lb = _compute_loss(recon_x, x, mu, log_var)
            val_lb += lb.item()
            normalizer += len(x)
            if idx < num_ot_batchs:
                orig_batches.append(x.view(bsz, -1))
                z = torch.randn(bsz, smp["latent_dim"])
                dec = torch.sigmoid(module._model._decoder(z))
                gen_batches.append(dec.view(bsz, -1))
        val_lb /= normalizer
        elbos[concept] = val_lb
        orig = torch.cat(orig_batches, dim=0)
        gen = torch.cat(gen_batches, dim=0)
        ots[concept] = sliced_wasserstein_distance(orig, gen).item()


# reconstruction
def bpd(raw_loss):
    num_pixels = H * W
    return raw_loss / (num_pixels * np.log(2))


ids = [v for k, v in elbos.items() if sep not in k]
id_bpd = bpd(sum(ids) / len(ids))
oods = [v for k, v in elbos.items() if sep in k]
ood_bpd = bpd(sum(oods) / len(oods)) if oods else None
reconstruction = {
    "final_val_elbo_bpd": [id_bpd],
    "ood_val_elbo_bpd": [ood_bpd],
}

# concept learning
concept_learning = {k: [v] for k, v in ots.items() if sep not in k}

# composition
composition = {k: [v] for k, v in ots.items() if sep in k}

# save results
result = pd.DataFrame(dict(smw) | reconstruction | concept_learning | composition)
result.to_csv(snakemake.output[0], index=False)
