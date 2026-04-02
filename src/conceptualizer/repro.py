import argparse
import os
import random
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ot.sliced import sliced_wasserstein_distance
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchinfo import summary
from tqdm import tqdm

from .arch import get_architectures
from .conceptualizer import dec_conceptualizer
from .loader import (
    mnist_loader,
    quad_causal_loader,
    quad_loader,
)
from .metric_eval_fix import get_loader
from .utils import (
    create_readme,
    generate_tag_path,
    init_log,
    log_epoch,
    save_model_architecture,
)

# Suppress specific warnings
warnings.filterwarnings(
    "ignore", message="A NumPy version >=1.17.3 and <1.25.0 is required"
)

print("imports imported...")


class VAE(nn.Module):
    def __init__(
        self,
        arch_id="original",
        arch_flag=None,
        latent_dim=None,
        eps_dim=None,
        eps_in_width=None,
        eps_out_width=None,
        eps_depth=None,
        c_dim=None,
        c_width=None,
        concepts=None,
        device=None,
        VAE_ARCHITECTURES=None,
    ):
        super(VAE, self).__init__()

        arch = VAE_ARCHITECTURES[arch_id]
        self.encoder = arch["encoder"]
        self.decoder = arch["decoder"]
        encode_dim = arch["encode_dim"]
        self.decode_dim = arch["decode_dim"]

        if "vanilla" in arch_flag:
            self.fc_mu = nn.Linear(encode_dim, self.decode_dim)
            self.fc_var = nn.Linear(encode_dim, self.decode_dim)
        else:
            self.fc_mu = nn.Linear(encode_dim, latent_dim)
            self.fc_var = nn.Linear(encode_dim, latent_dim)

        self.ivn_eps, self.expressive_layer, self.causal_layer, self.unpool = (
            dec_conceptualizer(
                eps_dim,
                eps_in_width,
                eps_out_width,
                eps_depth,
                c_dim,
                c_width,
                concepts,
                self.decode_dim,
                arch_flag,
                device,
            )
        )

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def conceptualize(self, z, batch_label: str, arch_flag=None):
        activation = nn.GELU()
        if arch_flag == "single-pooled-concept":
            z = self.ivn_eps(z, ["obs"])
        else:
            z = self.ivn_eps(z, batch_label)
        z = activation(z)
        epsilon = self.expressive_layer(z)
        if arch_flag == "single-pooled-concept":
            c = self.causal_layer(epsilon, ["obs"])
        else:
            c = self.causal_layer(epsilon, batch_label)
        return self.unpool(c)

    def decode(self, c):
        return self.decoder(c)

    def forward(self, x, batch_label: str, arch_flag=None):
        h = self.encode(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        if "vanilla" in arch_flag:
            return self.decode(z), mu, log_var
        c = self.conceptualize(z, batch_label, arch_flag=arch_flag)
        return self.decode(c), mu, log_var


class DictDataset(Dataset):
    def __init__(self, data_dict, arch_flag=None):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.lengths = [len(data) for data in data_dict.values()]
        self.total_length = sum(self.lengths)
        self.arch_flag = arch_flag

    def __len__(self):
        if self.arch_flag == "vanilla-obs":
            return len(self.data_dict["obs"])
        return self.total_length

    def __getitem__(self, item):
        idx, key = item
        dataset = self.data_dict[key]
        return dataset[idx], key


class DictBatchSampler:
    def __init__(self, data_dict, batch_size, arch_flag=None):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.batch_size = batch_size
        self.total_samples = sum(len(dataset) for dataset in data_dict.values())
        self.arch_flag = arch_flag
        if self.arch_flag == "vanilla-obs":
            self.total_samples = len(self.data_dict["obs"])

    def __iter__(self):
        samples_yielded = 0
        while samples_yielded < self.total_samples:
            key = random.choice(self.keys)
            if self.arch_flag == "vanilla-obs":
                key = "obs"
            dataset = self.data_dict[key]
            remaining = min(self.batch_size, self.total_samples - samples_yielded)
            indices = torch.randperm(len(dataset), device="cpu")[:remaining]
            yield [(idx.item(), key) for idx in indices]
            samples_yielded += len(indices)

    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size


def dict_collate_fn(batch):
    data = torch.stack(
        [item[0][0] if isinstance(item[0], tuple) else item[0] for item in batch]
    )
    key = batch[0][1]
    return data, key


print("classes defined...")


def main():
    EXPT_NOTES = "Experimenting on such and such 1234"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch_flag",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        required=True,
        help="Architecture flag (must be 0, 1, 2, 3, 4, or 5)",
    )
    parser.add_argument("--arch_id", type=str, required=True, help="Architecture ID")
    parser.add_argument(
        "--beta",
        type=float,
        required=False,
        default=1,
        help="regularization parameter for KL divergence term of loss",
    )
    parser.add_argument(
        "--reg",
        type=str,
        default="0.0.0.0",
        help="four regularizer weights: group sparsity, L2 sparsity, _, _",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="2.15.5",
        help="three dims for expressiveness: eps_depth, eps_in, and eps_out=c_width",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--microbatches",
        type=int,
        default=1,
        help="number of batches per optimizer step",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "quad",
            "quad_causal",
        ],
        default="mnist",
        help="dataset to use",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="ncfa-expts",
        help="which gpu to use",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default="",
        help="just plot/eval trained model",
    )
    parser.add_argument(
        "--finetune_pt",
        type=str,
        default="",
        help="path to vanilla-pooled checkpoint for fine-tuning (only used with arch_flag 4 or 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for reproducibility",
    )
    parser.add_argument(
        "--snakemake", action="store_true", help="format savedir for snakemake"
    )
    args = parser.parse_args()

    if args.arch_flag in [4, 5] and not args.finetune_pt:
        raise ValueError("arch_flag 4/5 requires --finetune_pt to be specified")
    reg = [float(s) for s in args.reg.split(".")]
    exp = [int(s) for s in args.exp.split(".")]

    print("args parsed...")
    assert torch.cuda.is_available(), "This script requires a GPU to run!"

    seed = args.seed
    if seed is not None:
        torch.manual_seed(args.seed)
        random.seed(seed)

    # Load data
    if args.dataset == "mnist":
        labels, datasets, img_channels, img_height, img_width = mnist_loader()
        h_labels, h_datasets, _, _, _ = mnist_loader(holdout=True)
        if "simple64" in args.arch_id:
            warnings.warn("Incompatible architecture!", UserWarning)
    elif args.dataset == "quad":
        labels, datasets, img_channels, img_height, img_width = quad_loader()
        h_labels, h_datasets, _, _, _ = quad_loader(holdout=True)
        if "conv" in args.arch_id:
            warnings.warn("Incompatible architecture!", UserWarning)
        split_char = "_"
    elif args.dataset == "quad_causal":
        labels, datasets, img_channels, img_height, img_width = quad_causal_loader()
        h_labels, h_datasets, _, _, _ = quad_causal_loader(holdout=True)
        if "conv" in args.arch_id:
            warnings.warn("Incompatible architecture!", UserWarning)
        split_char = "_"
    print("data loaded...\n")
    concepts = labels

    # Train/val split
    train_datasets = {}
    val_datasets = {}
    generator = torch.Generator(device="cpu")
    for label, dataset in datasets.items():
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_datasets[label], val_datasets[label] = random_split(
            dataset, [train_size, val_size], generator=generator
        )

    # Architecture parameters
    num_causal_vars = len(concepts) - 1
    eps_dim = num_causal_vars
    eps_depth = exp[0]
    eps_in_width = exp[1]
    eps_out_width = exp[2]
    c_dim = num_causal_vars
    c_width = exp[2]
    latent_dim = eps_dim * eps_in_width
    hidden_dim = 128

    options = [
        "vanilla-obs",
        "vanilla-pooled",
        "concepts",
        "single-pooled-concept",
        "fine-tune-concept",
        "fine-tune-concept-unfreeze",
    ]
    arch_flag = options[args.arch_flag]
    arch_id = args.arch_id
    VAE_ARCHITECTURES = get_architectures(hidden_dim)

    # Training parameters
    microbatches = args.microbatches
    batch_size = 512 // microbatches
    learning_rate = 1e-3
    epochs = args.epochs

    # Early stopping parameters
    early_stopping_patience = 20  # Number of epochs to wait for improvement
    early_stopping_min_delta = (
        1e-4  # Minimum change in validation ELBO to qualify as improvement
    )

    if not args.snakemake:
        epochs_per_checkpoint = 1
        expt_tag = f"{args.dataset}_{arch_id}-{arch_flag}-{args.reg}-{args.exp}"
        save_dir = generate_tag_path(expt_tag)
    else:
        epochs_per_checkpoint = args.epochs + 1
        expt_tag = "snakemake"
        save_dir = f"results/dataset={args.dataset}/arch={args.arch_id}/method={args.arch_flag}/expressivity={args.exp}/beta={args.beta:.1f}/sparsity={args.reg}/seed={args.seed}/"

    os.makedirs(save_dir, exist_ok=True)
    print(f"savedir is {save_dir}...\n")

    params = {
        "dimensions": {
            "eps_dim": eps_dim,
            "eps_in_width": eps_in_width,
            "eps_out_width": eps_out_width,
            "eps_depth": eps_depth,
            "c_dim": c_dim,
            "c_width": c_width,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "arch_flag": arch_flag,
        },
        "training": {
            "batch_size": batch_size,
            "microbatches": microbatches,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "group": reg[0],
            "ell_2": reg[1],
            "_": reg[2],
            "_": reg[3],
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "finetune_pt": args.finetune_pt,
        },
        "experiment": {
            "epochs_per_checkpoint": epochs_per_checkpoint,
            "expt_tag": expt_tag,
            "save_dir": save_dir,
        },
        "data": {
            "concepts": concepts,
            "colorized_contexts": None,
            "extra_colour_context": None,
            "nsamples": [len(d) for d in datasets.values()],
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        arch_id=arch_id,
        arch_flag=arch_flag,
        latent_dim=latent_dim,
        eps_dim=eps_dim,
        eps_in_width=eps_in_width,
        eps_out_width=eps_out_width,
        eps_depth=eps_depth,
        c_dim=c_dim,
        c_width=c_width,
        concepts=concepts,
        device=device,
        VAE_ARCHITECTURES=VAE_ARCHITECTURES,
    ).to(device)

    if arch_flag in ["fine-tune-concept", "fine-tune-concept-unfreeze"]:
        print(f"Loading checkpoint from {args.finetune_pt} for fine-tuning...")
        checkpoint_path = args.finetune_pt  # os.path.join(args.finetune_pt, "vae.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        pretrained_dict = checkpoint["model_state_dict"]
        model_dict = model.state_dict()

        pretrained_dict_filtered = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict
            and v.shape == model_dict[k].shape
            and not any(
                x in k
                for x in [
                    "ivn_eps",
                    "expressive_layer",
                    "causal_layer",
                    "unpool",
                ]
            )
        }

        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict)

        print(f"Loaded {len(pretrained_dict_filtered)} layers from checkpoint")

        for name, param in model.named_parameters():
            if name in pretrained_dict_filtered:
                param.requires_grad = False
                print(f"Froze layer: {name}")

        print("\nTrainable parameters (conceptualizer only):")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")

        print(f"\n{'=' * 60}")
        if arch_flag == "fine-tune-concept-unfreeze":
            print("Two-phase fine-tuning strategy:")
            print(
                f"  Phase 1 (epochs 0-{int(0.2 * epochs) - 1}): Train conceptualizer only"
            )
            print(
                f"  Phase 2 (epochs {int(0.2 * epochs)}-{epochs - 1}): Train all weights"
            )
        else:
            print("Single-phase fine-tuning strategy:")
            print("  Training conceptualizer only (encoder/decoder remain frozen)")
        print(f"{'=' * 60}\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("setting up dataloaders...\n")
    train_dataset = DictDataset(train_datasets, arch_flag=arch_flag)
    batch_sampler = DictBatchSampler(
        train_datasets, batch_size=batch_size, arch_flag=arch_flag
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, collate_fn=dict_collate_fn
    )

    val_dataset = DictDataset(val_datasets, arch_flag=arch_flag)
    val_batch_sampler = DictBatchSampler(
        val_datasets, batch_size=batch_size, arch_flag=arch_flag
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_batch_sampler, collate_fn=dict_collate_fn
    )

    if (
        args.dataset == "quad"
        or args.dataset == "quad_causal"
        or args.dataset == "mnist"
    ):
        h_dataset = DictDataset(h_datasets, arch_flag=arch_flag)
        h_batch_sampler = DictBatchSampler(
            h_datasets, batch_size=batch_size, arch_flag=arch_flag
        )
        h_loader = DataLoader(
            h_dataset, batch_sampler=h_batch_sampler, collate_fn=dict_collate_fn
        )
    print("...done\n")

    # Calculate total training iterations for scheduler
    iters_per_epoch = len(train_loader) // microbatches
    total_iters = iters_per_epoch * epochs
    warmup_iters = 200

    # Create learning rate scheduler with warmup and cosine annealing
    # Minimum LR is set to 0.01 * max_lr
    min_lr = 0.01 * learning_rate

    # Warmup scheduler: linearly increase from 0 to learning_rate over warmup_iters
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-10,  # Start from nearly 0
        end_factor=1.0,  # End at learning_rate
        total_iters=warmup_iters,
    )

    # Cosine annealing scheduler: decrease from learning_rate to min_lr
    cosine_iters = total_iters - warmup_iters
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_iters, eta_min=min_lr)

    # Combine warmup and cosine annealing
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters],
    )

    def loss_function(
        recon_x,
        x,
        mu,
        logvar,
        causal_grouped=None,
        causal_unpooled=None,
        causal_pooled=None,
    ):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if causal_grouped is not None and reg[0]:
            group = torch.norm(causal_grouped, p=1)
        else:
            group = 0

        if causal_unpooled is not None and reg[1]:
            ell_2 = torch.norm(causal_unpooled, p=2)
        else:
            ell_2 = 0

        KLD *= args.beta
        return BCE + KLD + reg[0] * group + reg[1] * ell_2

    def val_loss_function(recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD *= args.beta
        return BCE + KLD, BCE

    def plot_reconstructions(epoch=None, subdir=None):
        model.eval()
        concepts_list = list(datasets.keys())
        fig, axes = plt.subplots(
            len(concepts_list) * 2, 8, figsize=(15, 4 * len(concepts_list))
        )
        for idx, concept in enumerate(concepts_list):
            for batch, batch_label in train_loader:
                if batch_label == concept:
                    break
            batch = batch.to(device)
            with torch.no_grad():
                recon, _, _ = model(batch, [concept], arch_flag=arch_flag)

            for i in range(8):
                if args.dataset == "mnist":
                    axes[idx * 2, i].imshow(batch[i].cpu().squeeze(), cmap="gray")
                else:
                    axes[idx * 2, i].imshow(
                        batch[i].cpu().squeeze().permute(1, 2, 0), cmap="gray"
                    )
                axes[idx * 2, i].axis("off")
                axes[idx * 2 + 1, i].imshow(
                    recon[i]
                    .cpu()
                    .reshape(img_channels, img_height, img_width)
                    .permute(1, 2, 0),
                    cmap="gray",
                )
                axes[idx * 2 + 1, i].axis("off")

            axes[idx * 2, 0].set_title(f"Original {concept}")
            axes[idx * 2 + 1, 0].set_title(f"Reconstructed {concept}")

        if epoch is None:
            fig.suptitle("Reconstructions")
            plt.tight_layout()
            plt.savefig(f"{save_dir}reconstructions.png")
        elif subdir is None:
            fig.suptitle(f"Reconstructions / Epoch {epoch}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}reconstructions_epoch_{epoch}.png")
        else:
            fig.suptitle(f"Reconstructions / Epoch {epoch}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}{subdir}/reconstructions_epoch_{epoch}.png")
        plt.close()

    def plot_training_loss(losses, epoch=None, subdir=None):
        plt.figure()
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if epoch is None:
            plt.savefig(f"{save_dir}training_loss.png")
        elif subdir is None:
            plt.savefig(f"{save_dir}training_loss_epoch_{epoch}.png")
        else:
            plt.savefig(f"{save_dir}{subdir}/training_loss_epoch_{epoch}.png")
        plt.close()

    def plot_validation_loss(elbo, bce, epoch=None, subdir=None):
        epochs_list = list(range(len(elbo)))
        df = pd.DataFrame({"epoch": epochs_list, "ELBO": elbo, "BCE": bce})
        melted = df.melt(
            id_vars="epoch",
            value_vars=["ELBO", "BCE"],
            var_name="Metric",
            value_name="Loss",
        )

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=melted, x="epoch", y="Loss", hue="Metric", marker="o")
        plt.title("Validation Loss Components")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.tight_layout()
        if epoch is None:
            plt.savefig(f"{save_dir}validation_loss.png")
        elif subdir is None:
            plt.savefig(f"{save_dir}validation_loss_epoch_{epoch}.png")
        else:
            plt.savefig(f"{save_dir}{subdir}/validation_loss_epoch_{epoch}.png")
        plt.close()

    def _plot_random_samples(concepts_input, epoch=None, num_samples=8):
        model.eval()
        model.batch_label = concepts_input
        with torch.no_grad():
            gen = lambda seed: torch.Generator(device=device).manual_seed(seed)
            if "vanilla" in arch_flag:
                z = torch.randn(
                    num_samples, model.decode_dim, generator=gen(0), device=device
                )
                samples = model.decode(z)
            else:
                z = torch.randn(
                    num_samples, latent_dim, generator=gen(0), device=device
                )
                c = model.conceptualize(z, concepts_input, arch_flag=arch_flag)
                samples = model.decode(c)
        fig, axes = plt.subplots(2, 4, figsize=(8, 4))
        for i, ax in enumerate(axes.flat):
            ax.imshow(
                samples[i]
                .cpu()
                .reshape(img_channels, img_height, img_width)
                .permute(1, 2, 0),
                cmap="gray",
            )
            ax.axis("off")
        concepts_input = [arch_flag] if concepts_input == [None] else concepts_input
        fig.suptitle(f"Concept: {'-'.join(concepts_input)}")
        plt.tight_layout()
        dir_path = f"{save_dir}random_samples"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if epoch is None:
            plt.savefig(
                f"{dir_path}/{'-'.join(concepts_input)}.png",
                dpi=150,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"{dir_path}/{'-'.join(concepts_input)}_epoch_{epoch}.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close()

    def plot_random_samples(epoch=None):
        concepts_list = (
            datasets.keys()
            if arch_flag
            in [
                "concepts",
                "fine-tune-concept",
                "fine-tune-concept-unfreeze",
            ]
            else [None]
        )
        for concept in concepts_list:
            _plot_random_samples([concept], epoch=epoch)

    def plot_combo_random_samples(epoch=None):
        concepts_list = (
            datasets.keys()
            if arch_flag
            in [
                "concepts",
                "fine-tune-concept",
                "fine-tune-concept-unfreeze",
            ]
            else [None]
        )
        concepts_list = (c for c in concepts_list if c != "obs")
        combos = combinations(concepts_list, 2)
        for concepts_pair in combos:
            _plot_random_samples(concepts_pair, epoch=epoch)

    def gen_ood(num_samples, concepts_input):
        model.eval()
        model.batch_label = concepts_input
        if not os.path.exists(save_dir + "ood_samples"):
            os.makedirs(save_dir + "ood_samples")
        with torch.no_grad():
            gen = lambda seed: torch.Generator(device=device).manual_seed(seed)
            z = torch.randn(num_samples, latent_dim, generator=gen(0), device=device)
            c = model.conceptualize(z, concepts_input, arch_flag=arch_flag)
            samples = model.decode(c).flatten(1).cpu().numpy()
            pd.DataFrame(samples).to_csv(
                save_dir + "ood_samples/" + "-".join(concepts_input) + ".csv.gz",
                compression="gzip",
                index=False,
                header=False,
            )

    def plot_sparsity(epoch=None):
        model.eval()
        c = model.causal_layer

        to_plot_dict = {
            "obs raw": c.obs_weight.detach(),
            "obs pooled": c.pooler(c.obs_weight.detach().unsqueeze(0).unsqueeze(0))
            .squeeze(0)
            .squeeze(0),
            "ivn raw": c.ivn_weight.detach(),
            "ivn pooled": c.pooler(c.ivn_weight.detach().unsqueeze(0).unsqueeze(0))
            .squeeze(0)
            .squeeze(0),
        }

        for key in to_plot_dict:
            to_plot_dict[key] = to_plot_dict[key].cpu().numpy()

        num_plots = len(to_plot_dict)
        fig, axes = plt.subplots(
            num_plots, 2, figsize=(12, 5 * num_plots), constrained_layout=True
        )

        if num_plots == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, (name, weight_array) in enumerate(to_plot_dict.items()):
            ax_heat = axes[row, 0]
            sns.heatmap(
                np.abs(weight_array), ax=ax_heat, cmap="viridis", vmin=0, vmax=1
            )
            ax_heat.set_title(f"{name} -- Sparsity Pattern (Heatmap)")

            ax_hist = axes[row, 1]
            ax_hist.hist(weight_array.flatten(), bins=np.linspace(0, 1, 51))
            ax_hist.set_xlim(0, 1)
            ax_hist.set_title(f"{name} -- Weight Distribution (Histogram)")

        plt.savefig(f"{save_dir}/sparsity.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _sub_recon_flat(concepts_input: str, data_loader, model, num_batches=20):
        model.eval()
        orig_batches = []
        recon_batches = []
        for i, (batch, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            batch = batch.to(device)
            bsz = batch.size(0)
            flat = batch.view(bsz, -1)
            orig_batches.append(flat)
            recon = model(batch, concepts_input, arch_flag=arch_flag)[0].view(bsz, -1)
            recon_batches.append(recon)

        orig_flat = torch.cat(orig_batches, dim=0)
        recon_flat = torch.cat(recon_batches, dim=0)
        return orig_flat, recon_flat

    def _sub_recon_flat_gen(concepts_input: str, data_loader, model, num_batches=20):
        model.eval()
        model.batch_label = concepts_input
        orig_batches = []
        recon_batches = []
        for i, (batch, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            batch = batch.to(device)
            bsz = batch.size(0)
            flat = batch.view(bsz, -1)
            orig_batches.append(flat)
            with torch.no_grad():
                gen = lambda seed: torch.Generator(device=device).manual_seed(seed)
                if "vanilla" in arch_flag:
                    z = torch.randn(
                        bsz, model.decode_dim, generator=gen(0), device=device
                    )
                    generated = model.decode(z)
                else:
                    z = torch.randn(bsz, latent_dim, generator=gen(0), device=device)
                    c = model.conceptualize(z, concepts_input, arch_flag=arch_flag)
                    generated = model.decode(c)
            recon_batches.append(generated.view(bsz, -1))

        orig_flat = torch.cat(orig_batches, dim=0)
        recon_flat = torch.cat(recon_batches, dim=0)
        return orig_flat, recon_flat

    def _compute_ot(concepts_input: str, data_loader, model, num_batches=20):
        orig, recon = _sub_recon_flat(concepts_input, data_loader, model, num_batches)
        return sliced_wasserstein_distance(orig, recon).item()

    def _compute_ot_gen(concepts_input: str, data_loader, model, num_batches=20):
        orig, recon = _sub_recon_flat_gen(
            concepts_input, data_loader, model, num_batches
        )
        return sliced_wasserstein_distance(orig, recon).item()

    def compute_metrics(concepts_input, data_loader, model, path):
        metrics_list = []

        for concept in concepts_input:
            concept_loader = DataLoader(
                [(data, label) for data, label in data_loader if label == concept],
                batch_size=data_loader.batch_size,
            )

            concept_as_list = [concept]
            ot_gen_value = _compute_ot_gen(concept_as_list, concept_loader, model)

            metrics_list.append(
                {
                    "concept": concept,
                    "ot_gen": ot_gen_value,
                }
            )

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(f"{path}/metrics.csv", index=False)

    def compute_ood_metrics(concepts_input, data_loader, model, path):
        metrics_list = []

        for concept in concepts_input:
            concept_loader = DataLoader(
                [(data, label) for data, label in data_loader if label == concept],
                batch_size=data_loader.batch_size,
            )
            split_concept = concept.split(split_char)
            ot_gen_value = _compute_ot_gen(split_concept, concept_loader, model)

            metrics_list.append(
                {
                    "concept": concept,
                    "ood_ot_gen": ot_gen_value,
                }
            )

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(f"{path}/ood_metrics.csv", index=False)

    def compute_ood_val_elbo(concepts_input, data_loader, model, path):
        model.eval()
        metrics_list = []
        for concept in concepts_input:
            concept_loader = DataLoader(
                [(data, label) for data, label in data_loader if label == concept],
                batch_size=data_loader.batch_size,
            )
            split_concept = concept.split(split_char)
            val_lb = 0.0
            normalizer = 0.0

            with torch.no_grad():
                for data, label in concept_loader:
                    data = data.to(device)
                    recon_batch, mu, log_var = model(
                        data, split_concept, arch_flag=arch_flag
                    )

                    eps = 1e-7
                    recon_batch = recon_batch.to(device=device, dtype=torch.float32)
                    finite_mask = torch.isfinite(recon_batch)
                    if not finite_mask.all():
                        recon_batch = torch.where(
                            finite_mask, recon_batch, torch.tensor(0.5, device=device)
                        )
                    recon_batch = recon_batch.clamp(eps, 1.0 - eps)
                    if not torch.isfinite(recon_batch).all():
                        print("Skipping bad val batch")
                        continue

                    lb, _ = val_loss_function(recon_batch, data, mu, log_var)
                    val_lb += lb.item()
                    normalizer += len(data)
            val_lb /= normalizer

            metrics_list.append(
                {
                    "concept": concept,
                    "ood_val_elbo": val_lb,
                }
            )

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(f"{path}/ood_val_elbo.csv", index=False)

    def train():
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc="training epoch...", unit="batch", leave=False)
        optimizer.zero_grad()
        for batch_idx, (data, label) in enumerate(pbar, 1):
            data = data.to(device)
            recon_batch, mu, log_var = model(data, [label], arch_flag=arch_flag)

            eps = 1e-7
            recon_batch = recon_batch.to(device=data.device, dtype=torch.float32)
            finite_mask = torch.isfinite(recon_batch)
            if not finite_mask.all():
                recon_batch = torch.where(
                    finite_mask, recon_batch, torch.tensor(0.5, device=data.device)
                )
            recon_batch = recon_batch.clamp(eps, 1.0 - eps)
            if not torch.isfinite(recon_batch).all():
                print("Skipping bad train batch")
                continue

            if "vanilla" in arch_flag:
                causal_grouped_batch = causal_pooled_batch = causal_unpooled_batch = (
                    None
                )
            else:
                causal_grouped_batch = model.causal_layer.grouped
                causal_pooled_batch = model.causal_layer.pooled
                causal_unpooled_batch = model.causal_layer.unpooled
            loss = loss_function(
                recon_batch,
                data,
                mu,
                log_var,
                causal_grouped_batch,
                causal_unpooled_batch,
                causal_pooled_batch,
            )
            (loss / microbatches).backward()
            train_loss += loss.item()

            if batch_idx % microbatches == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Only clamp weights if they are not frozen
            if "vanilla" not in arch_flag:
                if model.causal_layer.obs_weight.requires_grad:
                    model.causal_layer.obs_weight.data.clamp_(0, 1)
                if model.causal_layer.ivn_weight.requires_grad:
                    model.causal_layer.ivn_weight.data.clamp_(0, 1)

        return train_loss / len(train_loader.dataset)

    def validate():
        model.eval()
        val_lb = 0.0
        val_bce = 0.0

        with torch.no_grad():
            for data, label in val_loader:
                data = data.to(device)
                recon_batch, mu, log_var = model(data, [label], arch_flag=arch_flag)

                eps = 1e-7
                recon_batch = recon_batch.to(device=device, dtype=torch.float32)
                finite_mask = torch.isfinite(recon_batch)
                if not finite_mask.all():
                    recon_batch = torch.where(
                        finite_mask, recon_batch, torch.tensor(0.5, device=device)
                    )
                recon_batch = recon_batch.clamp(eps, 1.0 - eps)
                if not torch.isfinite(recon_batch).all():
                    print("Skipping bad val batch")
                    continue

                lb, bce = val_loss_function(recon_batch, data, mu, log_var)
                val_lb += lb.item()
                val_bce += bce.item()

        n = len(val_loader.dataset)
        return val_lb / n, val_bce / n

    def train_model():
        losses = []
        val = {"elbo": [], "bce": []}

        # Early stopping variables
        best_val_elbo = float("inf")  # Lower is better for ELBO
        best_model_state = None
        best_optimizer_state = None
        best_epoch = 0
        patience_counter = 0

        unfreeze_epoch = (
            int(0.2 * epochs) if arch_flag == "fine-tune-concept-unfreeze" else -1
        )
        unfrozen = False

        pbar = tqdm(range(epochs), desc="Training...", unit="epoch")

        for epoch in pbar:
            if (
                arch_flag == "fine-tune-concept-unfreeze"
                and epoch == unfreeze_epoch
                and not unfrozen
            ):
                print(f"\n{'=' * 60}")
                print(
                    f"Epoch {epoch}/{epochs}: Unfreezing all weights for full fine-tuning"
                )
                print(f"{'=' * 60}")

                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        print(f"Unfroze: {name}")

                print("All weights now trainable (optimizer state preserved)")
                print(f"{'=' * 60}\n")
                unfrozen = True

            loss = train()
            losses.append(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

            (elbo, bce) = validate()
            val["elbo"].append(elbo)
            val["bce"].append(bce)

            should_track_early_stopping = True
            if arch_flag == "fine-tune-concept-unfreeze" and not unfrozen:
                should_track_early_stopping = False
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "val_elbo": f"{elbo:.4f}",
                        "status": "pre-unfreeze (ES disabled)",
                    }
                )

            if should_track_early_stopping:
                if elbo < best_val_elbo - early_stopping_min_delta:
                    best_val_elbo = elbo
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model state in memory
                    best_model_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    best_optimizer_state = {
                        k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                        for k, v in optimizer.state_dict().items()
                    }
                    pbar.set_postfix(
                        {
                            "loss": f"{loss:.4f}",
                            "val_elbo": f"{elbo:.4f}",
                            "status": "improved",
                        }
                    )
                else:
                    patience_counter += 1
                    pbar.set_postfix(
                        {
                            "loss": f"{loss:.4f}",
                            "val_elbo": f"{elbo:.4f}",
                            "patience": f"{patience_counter}/{early_stopping_patience}",
                        }
                    )

                    if patience_counter >= early_stopping_patience:
                        print(f"\n{'=' * 60}")
                        print(f"Early stopping triggered at epoch {epoch}")
                        print(
                            f"Best validation ELBO: {best_val_elbo:.4f} at epoch {best_epoch}"
                        )
                        print(f"{'=' * 60}\n")
                        break

            log_epoch(epoch, output_path=f"{save_dir}LOG")

            if epoch % epochs_per_checkpoint == 0 and epoch > 0:
                dir_path = f"{save_dir}vae_checkpoints"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "losses": losses,
                        "val_elbo": val["elbo"],
                        "val_bce": val["bce"],
                    },
                    f"{dir_path}/epoch_{epoch}.pth",
                )
                plot_training_loss(losses, epoch=epoch, subdir="vae_checkpoints")
                plot_validation_loss(
                    val["elbo"], val["bce"], epoch=epoch, subdir="vae_checkpoints"
                )
                plot_reconstructions(epoch=epoch, subdir="vae_checkpoints")
                plot_random_samples(epoch=epoch)
                plot_sparsity()

        # Reload best model before final saving
        if best_model_state is not None:
            print(f"\n{'=' * 60}")
            print(
                f"Reloading best model from epoch {best_epoch} with validation ELBO: {best_val_elbo:.4f}"
            )
            print(f"{'=' * 60}\n")
            # Move best model state back to device
            model.load_state_dict(
                {k: v.to(device) for k, v in best_model_state.items()}
            )
            # Move optimizer state back to device if needed
            optimizer.load_state_dict(
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in best_optimizer_state.items()
                }
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "val_elbo": val["elbo"],
                "val_bce": val["bce"],
                "best_epoch": best_epoch,
                "best_val_elbo": best_val_elbo,
            },
            f"{save_dir}vae.pth",
        )

    # Training/loading toggle
    if not args.eval_only:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        init_log(output_path=f"{save_dir}LOG")
        create_readme(params, notes=EXPT_NOTES, output_path=f"{save_dir}README.md")

        save_model_architecture(
            model,
            input_size=(1, img_channels, img_height, img_width),
            filename=f"{save_dir}model_architecture.txt",
        )
        with open(f"{save_dir}model_architecture.txt", "a") as f:
            print("\n\n-TORCHINFO SUMMARY-", file=f)
            print(summary(model, verbose=2), file=f)
        train_model()
        if args.snakemake:
            return

    # Load trained model
    model = VAE(
        arch_id=arch_id,
        arch_flag=arch_flag,
        latent_dim=latent_dim,
        eps_dim=eps_dim,
        eps_in_width=eps_in_width,
        eps_out_width=eps_out_width,
        eps_depth=eps_depth,
        c_dim=c_dim,
        c_width=c_width,
        concepts=concepts,
        device=device,
        VAE_ARCHITECTURES=VAE_ARCHITECTURES,
    ).to(device)
    checkpoint = torch.load(f"{save_dir}vae.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    losses = checkpoint["losses"]
    val_elbo = checkpoint["val_elbo"]
    val_bce = checkpoint["val_bce"]

    df = pd.DataFrame({"val_elbo": val_elbo})
    df.to_csv(f"{save_dir}val_elbo.csv", index=False)

    print(f"\ngenerating plots in {save_dir}...")
    plot_training_loss(losses)
    plot_validation_loss(val_elbo, val_bce)
    plot_reconstructions()
    plot_random_samples()
    plot_combo_random_samples()
    plot_sparsity()

    if args.dataset == "quad" or args.dataset == "quad_causal":
        single = ["obs", "quad1", "quad2", "quad3", "quad4", "size", "orientation"]
        double = [
            "quad1_quad2",
            "quad1_quad3",
            "quad1_quad4",
            "quad1_orientation",
            "quad1_size",
            "quad2_quad3",
            "quad2_quad4",
            "quad2_size",
            "quad2_orientation",
        ]
        if arch_flag == "vanilla-obs":
            h_loader = get_loader(h_datasets, batch_size)
        compute_metrics(single, h_loader, model, save_dir)
        compute_ood_metrics(double, h_loader, model, save_dir)
        compute_ood_val_elbo(double, h_loader, model, save_dir)
    elif args.dataset == "mnist":
        if arch_flag == "vanilla-obs":
            h_loader = get_loader(h_datasets, batch_size)
        compute_metrics(concepts, h_loader, model, save_dir)
        # save empty OOD files (since mnist has no OOD), since snakemake expects it
        open(os.path.join(save_dir, "ood_metrics.csv"), "w").close()
        open(os.path.join(save_dir, "ood_val_elbo.csv"), "w").close()
    else:
        if arch_flag == "vanilla-obs":
            train_loader = get_loader(train_datasets, batch_size)
        compute_metrics(concepts, train_loader, model, save_dir)

    print("...done!\n")


if __name__ == "__main__":
    main()
