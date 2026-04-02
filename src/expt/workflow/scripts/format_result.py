import numpy as np
import pandas as pd

smi = snakemake.input
smw = snakemake.wildcards


# reconstruction
def bpd(raw_loss):
    if smw["dataset"].startswith("quad"):
        num_pixels = 64 * 64
    elif smw["dataset"] == "mnist":
        num_pixels = 28 * 28
    else:
        raise ValueError("Not implemented yet.")
    return raw_loss / (num_pixels * np.log(2))


val_elbo = pd.read_csv(smi["reconstruction"])["val_elbo"].values
min_bpd = bpd(np.min(val_elbo))
final_bpd = bpd(val_elbo[-1])
if smw["dataset"] == "mnist":
    ood_bpd = None
else:
    ood_bpd = bpd(
        pd.read_csv(smi["ood_reconstruction"], index_col=0)
        .squeeze()
        .astype(float)
        .mean()
    )
reconstruction = {
    "min_val_elbo_bpd": [min_bpd],
    "final_val_elbo_bpd": [final_bpd],
    "ood_val_elbo_bpd": [ood_bpd],
}


concept_learning = pd.read_csv(smi["concept_learning"], index_col=0).squeeze().to_dict()


if smw["dataset"] == "mnist":
    composition = {}
else:
    composition = pd.read_csv(smi["composition"], index_col=0).squeeze().to_dict()


# save outputs
result = pd.DataFrame(dict(smw) | reconstruction | concept_learning | composition)
result.to_csv(snakemake.output[0], index=False)
