import os
import random
import shutil
import tarfile
import zipfile
from pathlib import Path

from conceptualizer.utils import quad_composition, quad_concept_learning
from zenodo_get import download

from quad_helper import gen_holdout, gen_train

dataset_name = str(snakemake.wildcards.dataset)
root = "data/"

####################################################################################################################
if dataset_name == "quad":
    download(
        record_or_doi="19346614",
        output_dir=root,
        file_glob="quad.zip",
    )
    zip_file_path = os.path.join(root, "quad.zip")
    extraction_directory = root

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        # Iterate through the contents of the zip file
        for file_info in zip_ref.infolist():
            # Check if the file ends with .npz
            if file_info.filename.endswith(".npz"):
                # Extract the .npz file to the specified directory
                zip_ref.extract(file_info, extraction_directory)

####################################################################################################################
elif dataset_name == "quad_causal":
    # generate data
    strength = 1.0
    expt = dataset_name
    Path(root + expt).mkdir(parents=True, exist_ok=True)

    n = 50000
    batch_size = 1000

    contexts = quad_concept_learning
    gen_train(root, expt, contexts, n, batch_size, strength)

    n = 10000
    contexts += quad_composition
    gen_holdout(root, expt, contexts, n, batch_size, strength)


####################################################################################################################
elif dataset_name == "mnist":
    download(
        record_or_doi="19346614",
        output_dir=root,
        file_glob="normalized_mnist_concepts.tar.gz",
    )
    tar_file_path = os.path.join(root, "normalized_mnist_concepts.tar.gz")
    extraction_directory = root

    # extract the archive (it contains normalized_mnist_concepts/...)
    with tarfile.open(tar_file_path, "r:gz") as tar:
        tar.extractall(extraction_directory)  # creates normalized_mnist_concepts/

    src_dir = os.path.join(root, "normalized_mnist_concepts")
    dst_dir = os.path.join(root, "mnist")
    os.makedirs(dst_dir, exist_ok=True)

    random.seed(131223)

    for fname in os.listdir(src_dir):
        if not fname.endswith(".csv"):
            continue

        src_path = os.path.join(src_dir, fname)

        # read full CSV (header + rows)
        with open(src_path, "r") as f:
            lines = f.readlines()

        if not lines:
            continue

        header = lines[0]
        rows = lines[1:]

        random.shuffle(rows)
        split_idx = int(len(rows) * 0.7)
        train_rows = rows[:split_idx]
        val_rows = rows[split_idx:]

        stem, ext = os.path.splitext(fname)  # e.g. normalized__obs + .csv

        train_path = os.path.join(dst_dir, f"{stem}{ext}")
        val_path = os.path.join(dst_dir, f"holdout_{stem}{ext}")

        # write train file
        with open(train_path, "w") as f_tr:
            f_tr.write(header)
            f_tr.writelines(train_rows)

        # write validation/holdout file
        with open(val_path, "w") as f_val:
            f_val.write(header)
            f_val.writelines(val_rows)

    shutil.rmtree(src_dir)

####################################################################################################################
elif dataset_name == "3dident":
    download(
        record_or_doi="19346613",
        output_dir=root,
        file_glob="3dident.zip",
    )
    zip_file_path = os.path.join(root, "3dident.zip")
    with zipfile.ZipFile(zip_file_path, "r") as z:
        z.extractall(root)

    source_dirs = {
        "obs": root + "3DIdent_Concepts/obs/images",
        "bg": root + "3DIdent_Concepts/bg/images",
        "obj": root + "3DIdent_Concepts/obj/images",
        "sl": root + "3DIdent_Concepts/sl/images",
        "bg-obj": root + "3DIdent_Concepts_ood/dataset/bg-obj/images",
        "bg-sl": root + "3DIdent_Concepts_ood/dataset/bg-sl/images",
        "obj-sl": root + "3DIdent_Concepts_ood/dataset/obj-sl/images",
    }

    target_dir = "3dident"
    root_path = Path(root)
    target_path = root_path / target_dir
    target_path.mkdir(exist_ok=True, parents=True)

    random.seed(131223)

    for subdir_name, source_path_str in source_dirs.items():
        source_path = Path(source_path_str)
        images = sorted(list(source_path.glob("*.png")))
        random.shuffle(images)
        split_idx = int(len(images) * 0.7)
        train_images = images[:split_idx]
        holdout_images = images[split_idx:]

        # Train (concepts only)
        if subdir_name not in ["bg-obj", "bg-sl", "obj-sl"]:
            train_subdir = target_path / subdir_name / "images"
            train_subdir.mkdir(parents=True, exist_ok=True)
            for img in train_images:
                link_path = train_subdir / img.name
                if not link_path.exists():
                    os.symlink(img, link_path)

        # Holdout (all)
        holdout_subdir = target_path / f"holdout_{subdir_name}" / "images"
        holdout_subdir.mkdir(parents=True, exist_ok=True)
        for img in holdout_images:
            link_path = holdout_subdir / img.name
            if not link_path.exists():
                os.symlink(img, link_path)

####################################################################################################################
else:
    # not implemented yet
    pass
