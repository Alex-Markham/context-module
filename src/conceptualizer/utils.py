import os
import random
import string
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset


def init_log(output_path=None):
    if output_path is None:
        output_path = "LOG"
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )  # Ensure directory exists

    with open(output_path, "w") as f:
        f.write(f"Log started at {datetime.now().strftime('%a %b %d %H:%M:%S')}\n")


def log_epoch(epoch, output_path=None):
    if output_path is None:
        output_path = "LOG"

    with open(output_path, "a") as f:
        f.write(
            f"{datetime.now().strftime('%a %b %d %H:%M:%S')} Epoch {epoch} finished\n"
        )


def generate_tag_path(tag, length=5, DIR="results"):
    # Generate random string of capital letters and numbers
    chars = string.ascii_uppercase + string.digits
    random_str = "".join(random.choices(chars, k=length))

    # Format and return the path
    return f"{DIR}-{tag}-{random_str}/"


def save_model_architecture(
    model, input_size=(1, 1, 28, 28), filename="model_architecture.txt"
):
    """
    Save the model's encoder and decoder architecture to a file.

    Args:
        model: The VAE model
        input_size: Input tensor size (batch_size, channels, height, width)
        filename: Output file name
    """

    def get_output_shape(model, input_size):
        with torch.no_grad():
            # Create a batch size of 4 to avoid BatchNorm issues
            if isinstance(input_size, tuple):
                x = torch.rand((4,) + input_size[1:])
            else:
                x = torch.rand((4, input_size))

            for layer in model:
                try:
                    x = layer(x)
                    if isinstance(
                        layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Flatten)
                    ):
                        yield f"{type(layer).__name__}: {tuple(x.shape)}"
                except Exception as e:
                    yield f"{type(layer).__name__}: Shape computation failed - {str(e)}"

    with open(filename, "w") as f:
        f.write("-ENCODER/DECODER SUMMARY-\n")

        # Write encoder architecture
        f.write("=" * 50 + "\n")
        f.write("Encoder Architecture:\n")
        f.write("=" * 50 + "\n")

        # Get encoder shapes
        f.write("\nLayer Output Shapes:\n")
        for shape_info in get_output_shape(model.encoder, input_size):
            f.write(f"{shape_info}\n")

        # Write encoder parameter summary
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        f.write(f"\nTotal encoder parameters: {encoder_params:,}\n")

        # Write decoder architecture
        f.write("\n" + "=" * 50 + "\n")
        f.write("Decoder Architecture:\n")
        f.write("=" * 50 + "\n")

        # Get decoder shapes using batch size of 4
        f.write("\nLayer Output Shapes:\n")
        for shape_info in get_output_shape(model.decoder, (4, model.decode_dim)):
            f.write(f"{shape_info}\n")

        # Write decoder parameter summary
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        f.write(f"\nTotal decoder parameters: {decoder_params:,}\n")

        # Write total model parameters
        total_params = sum(p.numel() for p in model.parameters())
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Total model parameters: {total_params:,}\n")


def samples_per_dataset(datasets_tuple):
    """
    Count the number of samples in each TensorDataset within a tuple.

    Args:
        datasets_tuple: A tuple containing TensorDataset objects

    Returns:
        A list with the number of samples in each dataset
    """
    sample_counts = []

    for dataset in datasets_tuple:
        num_samples = len(dataset)
        # if hasattr(dataset, 'tensors') and len(dataset.tensors) > 0:
        #     num_samples = dataset.tensors[0].shape[0]

        sample_counts.append(num_samples)

    return sample_counts


def summarize_run(concepts, indices_to_colour, nsamples):
    """
    Generate a line-by-line summary of concepts, sample sizes, and colorization
    with columns aligned horizontally.

    Parameters:
    - concepts: tuple of concept names
    - indices_to_colour: list of indices that have been colorized
    - nsamples: list of sample counts for each concept

    Returns:
    - formatted_list: string with each concept on a new line, aligned in columns
    """
    # Find the maximum length of concept names for proper alignment
    max_concept_len = max(len(concept) for concept in concepts)

    # Find the maximum length of sample numbers for proper alignment
    max_sample_len = max(len(str(sample)) for sample in nsamples)

    lines = []

    for i, concept in enumerate(concepts):
        # Determine if this concept is colorized
        is_colorized = i in indices_to_colour
        colorized_marker = "C" if is_colorized else " "

        # Format the line with proper alignment
        # Left-align concept name, right-align sample count, align colorization marker
        line = f"'{concept}'{' ' * (max_concept_len - len(concept) + 2)} {str(nsamples[i]).rjust(max_sample_len)}  {colorized_marker}"
        lines.append(line)

    # Join all lines with newlines
    return "\n".join(lines)


def create_readme(hyperparams, data=None, notes=None, output_path="README.md"):
    """
    Creates a README file documenting the hyperparameters used in the experiment.

    Args:
        hyperparams (dict): Dictionary containing the hyperparameters
        output_path (str): Path where the README file should be saved
    """
    with open(output_path, "w") as f:
        # Write header with timestamp
        f.write("# Experiment Configuration\n")
        f.write(f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Notes: {notes}\n\n")

        # Write data configuration
        f.write("## Data Configuration\n")
        for param, value in hyperparams["data"].items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")

        # Write experiment settings
        f.write("## Experiment Settings\n")
        for param, value in hyperparams["experiment"].items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")

        # Write model dimensions
        f.write("## Model Architecture\n")
        for param, value in hyperparams["dimensions"].items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")

        # Write training parameters
        f.write("## Training Parameters\n")
        for param, value in hyperparams["training"].items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")

        if data is not None:
            # Add TensorDataset summary section
            f.write("## Dataset Summary\n")

            # Function to summarize tensor datasets
            def summarize_tensor_datasets(datasets_tuple):
                summaries = []

                for i, dataset in enumerate(datasets_tuple):
                    if not isinstance(dataset, TensorDataset):
                        raise TypeError(f"Item {i} is not a TensorDataset")

                    dataset_summary = {"dataset_index": i, "tensors": []}

                    # Get total number of samples
                    num_samples = len(dataset)
                    dataset_summary["num_samples"] = num_samples

                    # Process each tensor in the dataset
                    for j, tensor in enumerate(dataset.tensors):
                        tensor_info = {
                            "tensor_index": j,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                        }

                        # Determine if this is likely a feature tensor or target tensor
                        tensor_role = "features" if j == 0 else "target"
                        tensor_info["likely_role"] = tensor_role

                        # Calculate dimensions details
                        if len(tensor.shape) == 1:
                            tensor_info["dimensions_type"] = "1D"
                            tensor_info["num_features"] = 1
                            tensor_info["num_channels"] = 1
                        elif len(tensor.shape) == 2:
                            tensor_info["dimensions_type"] = "2D"
                            tensor_info["num_features"] = tensor.shape[1]
                            tensor_info["num_channels"] = 1
                        elif len(tensor.shape) == 3:
                            tensor_info["dimensions_type"] = "3D"
                            tensor_info["num_features"] = tensor.shape[2]
                            tensor_info["num_channels"] = tensor.shape[1]
                        elif len(tensor.shape) == 4:
                            tensor_info["dimensions_type"] = "4D"
                            tensor_info["num_channels"] = tensor.shape[1]
                            tensor_info["height"] = tensor.shape[2]
                            tensor_info["width"] = tensor.shape[3]
                            tensor_info["num_features"] = (
                                tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
                            )
                        else:
                            tensor_info["dimensions_type"] = f"{len(tensor.shape)}D"
                            tensor_info["num_features"] = (
                                "Complex - multiple dimensions"
                            )
                            tensor_info["num_channels"] = (
                                "Complex - multiple dimensions"
                            )

                        dataset_summary["tensors"].append(tensor_info)

                    summaries.append(dataset_summary)

                return summaries

            # Generate summaries for the dataset tuple
            summaries = summarize_tensor_datasets(data)

            # Write the summaries to the README
            for dataset_summary in summaries:
                dataset_idx = dataset_summary["dataset_index"]
                num_samples = dataset_summary["num_samples"]

                f.write(
                    f"\t### Dataset {hyperparams['data']['concepts'][dataset_idx]}\n"
                )
                f.write(f"\t- Number of samples: {num_samples}\n")

                for tensor_info in dataset_summary["tensors"]:
                    tensor_idx = tensor_info["tensor_index"]
                    shape = tensor_info["shape"]
                    dtype = tensor_info["dtype"]
                    likely_role = tensor_info["likely_role"]

                    # f.write(f"#### Tensor {tensor_idx} ({likely_role})\n")
                    f.write(f"\t- Shape: {shape}\n")
                    f.write(f"\t- Dtype: {dtype}\n")

                    if "num_features" in tensor_info:
                        f.write(
                            f"\t- Number of features: {tensor_info['num_features']}\n"
                        )

                    if "num_channels" in tensor_info:
                        f.write(
                            f"\t- Number of channels: {tensor_info['num_channels']}\n"
                        )

                    if "height" in tensor_info and "width" in tensor_info:
                        f.write(
                            f"\t- Height: {tensor_info['height']}, Width: {tensor_info['width']}\n"
                        )

                    f.write("\n")

                f.write("\n")

        # Write raw hyperparameters dictionary with actual values
        f.write("{\n")
        f.write("    # Architecture dimensions\n")
        f.write("    'dimensions': {\n")
        f.write(
            f"        'eps_dim': {hyperparams['dimensions']['eps_dim']},  # eps_dim\n"
        )
        f.write(
            f"        'eps_in_width': {hyperparams['dimensions']['eps_in_width']},  # eps_in_width\n"
        )
        f.write(
            f"        'eps_out_width': {hyperparams['dimensions']['eps_out_width']},  # eps_out_width\n"
        )
        f.write(
            f"        'eps_depth': {hyperparams['dimensions']['eps_depth']},  # eps_depth\n"
        )
        f.write(f"        'c_dim': {hyperparams['dimensions']['c_dim']},  # c_dim\n")
        f.write(
            f"        'c_width': {hyperparams['dimensions']['c_width']},  # c_width\n"
        )
        f.write(
            f"        'latent_dim': {hyperparams['dimensions']['latent_dim']},  # latent_dim\n"
        )
        f.write(
            f"        'hidden_dim': {hyperparams['dimensions']['hidden_dim']},  # hidden_dim\n"
        )
        f.write("    },\n")
        f.write("    \n")
        f.write("    # Training parameters\n")
        f.write("    'training': {\n")
        f.write(
            f"        'batch_size': {hyperparams['training']['batch_size']},  # batch_size\n"
        )
        f.write(
            f"        'learning_rate': {hyperparams['training']['learning_rate']},  # learning_rate\n"
        )
        f.write(f"        'epochs': {hyperparams['training']['epochs']},  # epochs\n")
        f.write("    },\n")
        f.write("    \n")
        f.write("    # Checkpointing and experiment settings\n")
        f.write("    'experiment': {\n")
        f.write(
            f"        'epochs_per_checkpoint': {hyperparams['experiment']['epochs_per_checkpoint']},  # epochs_per_checkpoint\n"
        )
        f.write(
            f"        'expt_tag': '{hyperparams['experiment']['expt_tag']}',  # expt_tag\n"
        )
        f.write("    },\n")
        f.write("    \n")
        f.write("    # Data configuration\n")
        f.write("    'data': {\n")
        f.write(
            f"        'concepts': '{hyperparams['data']['concepts']}'  # concepts\n"
        )
        f.write(
            f"        'colorized_contexts': '{hyperparams['data']['colorized_contexts']}'  # colorized_contexts\n"
        )
        f.write(
            f"        'extra_colour_context': '{hyperparams['data']['extra_colour_context']}'  # extra_colour_context\n"
        )
        f.write(
            f"        'nsamples': '{hyperparams['data']['nsamples']}'  # nsamples\n"
        )
        f.write("    }\n")
        f.write("}\n")


quad_concept_learning = [
    "obs",
    "quad1",
    "quad2",
    "quad3",
    "quad4",
    "size",
    "orientation",
]
quad_composition = [
    "quad1_quad2",
    "quad1_quad3",
    "quad1_quad4",
    "quad1_size",
    "quad1_orientation",
    "quad2_quad3",
    "quad2_quad4",
    "quad2_size",
    "quad2_orientation",
]


class InterfaceDisentDataset(Dataset):
    """Wraps a dataset to return tensors directly instead of tuples"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # If it's a tuple from TensorDataset, extract the first element
        if isinstance(item, tuple):
            return item[0]
        return item
