import os
import warnings

import numpy as np
import pandas as pd
import sempler
from matplotlib.path import Path
from PIL import Image
from scipy.stats import norm

warnings.filterwarnings("ignore")
emp_means = np.zeros(8)
emp_stds = np.ones(8)


def sample_latents(
    context,
    n_samples: int = int(1e4),
    seed: int = 131223,
    strength: float = 1.0,
) -> np.ndarray:
    """Generate observational/interventional samples from linear Gaussian SCM."""
    labels = [
        "quad1",
        "quad2",
        "quad3",
        "quad4",
        "size",
        "color",
        "shape",
        "orientation",
    ]
    W = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.23, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.56, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.34, 0.0, -0.72, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.95, 0.0, 0.0, 0.0, 1.67, 0.0, 0.0, 0.0],
            [0.0, 0.58, 0.0, 0.0, 0.0, -1.34, 0.0, 0.0],
            [1.12, 0.0, 0.0, 0.0, -0.81, 0.0, 1.45, 0.0],
        ],
        dtype=np.float32,
    )

    obs_means = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    obs_vars = np.array(
        [
            1.3855493,
            1.16426732,
            1.00502177,
            1.95646855,
            1.27482347,
            1.42,
            1.31,
            1.18,
        ]
    )

    int_means = (
        np.array(
            [
                1.16920556,
                1.47496736,
                -1.60596209,
                -1.96578519,
                1.0117071,
                -0.85,
                1.23,
                -0.67,
            ]
        )
        * strength
    )
    int_vars = np.array(
        [
            1.94878138,
            1.35886756,
            1.57343313,
            1.1021785,
            1.30511067,
            1.52,
            1.41,
            1.26,
        ]
    )

    # Create SCM with observational variances
    lganm = sempler.LGANM(
        W=W, means=obs_means, variances=obs_vars, random_state=4719820817
    )
    if context == "obs":
        sample = lganm.sample(n_samples, random_state=seed)
        global emp_means, emp_stds
        emp_means = np.mean(sample, axis=0, keepdims=True)
        emp_stds = np.std(sample, axis=0, keepdims=True)
    else:
        if "_" in context:
            label_1, label_2 = context.split("_")
            node_1, node_2 = labels.index(label_1), labels.index(label_2)
            joint_interventions = {
                node_1: (int_means[node_1], int_vars[node_1]),
                node_2: (int_means[node_2], int_vars[node_2]),
            }
        else:
            node_1 = labels.index(context)
            joint_interventions = {
                node_1: (int_means[node_1], int_vars[node_1]),
            }
        sample = lganm.sample(
            n_samples, do_interventions=joint_interventions, random_state=seed
        )
    sample = (sample - emp_means) / emp_stds

    return pd.DataFrame(sample, columns=labels)


def latent_to_image(latents):
    n = latents.shape[0]
    images = np.zeros((n, 64, 64, 3), dtype=np.uint8)

    size = np.clip(latents[:, 4], 0.2, 1.0)

    images[:, 32:64, 0:32] = (_scalar_to_rgb_vectorized(latents[:, 0]) * 255).astype(
        np.uint8
    )[:, None, None, :]
    images[:, 32:64, 32:64] = (_scalar_to_rgb_vectorized(latents[:, 1]) * 255).astype(
        np.uint8
    )[:, None, None, :]
    images[:, 0:32, 0:32] = (_scalar_to_rgb_vectorized(latents[:, 2]) * 255).astype(
        np.uint8
    )[:, None, None, :]
    images[:, 0:32, 32:64] = (_scalar_to_rgb_vectorized(latents[:, 3]) * 255).astype(
        np.uint8
    )[:, None, None, :]

    shapes = latents[:, 6].astype(int)
    colors = latents[:, 5]
    orientations = latents[:, 7]

    for i in range(n):
        _draw_shape_fast(images[i], shapes[i], size[i], orientations[i], colors[i])

    return [images[i] for i in range(n)]


def _scalar_to_rgb_vectorized(values):
    values = np.clip(values, 0, 1)
    colors = np.array(
        [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)]
    )

    seg_size = 1.0 / (len(colors) - 1)
    seg = np.floor(values / seg_size).astype(int)
    seg = np.clip(seg, 0, len(colors) - 2)

    t = (values - seg * seg_size) / seg_size
    t = t[:, None]

    return colors[seg] * (1 - t) + colors[seg + 1] * t


def _scalar_to_rgb(value):
    value = max(0, min(1, value))
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)]
    seg_size = 1.0 / (len(colors) - 1)
    seg = int(value / seg_size)
    if seg == len(colors) - 1:
        return np.array(colors[-1])
    t = (value - seg * seg_size) / seg_size
    return np.array(
        [colors[seg][j] + t * (colors[seg + 1][j] - colors[seg][j]) for j in range(3)]
    )


def _draw_shape_fast(img, shape_type, size, orientation, color_val):
    actual_size = size * 48
    angle_rad = orientation * 2 * np.pi
    rgb = (_scalar_to_rgb(color_val) * 255).astype(np.uint8)

    cx, cy = 32, 32
    y, x = np.ogrid[:64, :64]

    if shape_type == 0:
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= (actual_size / 2) ** 2
    elif shape_type == 1:
        dx, dy = x - cx, y - cy
        x_rot = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
        y_rot = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)
        mask = (np.abs(x_rot) <= actual_size / 2) & (np.abs(y_rot) <= actual_size / 2)
    elif shape_type == 2:
        radius = actual_size / 2
        points = np.array(
            [
                [
                    cx + radius * np.cos(angle_rad + i * 2 * np.pi / 3),
                    cy + radius * np.sin(angle_rad + i * 2 * np.pi / 3),
                ]
                for i in range(3)
            ]
        )
        yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
        mask = Path(points).contains_points(coords).reshape(64, 64)
    elif shape_type == 3:
        dx, dy = x - cx, y - cy
        x_rot = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
        y_rot = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)
        mask = (x_rot / actual_size) ** 2 + (y_rot / (actual_size / 2)) ** 2 <= 0.25
    else:
        mask = np.zeros((64, 64), dtype=bool)

    img[mask] = rgb


def save_images(images, folder_path, start_index):
    os.makedirs(folder_path, exist_ok=True)
    for i, img_array in enumerate(images):
        img = Image.fromarray(img_array)
        img.save(os.path.join(folder_path, f"{start_index + i}.png"), format="PNG")


def process_gaussian_latents(latents):
    # Map all dims (except shape) to [0,1] via standard normal CDF
    mapped = norm.cdf(latents)
    # Size in [0.2, 1]
    mapped[:, 4] = 0.2 + 0.8 * mapped[:, 4]
    # Shape: map [0,1] -> {0,1,2,3}
    mapped[:, 6] = np.clip(np.round(mapped[:, 6] * 3), 0, 3)
    # Ensure orientation, color, quadrants in [0,1]
    mapped[:, :4] = np.clip(mapped[:, :4], 0, 1)
    mapped[:, 5] = np.clip(mapped[:, 5], 0, 1)
    mapped[:, 7] = np.clip(mapped[:, 7], 0, 1)
    return mapped


def gen_train(root, expt, contexts, n, batch_size, strength):
    for context in contexts:
        context_dir = root + expt + "/train_" + context
        images_dir = os.path.join(context_dir, "images_64")
        os.makedirs(images_dir, exist_ok=True)

        inputs = sample_latents(context, n, strength=strength)
        inputs.to_csv(os.path.join(context_dir, f"{context}.csv"), index=False)

        for start in range(0, n, batch_size):
            batch_input = inputs.iloc[start : start + batch_size]
            processed = process_gaussian_latents(batch_input)
            images = latent_to_image(processed)
            save_images(images, images_dir, start)


def gen_holdout(root, expt, contexts, n, batch_size, strength):
    for context in contexts:
        context_dir = root + expt + "/holdout_" + context
        images_dir = os.path.join(context_dir, "images_64")
        os.makedirs(images_dir, exist_ok=True)

        inputs = sample_latents(context, n, strength=strength)
        inputs.to_csv(os.path.join(context_dir, f"{context}.csv"), index=False)

        for start in range(0, n, batch_size):
            batch_input = inputs.iloc[start : start + batch_size]
            processed = process_gaussian_latents(batch_input)
            images = latent_to_image(processed)
            save_images(images, images_dir, start)
