import numpy as np
import pandas as pd
from jinja2 import Template


def compute_stats(results, group, metric_lists):
    """
    Compute mean and SE for all metrics grouped by group.

    Args:
        results: DataFrame with results
        group: column name to group by
        metric_lists: dict like {'reconstruction': [...], 'concept_learning': [...], 'composition': [...]}

    Returns:
        dict with 'means', 'se' DataFrames, and 'group_labels'
    """
    # Flatten all metrics and filter to existing columns
    all_metrics = []
    for metric_list in metric_lists.values():
        all_metrics.extend(metric_list)
    all_metrics = [m for m in all_metrics if m in results.columns]

    metrics_df = results[[group] + all_metrics]
    grouped = metrics_df.groupby(group)

    means = grouped.mean()
    stds = grouped.std(ddof=1)
    counts = grouped.count()
    se = stds / np.sqrt(counts)

    return {
        "means": means,
        "se": se,
        "group_labels": list(means.index),
    }


def format_value(val, err, is_min=False):
    """Format a value ± error, optionally bold if minimum."""
    if pd.isna(val) or pd.isna(err):
        return "---"
    val_str = f"{val:.3f}"
    err_str = f"{err:.3f}"
    if is_min:
        return f"\\textbf{{{val_str}}} ± {err_str}"
    else:
        return f"{val_str} ± {err_str}"


def build_latex_table(stats, metric_groups, col_order, col_names, compo_display):
    """
    Build LaTeX table from stats.

    Args:
        stats: dict from compute_stats()
        metric_groups: dict like {'reconstruction': [...], 'concept_learning': [...], 'composition': [...]}
        col_order: list of group labels in desired order
        col_names: dict mapping group labels to display names

    Returns:
        LaTeX string
    """
    means = stats["means"].loc[col_order]
    se = stats["se"].loc[col_order]

    # Metric display name mapping
    row_display_names = {
        "final_val_elbo_bpd": "in-distribution",
        "ood_val_elbo_bpd": "out-of-distribution",
    }
    row_display_names.update(compo_display)

    # Build rows for each metric group
    rows = []

    for group_name, metrics in metric_groups.items():
        # Filter to existing metrics
        metrics = [m for m in metrics if m in means.columns]
        if not metrics:
            continue

        if group_name == "reconstruction":
            group_metric = "(ELBO BPD)"
        else:
            group_metric = "(sliced Wasserstein)"
        rows.append({"type": "header", "name": group_name, "cells": group_metric})

        # Add mean ± SE rows
        for metric in metrics:
            row_means = means[metric]
            row_se = se[metric]
            min_idx = row_means.idxmin()

            cells = []
            for group_label in col_order:
                is_min = group_label == min_idx
                cells.append(
                    format_value(row_means[group_label], row_se[group_label], is_min)
                )

            # Use display name if available, otherwise use metric name
            display_name = row_display_names.get(metric, metric)

            rows.append(
                {
                    "type": "data",
                    "name": display_name,
                    "cells": cells,  # <-- list of strings
                }
            )

        # Add entire-group mean and SE row for Concept Learning and Composition
        if group_name in ["concept learning", "composition"]:
            # Compute mean and SE across all groups for all metrics in this section
            group_means = []
            group_ses = []
            for metric in metrics:
                group_means.append(means[metric].mean())
                # SE of the mean across groups: std / sqrt(n_groups)
                group_ses.append(se[metric].mean())

            # Create a row with the average mean and SE
            cells = []
            col_means = []
            col_ses = []

            # First pass: collect all means and SEs for this column group
            for i in range(len(col_order)):
                # Average across metrics for this column
                metric_means_for_col = [
                    means[metric][col_order[i]] for metric in metrics
                ]
                metric_ses_for_col = [se[metric][col_order[i]] for metric in metrics]
                col_mean = np.mean(metric_means_for_col)
                col_se = np.mean(metric_ses_for_col)
                col_means.append(col_mean)
                col_ses.append(col_se)

            # Find the minimum mean
            min_idx = np.argmin(col_means)

            # Second pass: format cells with bolding for minimum
            for i in range(len(col_order)):
                is_min = i == min_idx
                cells.append(format_value(col_means[i], col_ses[i], is_min=is_min))

            # Add spacing before group mean row
            rows.append({"type": "addlinespace"})

            rows.append(
                {
                    "type": "data",
                    "name": f"{group_name} mean",
                    "cells": cells,
                }
            )

        rows.append({"type": "rule"})

    # Jinja2 template for LaTeX table
    template_str = r"""\begin{tabular}{r{{ 'c' * num_cols }}}
\toprule
& {% for col_label in col_order %}\textbf{ {{- col_names[col_label] -}} }{{ " & " if not loop.last else "" }}{% endfor %} \\
\midrule
{%- for row in rows -%}
{%- if row.type == 'header' %}
\multicolumn{ {{ num_cols + 1 }} }{l}{\textbf{ {{- row.name -}} }  {{ row.cells }}} \\
\midrule
{%- elif row.type == 'data' %}
\textbf{ {{- row.name -}} } & {{ row.cells | join(' & ') }} \\
{%- elif row.type == 'addlinespace' %}
\addlinespace
{%- elif row.type == 'rule' %}
\midrule
{%- endif %}
{%- endfor %}
\bottomrule
\end{tabular}
"""

    template = Template(template_str)
    latex = template.render(
        num_cols=len(col_order),
        col_order=col_order,
        col_names=col_names,
        rows=rows,
    )

    return latex


# Main execution (Snakemake)
if __name__ == "__main__":
    results = pd.read_csv(snakemake.input[0])
    dataset = snakemake.wildcards.dataset
    expt = snakemake.wildcards.expt

    # Dataset-specific metric groups
    match dataset:
        case s if s.startswith("quad"):
            from conceptualizer.utils import quad_composition as composition
            from conceptualizer.utils import quad_concept_learning as concept_learning

            sep = "_"
        case "mnist":
            concept_learning = [
                "obs",
                "scaled",
                "shear",
                "shift",
                "swel",
                "thic",
                "thin",
            ]
            composition = []
        case "3dident":
            concept_learning = ["obs", "bg", "obj", "sl"]
            composition = ["bg-obj", "bg-sl", "obj-sl"]
            sep = "-"
        case _:
            raise NotImplementedError(
                f"No table formatting defined for {dataset} dataset."
            )

    # reconstruction metrics for all cases:
    reconstruction = [
        "final_val_elbo_bpd",
        "ood_val_elbo_bpd",
    ]

    metric_groups = {
        "reconstruction": reconstruction,
        "concept learning": concept_learning,
        "composition": composition,
    }

    # composition display names:
    compo_display = {s: f"({x}, {y})" for s in composition for x, y in [s.split(sep)]}

    # Experiment-specific grouping and column ordering
    match expt:
        case "ablation":
            group_col = "arch_flag"
            col_order = [
                "2",
                "5",
                "4",
                "0",
                "1",
                "3",
                "Ada-GVAE",
                "BetaTCVAE",
                "TVAE",
            ]
            col_names = {
                "0": r"base",
                "1": r"base pooled",
                "2": r"CM (end-to-end)",
                "3": r"CM pooled",
                "4": r"CM (frozen)",
                "5": r"CM (fine-tune)",
                "Ada-GVAE": r"Ada-GVAE",
                "BetaTCVAE": r"BetaTCVAE",
                "TVAE": r"TVAE",
            }
        case "beta":
            group_col = "beta"
            col_order = sorted(results[group_col].unique())
            col_names = {col: str(col) for col in col_order}
        case s if s.startswith("expressivity_"):
            group_col = "exp"
            col_order = sorted(results[group_col].unique())
            col_names = {col: str(col) for col in col_order}
        case "groupnorm" | "l2norm":
            group_col = "reg"
            col_order = sorted(results[group_col].unique())
            col_names = {col: str(col) for col in col_order}
        case "disent":
            group_col = "arch_flag"
            col_order = sorted(results[group_col].unique())
            col_names = {col: str(col) for col in col_order}
        case _:
            raise NotImplementedError(
                f"No table formatting defined for {expt} experiment."
            )

    stats = compute_stats(results, group_col, metric_groups)
    latex = build_latex_table(stats, metric_groups, col_order, col_names, compo_display)

    with open(snakemake.output[0], "w") as f:
        f.write(latex)
