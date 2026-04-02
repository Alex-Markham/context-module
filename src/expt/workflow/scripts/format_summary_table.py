import re
from pathlib import Path

import pandas as pd
from jinja2 import Template


def extract_sub_summary(filepath):
    """
    Extract metrics from a LaTeX ablation table and return a pandas DataFrame.

    Extracts mean values (without uncertainty) for the rows:
    - "in-distribution" → reconstruction
    - "concept learning mean" → concept_learning
    - "composition mean" → composition

    Parameters:
    -----------
    filepath : str or Path
        Path to the .tex file containing the LaTeX table

    Returns:
    --------
    pd.DataFrame
        DataFrame with 3 rows (reconstruction, concept_learning, composition).
        Missing rows are filled with NaN.
    """

    filepath = Path(filepath)

    with open(filepath, "r") as f:
        content = f.read()

    # Extract all lines that are table rows (contain \\)
    lines = content.split("\n")
    rows = [line for line in lines if "\\\\" in line and "\\textbf{" in line]

    # Parse the header row to get method names
    header_line = [
        line for line in lines if ("CM (end-to-end)" in line or "Ada-GVAE" in line)
    ][0]
    methods = header_line.lstrip("& ").rstrip(" \\").split(" & ")
    methods = [m.removeprefix(r"\textbf{").removesuffix(r"}") for m in methods]

    # extract summary metrics
    metric_display_names = {
        "in-distribution": "recon",
        "concept learning mean": "concept",
        "composition mean": "compo",
    }

    metric_rows = [
        row for row in rows if any(k in row for k in metric_display_names.keys())
    ]

    # parse each row into list of cells (strip spaces and trailing \\)
    def parse_row(line):
        cells = [c.strip() for c in line.split("&") if c.strip()]
        cells[-1] = cells[-1].rstrip("\\").strip()
        m = re.match(r"\\textbf\{\s*(.*?)\s*\}", cells[0])
        cells[0] = m.group(1) if m else cells[0]
        cells = [
            cell.split(" ± ")[0].removeprefix(r"\textbf{").removesuffix(r"}")
            for cell in cells
        ]
        return cells

    parsed = [parse_row(r) for r in metric_rows]

    sub_summary = {row[0]: row[1:] for row in parsed}

    sub_summary_df = pd.DataFrame(sub_summary)
    sub_summary_df.index = methods
    sub_summary_df.rename(columns=metric_display_names, inplace=True)
    sub_summary_df = sub_summary_df.reindex(columns=metric_display_names.values())

    return sub_summary_df


def generate_cmidrules(num_datasets):
    """Generate cmidrule commands for n datasets with 3 columns each."""
    rules = []
    for i in range(num_datasets):
        start = 3 + i * 3
        end = 5 + i * 3
        rules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    return " ".join(rules)


template_str = r"""\begin{tabular}{c r {% for dataset in datasets %}*{3}{c}{% if not loop.last %} {% endif %}{% endfor %}}
\toprule
& &
{%- for dataset in datasets %} \multicolumn{3}{c}{\textbf{ {{-dataset-}} }}{{ " &" if not loop.last else "" }}
{%- endfor %} \\
{{cmidrules}}
& &
{%- for dataset in datasets %} \textbf{Concept} & \textbf{Compo} & \textbf{Recon}{{ " &" if not loop.last else "" }}
{%- endfor %} \\
\midrule
{%- for group_name, methods in method_groups %}
\multirow{ {{-methods|length-}} }{*}{\rotatebox{90}{\scriptsize\textbf{ {{-group_name-}} }}} &
{%- for method in methods %}
{%- if loop.index > 1 %} &
{%- endif %} \textbf{ {{-method-}} }
{%- for dataset in datasets %} & {{ df.loc[method, (dataset, 'concept')] }} & {{ df.loc[method, (dataset, 'compo')] }} & {{ df.loc[method, (dataset, 'recon')] }}
{%- endfor %} \\
{% endfor %}{%- if not loop.last %}\addlinespace
{%- if loop.index == 1 %}\midrule
{%- endif %}\addlinespace
{%- endif %}{%- endfor %}\bottomrule
\end{tabular}
"""


def render_table(df, datasets, method_groups):
    """Render LaTeX table from DataFrame."""
    template = Template(template_str)
    cmidrules = generate_cmidrules(len(datasets))
    return template.render(
        df=df, datasets=datasets, method_groups=method_groups, cmidrules=cmidrules
    )


if __name__ == "__main__":
    # parse auto-formatted tables
    auto = ["quad", "quad_causal", "mnist", "ident3d"]
    display_names = {
        "quad": r"\texttt{quad}* (independent)",
        "quad_causal": r"\texttt{quad}* (dependent)",
        "mnist": "MNIST**",
        "3dident": "3DIdent**",
    }
    result = pd.concat(
        [extract_sub_summary(snakemake.input[a]) for a in auto],
        axis=1,
        keys=display_names.keys(),
    )

    # parse manually formatted table
    df = pd.read_csv(snakemake.input["nvae"])
    manual = (
        df.set_index(["method", "dataset"])
        .unstack(level="dataset")
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )
    result.update(manual)

    # prepare the final table
    ## bold min per col
    numeric_result = result.astype(float)
    for col in result.columns:
        if col == ("mnist", "compo"):
            continue
        min_val = numeric_result[col].min()
        formatted_min = r"\textbf{" + str(min_val) + "}"
        is_min = numeric_result[col] == min_val
        result.loc[is_min, col] = formatted_min
    result = result.astype(str)
    ## change col and NaN display names
    result.rename(columns=display_names, inplace=True)
    result = result.replace("nan", "-")

    ## group methods and format final table
    method_groups = [
        ("ours", ["CM (end-to-end)", "CM (fine-tune)", "CM (frozen)"]),
        ("ablations", ["base", "base pooled", "CM pooled"]),
        ("baselines", ["Ada-GVAE", "BetaTCVAE", "TVAE"]),
    ]
    table = render_table(result, display_names.values(), method_groups)
    with open(snakemake.output[0], "w") as f:
        f.write(table)
