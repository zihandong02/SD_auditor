#!/usr/bin/env python3
"""
plot.py

Load a CSV of experiment results (one row per tau) and
create a combined CI‐length vs τ and coverage vs τ figure.
"""
import argparse
import ast
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_l2_vs_tau(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str = "l2_err",
    c_value: float | None = None,
):
    """
    Plot L2 error vs tau in a single axis.

    df:        DataFrame indexed by tau; columns start with 'mean_l2_'.
               Example: mean_l2_opt, mean_l2_mar, mean_l2_base, mean_l2_ols
    out_dir:   output directory
    prefix:    filename prefix (default: 'l2_err' -> 'l2_err_vs_tau.pdf')
    c_value:   number shown at the top title, e.g., 'c = 0.5'
    """
    # style
    colors  = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    markers = ["o", "s", "^", "D"]

    # detect L2 columns
    l2_cols = [c for c in df.columns if c.startswith("mean_l2_")]
    if not l2_cols:
        raise ValueError("No columns starting with 'mean_l2_' found in the CSV.")

    # pretty labels
    nice = {
        "mean_l2_opt":  "PCAL-CA",
        "mean_l2_mar":  "PCAL",
        "mean_l2_base": "label+unlabeled",
        "mean_l2_ols":  "label-only",
    }
    labels = [nice.get(col, col.replace("mean_l2_", "").upper()) for col in l2_cols]

    # figure
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    for col, color, marker, lab in zip(l2_cols, colors, markers, labels):
        ax.plot(df.index, df[col], marker=marker, linestyle="-", color=color, label=lab)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\ell_2$ error")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Method", loc="best")

    if c_value is not None:
        fig.suptitle(fr"$c = {c_value:g}$", y=1.02, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savepath = out_dir / f"{prefix}_vs_tau.pdf"
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved L2 plot to {savepath}")

def plot_ci_and_cov_vs_tau(
    df: pd.DataFrame,
    alpha_level: float,
    out_dir: Path,
    prefix: str = "ci_cov",
    c_value: float | None = None,
):
    """
    Plot CI length and coverage vs tau in a single figure.

    df:           DataFrame indexed by tau, containing at least columns
                  'mean_len_*' for CI lengths and 'covg_*' for coverages.
    alpha_level:  significance level for coverage reference line
    out_dir:      directory in which to save the figure
    prefix:       filename prefix
    c_value:      number shown at the top title, e.g., 'c = 0.5'
    """
    # pick vibrant colors and distinct markers
    colors  = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    markers = ["o", "s", "^", "D"]
    suffixes = ["opt", "mar", "base", "ols"]
    labels   = ["PCAL-CA", "PCAL", "label+unlabeled", "label-only"]

    # pair each curve with its column by suffix, not by column order
    length_cols   = [f"mean_len_{s}" for s in suffixes]
    coverage_cols = [f"covg_{s}" for s in suffixes]

    # create a 1×2 subplot
    fig, (ax_len, ax_cov) = plt.subplots(1, 2, figsize=(12, 4))

    # -----------------------
    # Plot CI length vs τ
    # -----------------------
    for col, c, m, lab in zip(length_cols, colors, markers, labels):
        ax_len.plot(df.index, df[col], marker=m, linestyle="-", color=c, label=lab)
    ax_len.set_xlabel(r"$\tau$")
    ax_len.set_ylabel("CI length")
    ax_len.set_title(r"Lengths of confidence intervals with varying $\tau$")
    ax_len.grid(True, linestyle="--", alpha=0.5)
    ax_len.legend(title="Method")

    # --------------------------
    # Plot coverage vs τ
    # --------------------------
    for col, c, m, lab in zip(coverage_cols, colors, markers, labels):
        ax_cov.plot(df.index, df[col], marker=m, linestyle="-", color=c, label=lab)
    ax_cov.axhline(
        1 - alpha_level, linestyle="--", color="gray",
        label=f"$1-\\alpha = {1-alpha_level:.2f}$"
    )
    ax_cov.set_xlabel(r"$\tau$")
    ax_cov.set_ylabel("CI coverage")
    ax_cov.set_title(r"Coverage of confidence interval with varying $\tau$")
    ax_cov.set_ylim(0.5, 1.05)
    ax_cov.grid(True, linestyle="--", alpha=0.5)
    ax_cov.legend(title="Method", loc="lower right")

    # title at the very top
    if c_value is not None:
        fig.suptitle(fr"$c = {c_value:g}$", y=1.02, fontsize=12)

    # adjust layout and save (leave room for suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    suffix = "" if c_value is None else f"_c{c_value:g}"
    savepath = out_dir / f"{prefix}_vs_tau{suffix}.pdf"
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved combined plot to {savepath}")


def main():
    parser = argparse.ArgumentParser(description="Plot CI length & coverage vs τ")
    parser.add_argument("--csv", required=True, help="Path to the CSV file produced by main.py")
    parser.add_argument("--out_dir", required=True, help="Directory in which to save the figure")
    parser.add_argument("--alpha_level", type=float, default=0.10, help="Significance level (default: 0.10)")
    parser.add_argument("--c", type=float, required=True, help="Value of c to display at the top of the figure")
    parser.add_argument("--prefix", default="ci_cov",
                        help="Filename prefix; the figure is written to "
                             "'{prefix}_vs_tau_c{c}.pdf'")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # if alpha_opt exists and is a string-form list, parse it
    if "alpha_opt" in df.columns:
        df["alpha_opt"] = df["alpha_opt"].apply(ast.literal_eval)

    # set tau as the index
    df = df.set_index("tau").sort_index()
    print(f"[INFO] τ values: {df.index.tolist()}")

    plot_ci_and_cov_vs_tau(
        df,
        alpha_level=args.alpha_level,
        out_dir=out_dir,
        prefix=args.prefix,
        c_value=args.c,
    )


if __name__ == "__main__":
    main()