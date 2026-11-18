"""Plot the energy / FLOPs / params / accuracy results of evaluator.py.

Usage:
    python graph.py                          # reads evaluator_results.csv, writes ./plots
    python graph.py --csv other.csv --out plots --train-col training_consumption_kwh

Layout of the output:  plots/<subset>/<NN_plot_name>.png
Subsets: ced, whisper, ced_whisper, rag, non_rag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FAMILY_COLORS = {"CED": "tab:blue", "Whisper": "tab:orange"}
HEAD_MARKERS = {"linear": "o", "mlp": "s", "none": "^"}
HEAD_LABELS = {"linear": "linear head", "mlp": "MLP head", "none": "RAG (no head)"}
KWH_TO_J = 3.6e6

FLOPS, PARAMS = "flops_per_passage", "total_params"
TEST_E, F1, ACC = "test_consumption_kwh_per_passage", "test_f1", "test_accuracy"

# Row filters. Each plot below then applies its own extra filter (e.g. non-RAG only).
SUBSETS = {
    "ced": ("CED models only", lambda d: d.family == "CED"),
    "whisper": ("Whisper models only", lambda d: d.family == "Whisper"),
    "ced_whisper": ("CED & Whisper", lambda d: d.family.notna()),
    "rag": ("RAG only", lambda d: d.is_rag),
    "non_rag": ("Non-RAG only", lambda d: ~d.is_rag),
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")  # sniff ',' or '\t'
    df = df.dropna(subset=["model_name"]).copy()
    df["is_rag"] = df.model_name.str.contains("(RAG)", regex=False)
    df["base_model"] = df.model_name.str.replace(" (RAG)", "", regex=False).str.strip()
    df["family"] = np.where(df.base_model.str.contains("ced", case=False), "CED", "Whisper")
    df["size"] = df.base_model.str.split("/").str[-1].str.replace(r"^(ced|whisper)-", "", regex=True)
    if "head" not in df.columns:
        df["head"] = np.where(df.is_rag, "none", "linear")
    df["head"] = df["head"].fillna("none").astype(str)
    for col in df.columns.difference(["model_name", "base_model", "family", "size", "head", "is_rag"]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Generic plot helpers
# --------------------------------------------------------------------------- #
def _legend(ax, df):
    handles = [
        Line2D([], [], marker="o", ls="", color=c, label=f, markeredgecolor="k")
        for f, c in FAMILY_COLORS.items() if f in set(df.family)
    ]
    handles += [
        Line2D([], [], marker=HEAD_MARKERS[h], ls="", color="lightgray", markeredgecolor="k",
               label=HEAD_LABELS.get(h, h))
        for h in HEAD_MARKERS if h in set(df["head"])
    ]
    ax.legend(handles=handles, fontsize=8, loc="best")


def scatter(df, x, y, xlabel, ylabel, title, path, logx=True, logy=True):
    d = df.dropna(subset=[x, y])
    if logx:
        d = d[d[x] > 0]
    if logy:
        d = d[d[y] > 0]
    if d.empty:
        return False

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for (fam, head), g in d.groupby(["family", "head"]):
        ax.scatter(g[x], g[y], c=FAMILY_COLORS[fam], marker=HEAD_MARKERS.get(head, "o"),
                   s=60, edgecolor="k", linewidth=0.6, alpha=0.9, zorder=3)
        for _, r in g.iterrows():
            ax.annotate(r["size"], (r[x], r[y]), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, zorder=4)

    ax.set_xscale("log" if logx else "linear")
    ax.set_yscale("log" if logy else "linear")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    _legend(ax, d)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _model_order(d):
    return (d.drop_duplicates("base_model")
             .sort_values(["family", FLOPS])[["base_model", "family", "size"]])


def grouped_bars(df, metric, ylabel, title, path):
    d = df.dropna(subset=[metric])
    if d.empty:
        return False
    order = _model_order(d)
    heads = [h for h in HEAD_MARKERS if h in set(d["head"])]
    multi_family = d.family.nunique() > 1
    labels = [f"{r.family}\n{r.size}" if multi_family else r.size for r in order.itertuples()]
    pivot = d.pivot_table(index="base_model", columns="head", values=metric).reindex(order.base_model)

    width = 0.8 / len(heads)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(order) + 2), 4.8))
    for i, h in enumerate(heads):
        ax.bar(np.arange(len(order)) + (i - (len(heads) - 1) / 2) * width, pivot[h],
               width, label=HEAD_LABELS[h], edgecolor="k", linewidth=0.5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def training_breakdown(df, path):
    """Stacked bars: embedding extraction vs head training, per model and head."""
    need = {"encoding_consumption_kwh", "training_consumption_kwh"}
    d = df[~df.is_rag]
    if d.empty or not need.issubset(d.columns):
        return False
    d = d.sort_values(["family", FLOPS, "head"])
    multi_family = d.family.nunique() > 1
    labels = [f"{r.family}\n{r.size}\n{r.head}" if multi_family else f"{r.size}\n{r.head}"
              for r in d.itertuples()]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(d) + 2), 4.8))
    ax.bar(x, d.encoding_consumption_kwh, label="encoding of the train set", edgecolor="k", linewidth=0.5)
    ax.bar(x, d.training_consumption_kwh, bottom=d.encoding_consumption_kwh,
           label="head training", edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("energy (kWh)")
    ax.set_title("Where the training energy goes", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="evaluator_results.csv")
    parser.add_argument("--out", default="plots")
    parser.add_argument("--train-col", default=None,
                        help="Training-energy column. Default: total_training_consumption_kwh if "
                             "present (head training + encoding), else training_consumption_kwh.")
    args = parser.parse_args()

    df = load(args.csv)
    train_col = args.train_col or (
        "total_training_consumption_kwh" if "total_training_consumption_kwh" in df.columns
        else "training_consumption_kwh")
    train_label = ("total training energy: encoding + head (kWh)" if train_col.startswith("total")
                   else "training energy (kWh)")
    if "total_training_consumption_kwh" in df.columns:
        df["train_total"] = df["total_training_consumption_kwh"]
    df["j_per_gflop"] = df[TEST_E] * KWH_TO_J / (df[FLOPS] / 1e9)
    test_label = "test energy per passage (kWh)"

    # (file name, rows filter, kind, kwargs).  Filter None = keep all rows of the subset.
    non_rag = lambda d: d[~d.is_rag]
    plots = [
        # ---- the 5 requested plots ----
        ("01_train_energy_vs_flops", non_rag, "scatter",
         dict(x=FLOPS, y=train_col, xlabel="FLOPs per passage", ylabel=train_label,
              title="Training energy vs FLOPs")),
        ("02_train_energy_vs_params", non_rag, "scatter",
         dict(x=PARAMS, y=train_col, xlabel="total parameters", ylabel=train_label,
              title="Training energy vs total parameters")),
        ("03_test_energy_vs_flops", None, "scatter",
         dict(x=FLOPS, y=TEST_E, xlabel="FLOPs per passage", ylabel=test_label,
              title="Test energy vs FLOPs")),
        ("04_test_energy_vs_params", None, "scatter",
         dict(x=PARAMS, y=TEST_E, xlabel="total parameters", ylabel=test_label,
              title="Test energy vs total parameters")),
        ("05_test_energy_vs_f1", None, "scatter",
         dict(x=F1, y=TEST_E, xlabel="test F1", ylabel=test_label, logx=False,
              title="Test energy vs test F1")),
        # ---- extra plots ----
        ("06_test_energy_vs_accuracy", None, "scatter",
         dict(x=ACC, y=TEST_E, xlabel="test exact-match accuracy", ylabel=test_label,
              logx=False,
              title="Test energy vs exact-match accuracy")),
        ("07_flops_vs_f1", None, "scatter",
         dict(x=F1, y=FLOPS, xlabel="test F1", ylabel="FLOPs per passage", logx=False,
              title="FLOPs vs test F1")),
        ("08_energy_efficiency", None, "scatter",
         dict(x=FLOPS, y="j_per_gflop", xlabel="FLOPs per passage",
              ylabel="test energy per GFLOP (J/GFLOP)",
              title="Energy per GFLOP: small models use the GPU poorly")),
        ("09_total_train_energy_vs_f1", None, "scatter",
         dict(x=F1, y="train_total", xlabel="test F1",
              ylabel="total training energy: encoding + head (kWh)", logx=False,
              title="Total training cost vs test F1")),
        ("10_f1_by_model_and_method", None, "bars",
         dict(metric=F1, ylabel="test F1", title="Test F1: linear vs MLP head vs RAG")),
        ("11_accuracy_by_model_and_method", None, "bars",
         dict(metric=ACC, ylabel="test exact-match accuracy",
              title="Test accuracy: linear vs MLP head vs RAG")),
        ("12_training_energy_breakdown", None, "breakdown", {}),
    ]

    out_root = Path(args.out)
    n_saved = 0
    for key, (subset_title, keep) in SUBSETS.items():
        sub = df[keep(df)]
        folder = out_root / key
        folder.mkdir(parents=True, exist_ok=True)
        for name, row_filter, kind, kw in plots:
            d = row_filter(sub) if row_filter else sub
            if d.empty:
                continue  # e.g. training plots have nothing to show for "RAG only"
            path = folder / f"{name}.png"
            if kind == "scatter":
                kw = dict(kw, title=f"{kw['title']}  [{subset_title}]")
                ok = scatter(d, path=path, **kw)
            elif kind == "bars":
                kw = dict(kw, title=f"{kw['title']}  [{subset_title}]")
                ok = grouped_bars(d, path=path, **kw)
            else:
                ok = training_breakdown(d, path)
            n_saved += bool(ok)
    print(f"Saved {n_saved} figures under {out_root.resolve()}")


if __name__ == "__main__":
    main()