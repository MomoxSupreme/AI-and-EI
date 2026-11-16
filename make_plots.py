"""
Génère les graphiques énergie / performance.

    python make_plots.py mine       -> figures/mine/     (tes résultats seulement)
    python make_plots.py compare    -> figures/compare/  (tes résultats + ceux de ton camarade)

Options : --mine results.csv  --mate evaluator_results.csv  --out figures

Notes
 - Si ton csv contient une colonne `test_accuracy` (exact-match, voir add_exact_match.py), elle est utilisée pour les
   graphiques 3 et 6. Sinon : mode mine -> mAP à la place ; mode compare -> graphique 3 sauté, graphique 6 en F1.
 - Si ton csv contient plusieurs graines pour un même modèle, les valeurs sont moyennées.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

KWH_TO_J = 3.6e6

SERIES_ORDER = ["CED", "Whisper", "M5", "CNN2D", "CRNN", "SVM", "Prior"]
COLORS = {
    "CED": "#1f77b4", "Whisper": "#ff7f0e", "M5": "#2ca02c", "CNN2D": "#d62728",
    "CRNN": "#9467bd", "SVM": "#8c564b", "Prior": "#7f7f7f",
}
HEAD_ORDER = ["linear", "mlp", "rag", "scratch"]
MARKERS = {"linear": "o", "mlp": "s", "rag": "^", "scratch": "D"}
HATCHES = {"linear": "", "mlp": "//", "rag": "xx", "scratch": ".."}
HEAD_LABELS = {
    "linear": "linear head", "mlp": "MLP head", "rag": "RAG (no head)", "scratch": "trained from scratch",
}
ENC_COLOR, HEAD_COLOR = "#1f77b4", "#ff7f0e"

MATE_SIZES = {
    "CED": ["tiny", "mini", "small", "base"],
    "Whisper": ["tiny", "base", "small", "medium", "large-v2"],
}
MINE_NAMES = {"m5": "M5", "cnn2d": "CNN2D", "crnn": "CRNN", "svm": "SVM", "prior": "Prior"}


# ----------------------------------------------------------------------------
# chargement : on ramène les deux csv au même schéma
# ----------------------------------------------------------------------------
def load_mine(path):
    d = pd.read_csv(path)
    width = d["width"]
    out = pd.DataFrame({
        "series": d["family"].map(lambda f: MINE_NAMES.get(f, str(f).upper())),
        "variant": [("" if pd.isna(w) else f"w{w:g}") for w in width],
        "head": "scratch",
        "order": width.fillna(0.0),
        "params": d["total_params"],
        "flops": d["flops_per_passage"],
        "train_kwh": d["training_consumption_kwh"],
        "train_enc_kwh": np.nan,
        "train_head_kwh": np.nan,
        "test_kwh": d["test_consumption_kwh_per_passage"],
        "f1": d["test_f1"],
        "map": d["test_map"],
        "acc": d["test_accuracy"] if "test_accuracy" in d.columns else np.nan,
    })
    # plusieurs graines -> moyenne
    return out.groupby(["series", "variant", "head"], as_index=False, sort=False).mean(numeric_only=True)


def load_mate(path):
    d = pd.read_csv(path)
    is_rag = d["model_name"].str.contains("(RAG)", regex=False)
    base = d["model_name"].str.replace(" (RAG)", "", regex=False)
    short = base.str.split("/").str[-1]                 # ced-tiny, whisper-large-v2
    family = short.str.split("-").str[0]                # ced, whisper
    variant = short.str.split("-", n=1).str[1]          # tiny, large-v2
    series = family.map({"ced": "CED", "whisper": "Whisper"}).fillna(family.str.title())
    order = [
        MATE_SIZES[s].index(v) if v in MATE_SIZES.get(s, []) else 99
        for s, v in zip(series, variant)
    ]
    return pd.DataFrame({
        "series": series,
        "variant": variant,
        "head": np.where(is_rag, "rag", d["head"]),
        "order": order,
        "params": d["total_params"],
        "flops": d["flops_per_passage"],
        "train_kwh": d["total_training_consumption_kwh"],
        "train_enc_kwh": d["encoding_consumption_kwh"],
        "train_head_kwh": d["training_consumption_kwh"],
        "test_kwh": d["test_consumption_kwh_per_passage"],
        "f1": d["test_f1"],
        "map": np.nan,
        "acc": d["test_accuracy"],
    })


def finalize(df):
    df = df.copy()
    df["srank"] = df["series"].map(lambda s: SERIES_ORDER.index(s) if s in SERIES_ORDER else 99)
    df["hrank"] = df["head"].map(HEAD_ORDER.index)
    df = df.sort_values(["srank", "order", "hrank"]).reset_index(drop=True)
    df["label"] = (df["series"] + " " + df["variant"]).str.strip()
    df["group"] = df["series"] + "|" + df["variant"]
    return df


# ----------------------------------------------------------------------------
# utilitaires de tracé
# ----------------------------------------------------------------------------
def _color(series):
    return COLORS.get(series, "gray")


def _tag(d):
    present = set(d["series"])
    names = [s for s in SERIES_ORDER if s in present] + sorted(present - set(SERIES_ORDER))
    return ", ".join(names)


def _style_axes(ax, xlabel, ylabel, title, logx=False, logy=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)


def _legend_scatter(ax, d):
    handles = [
        Line2D([0], [0], marker="o", ls="", markerfacecolor=_color(s), markeredgecolor="black",
               markersize=8, label=s)
        for s in [x for x in SERIES_ORDER if x in set(d["series"])]
    ]
    heads = [h for h in HEAD_ORDER if h in set(d["head"])]
    if len(heads) > 1:
        handles += [
            Line2D([0], [0], marker=MARKERS[h], ls="", markerfacecolor="lightgray", markeredgecolor="black",
                   markersize=8, label=HEAD_LABELS[h])
            for h in heads
        ]
    ax.legend(handles=handles, fontsize=8, loc="best")


def scatter_plot(df, xcol, ycol, xlabel, ylabel, title, path, logx=False, logy=True):
    d = df.dropna(subset=[xcol, ycol])
    if logx:
        d = d[d[xcol] > 0]
    if logy:
        d = d[d[ycol] > 0]
    if d.empty:
        print(f"[ignoré] {os.path.basename(path)} : pas de données")
        return

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for head, sub in d.groupby("head"):
        ax.scatter(
            sub[xcol], sub[ycol], c=[_color(s) for s in sub["series"]], marker=MARKERS[head],
            s=70, edgecolors="black", linewidths=0.7, zorder=3,
        )

    seen = set()  # une étiquette par modèle (le premier point rencontré)
    for _, r in d.iterrows():
        if r["group"] in seen:
            continue
        seen.add(r["group"])
        text = r["variant"] if r["series"] in MATE_SIZES else r["label"]
        ax.annotate(text, (r[xcol], r[ycol]), textcoords="offset points", xytext=(5, 5), fontsize=7)

    _style_axes(ax, xlabel, ylabel, f"{title}  [{_tag(d)}]", logx, logy)
    _legend_scatter(ax, d)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {path}")


def grouped_bars(df, col, ylabel, title, path):
    d = df.dropna(subset=[col])
    groups = list(dict.fromkeys(d["group"]))
    if not groups:
        print(f"[ignoré] {os.path.basename(path)} : pas de données")
        return

    nmax = int(d.groupby("group").size().max())
    w = 0.8 / nmax
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(groups) + 2), 6))

    ticks = []
    for g, key in enumerate(groups):
        sub = d[d["group"] == key]
        n = len(sub)
        for j, (_, r) in enumerate(sub.iterrows()):
            ax.bar(
                g + (j - (n - 1) / 2) * w, r[col], width=w, color=_color(r["series"]),
                edgecolor="black", linewidth=0.6, hatch=HATCHES[r["head"]],
            )
        first = sub.iloc[0]
        ticks.append(f"{first['series']}\n{first['variant']}".strip())

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(ticks, fontsize=7)
    ax.set_ylim(0, 1.0)
    _style_axes(ax, "", ylabel, f"{title}  [{_tag(d)}]")
    ax.grid(False, axis="x")

    handles = [Patch(facecolor=_color(s), edgecolor="black", label=s)
               for s in [x for x in SERIES_ORDER if x in set(d["series"])]]
    heads = [h for h in HEAD_ORDER if h in set(d["head"])]
    if len(heads) > 1:
        handles += [Patch(facecolor="white", edgecolor="black", hatch=HATCHES[h], label=HEAD_LABELS[h])
                    for h in heads]
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {path}")


def stacked_training_energy(df, path):
    d = df[df["head"] != "rag"].dropna(subset=["train_kwh"]).reset_index(drop=True)
    if d.empty:
        print(f"[ignoré] {os.path.basename(path)} : pas de données")
        return

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(d) + 2), 6))
    ticks = []
    for i, r in d.iterrows():
        if r["head"] == "scratch":
            ax.bar(i, r["train_kwh"], color=_color(r["series"]), edgecolor="black", linewidth=0.6)
            ticks.append(f"{r['series']}\n{r['variant']}".strip())
        else:
            enc = r["train_enc_kwh"]
            ax.bar(i, enc, color=ENC_COLOR, edgecolor="black", linewidth=0.6)
            ax.bar(i, r["train_kwh"] - enc, bottom=enc, color=HEAD_COLOR, edgecolor="black", linewidth=0.6)
            ticks.append(f"{r['series']}\n{r['variant']}\n{r['head']}")

    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(ticks, fontsize=6)
    _style_axes(ax, "", "training energy (kWh)", "Where the training energy goes")
    ax.grid(False, axis="x")

    handles = []
    if (d["head"] != "scratch").any():
        handles += [Patch(facecolor=ENC_COLOR, edgecolor="black", label="encoding of the train set"),
                    Patch(facecolor=HEAD_COLOR, edgecolor="black", label="head training")]
    scratch_series = [s for s in SERIES_ORDER if s in set(d.loc[d["head"] == "scratch", "series"])]
    handles += [Patch(facecolor=_color(s), edgecolor="black", label=f"{s} (end-to-end training)")
                for s in scratch_series]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {path}")


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["mine", "compare"])
    parser.add_argument("--mine", default="results.csv")
    parser.add_argument("--mate", default="evaluator_results.csv")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    mine = load_mine(args.mine)
    has_acc_mine = mine["acc"].notna().any()
    df = finalize(mine if args.mode == "mine" else pd.concat([load_mate(args.mate), mine], ignore_index=True))

    out = os.path.join(args.out, args.mode)
    os.makedirs(out, exist_ok=True)

    # métrique de qualité des graphiques 3 et 6
    if has_acc_mine:
        qcol, qname = "acc", "exact-match accuracy"
    elif args.mode == "mine":
        qcol, qname = "map", "mAP"
        print("[info] pas de colonne test_accuracy dans ton csv : mAP utilisée à la place (voir add_exact_match.py)")
    else:
        qcol, qname = "f1", "F1"
        print("[info] pas de colonne test_accuracy dans ton csv : graphique 3 ignoré, graphique 6 tracé en F1")

    # 1. énergie d'entraînement vs paramètres (têtes linear / MLP / modèles from scratch, pas le RAG)
    scatter_plot(df[df["head"] != "rag"], "params", "train_kwh", "total parameters",
                 "total training energy (kWh)", "Training energy vs total parameters",
                 os.path.join(out, "1_training_energy_vs_params.png"), logx=True, logy=True)

    # 2. énergie par GFLOP vs FLOPs
    d2 = df[df["flops"] > 0].copy()
    d2["j_per_gflop"] = d2["test_kwh"] * KWH_TO_J / (d2["flops"] / 1e9)
    scatter_plot(d2, "flops", "j_per_gflop", "FLOPs per passage", "test energy per GFLOP (J/GFLOP)",
                 "Test energy per GFLOP", os.path.join(out, "2_energy_per_gflop.png"), logx=True, logy=True)

    # 3. barres : exact-match accuracy (ou mAP)
    if qcol != "f1":
        grouped_bars(df, qcol, f"test {qname}", f"Test {qname}", os.path.join(out, "3_quality_bars.png"))

    # 4. barres : F1
    grouped_bars(df, "f1", "test F1", "Test F1", os.path.join(out, "4_f1_bars.png"))

    # 5. coût d'entraînement vs F1
    scatter_plot(df, "f1", "train_kwh", "test F1", "total training energy (kWh)",
                 "Total training cost vs test F1", os.path.join(out, "5_training_cost_vs_f1.png"),
                 logx=False, logy=True)

    # 6. énergie de test vs qualité
    scatter_plot(df, qcol, "test_kwh", f"test {qname}", "test energy per passage (kWh)",
                 f"Test energy vs {qname}", os.path.join(out, "6_test_energy_vs_quality.png"),
                 logx=False, logy=True)

    # 7. répartition de l'énergie d'entraînement
    stacked_training_energy(df, os.path.join(out, "7_training_energy_split.png"))


if __name__ == "__main__":
    main()