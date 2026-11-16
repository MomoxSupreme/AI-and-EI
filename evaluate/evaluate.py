from functools import partial

import numpy as np
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from sklearn.metrics import average_precision_score, f1_score

from Models.models import model_name
from test import test_model, test_svm
from training import train_model, train_svm
from Utils.config_reader import read_config
from Utils.datasets import DATASET_FOR_KIND, MEL_HOP_LENGTH, N_MELS, DesedDataset, make_loader
from Utils.stats import get_pytorch_stats, get_svm_stats

"""
Construit results.csv : une ligne par (modèle, largeur, graine) + une ligne de baseline triviale.
codecarbon mesure l'énergie / les émissions, Utils.stats les paramètres et FLOPs.

Règle de mesure : pour tous les modèles, l'énergie de test couvre toute la chaîne
fichier wav -> prédiction (chargement, rééchantillonnage, mel ou MFCC compris).
"""

# --- à régler ici -------------------------------------------------------------------------------
# largeurs testées par famille (1.0 = architecture de référence)
WIDTHS = {
    "m5": [0.25, 0.5, 1.0],
    "cnn2d": [0.5, 1.0, 2.0],
    "crnn": [0.5, 1.0, 2.0],
}
SEEDS = None          # None -> graine de config.ini ; [0, 1, 2] pour 3 répétitions (3x plus long)
SMOKE_TEST = False    # True : 1 largeur par famille, 1 epoch, pour vérifier que tout tourne
# -----------------------------------------------------------------------------------------------

COLUMNS = [
    "model_name", "family", "width", "seed",
    "total_params", "trainable_params", "flops_per_passage",
    "training_consumption_kwh", "training_emissions_kg",
    "best_val_f1", "test_f1", "test_map", "test_consumption_kwh_per_passage",
]


def measure_training(model_name, train_function):
    """Mesure l'énergie et les émissions d'une fonction d'entraînement."""
    tracker = EmissionsTracker(project_name=f"{model_name}_train", save_to_file=False, log_level="error")

    tracker.start()
    model, history = train_function()
    tracker.stop()

    energy_data = tracker.final_emissions_data
    print(f"{model_name} (Train): {energy_data.energy_consumed:.6f} kWh | {energy_data.emissions:.6f} kg CO2eq")
    return model, history, energy_data.energy_consumed, energy_data.emissions


def measure_testing(model_name, model, test_function, test_data, n_samples, config):
    """Mesure l'énergie par passage (clip), le F1 macro et la mAP en test."""
    tracker = EmissionsTracker(project_name=f"{model_name}_test", save_to_file=False, log_level="error")

    tracker.start()
    test_f1, test_map = test_function(model, test_data, config=config)
    tracker.stop()

    total_test_energy = tracker.final_emissions_data.energy_consumed
    energy_per_passage = total_test_energy / n_samples if n_samples > 0 else 0.0

    print(f"{model_name} (Test): F1 {test_f1:.4f} | mAP {test_map:.4f} | {energy_per_passage:.9f} kWh/passage")
    return test_f1, test_map, energy_per_passage


def prior_baseline_row(cfg):
    """Baseline triviale : prédit la fréquence de chaque classe (calculée sur le train). Aucun audio lu."""

    def labels_of(split):
        ds = DesedDataset(split=split, config=cfg)
        return np.stack([ds.labels[str(f)] for f in ds.files])

    p = labels_of("train").mean(axis=0)
    Y = labels_of("test").astype(int)
    probs = np.tile(p, (len(Y), 1))

    row = {c: np.nan for c in COLUMNS}
    row.update({
        "model_name": "prior", "family": "prior", "total_params": len(p), "trainable_params": 0,
        "flops_per_passage": 0,
        "test_f1": f1_score(Y, (probs >= 0.5).astype(int), average="macro", zero_division=0),
        "test_map": average_precision_score(Y, probs, average="macro"),
    })
    return row


def run_one(kind, width, run_cfg, test_loaders, sample_inputs, n_test):
    if kind == "svm":
        name = "svm"
        train_fn = lambda: train_svm(run_cfg)
        test_fn, test_data = test_svm, None   # test_svm extrait lui-même les MFCC, dans la mesure d'énergie
    else:
        name = model_name(kind, width)
        train_fn = lambda: train_model(kind, width, run_cfg, plot=False)
        test_fn, test_data = partial(test_model, kind, width), test_loaders[kind]

    model, history, train_energy, train_emissions = measure_training(name, train_fn)
    best_val_f1 = max(history["val_f1"]) if history.get("val_f1") else 0.0  # SVM : F1 moyen en validation croisée

    if kind == "svm":
        svm_list, _scaler = model
        total_params, flops = get_svm_stats(svm_list)
        trainable_params = total_params
    else:
        total_params, flops = get_pytorch_stats(model, sample_inputs[kind])
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    test_f1, test_map, test_energy = measure_testing(name, model, test_fn, test_data, n_test, run_cfg)

    return {
        "model_name": name, "family": kind, "width": width, "seed": run_cfg["seed"],
        "total_params": total_params, "trainable_params": trainable_params, "flops_per_passage": flops,
        "training_consumption_kwh": train_energy, "training_emissions_kg": train_emissions,
        "best_val_f1": best_val_f1, "test_f1": test_f1, "test_map": test_map,
        "test_consumption_kwh_per_passage": test_energy,
    }


def main():
    cfg = read_config()
    widths = {k: list(v) for k, v in WIDTHS.items()}
    if SMOKE_TEST:
        widths = {k: v[:1] for k, v in widths.items()}
        cfg = {**cfg, "epochs": 1}
    seeds = SEEDS or [cfg["seed"]]

    # nombre de clips du split test (lit seulement le .tsv, aucun audio)
    n_test = len(DesedDataset(split="test", config=cfg))

    # tailles d'entrée SANS axe batch : get_pytorch_stats ajoute lui-même le batch (unsqueeze(0))
    n_audio_samples = int(cfg["target_rate"]) * int(cfg["audio_seconds"])  # 160000
    n_frames = n_audio_samples // MEL_HOP_LENGTH + 1                       # 626
    sample_inputs = {
        "m5": torch.randn(1, n_audio_samples),
        "cnn2d": torch.randn(1, N_MELS, n_frames),
        "crnn": torch.randn(1, N_MELS, n_frames),
    }
    test_loaders = {
        kind: make_loader(DATASET_FOR_KIND[kind](split="test", config=cfg), 32, False, cfg)
        for kind in widths
    }

    runs = [("svm", None)] + [(kind, w) for kind, ws in widths.items() for w in ws]
    rows = [prior_baseline_row(cfg)]

    for seed in seeds:
        run_cfg = {**cfg, "seed": seed}   # la graine change aussi le split train/val
        for kind, width in runs:
            print(f"\n===== {kind} width={width} seed={seed} =====")
            rows.append(run_one(kind, width, run_cfg, test_loaders, sample_inputs, n_test))
            pd.DataFrame(rows, columns=COLUMNS).to_csv("results.csv", index=False)  # sauvegarde après chaque run

    print("\nrésultats écrits dans results.csv")


if __name__ == "__main__":
    main()