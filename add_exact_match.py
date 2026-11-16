"""
Ajoute la colonne `test_accuracy` (exact-match : tous les labels du clip corrects) à results.csv,
en rechargeant les checkpoints sauvegardés par evaluate.py. Aucun ré-entraînement.

    python add_exact_match.py

À lancer à la racine du projet (là où se trouvent config.ini et Models/Saved_Models).
Une copie de l'original est gardée dans results_before_accuracy.csv.
"""

import os
import shutil

import joblib
import numpy as np
import pandas as pd
import torch

from Models.models import build_model, checkpoint_name
from Utils.config_reader import read_config
from Utils.datasets import DATASET_FOR_KIND, DesedDataset, load_svm_arrays, make_loader

CSV = "results.csv"
SAVE_DIR = "./Models/Saved_Models"


def exact_match_nn(model, loader, device, threshold=0.5):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for x, y in loader:
            pred = (torch.sigmoid(model(x.to(device))).cpu() >= threshold).int()
            correct += (pred == y.int()).all(dim=1).sum().item()
            total += y.size(0)
    return correct / total


def exact_match_svm(seed, cfg):
    saved = joblib.load(os.path.join(SAVE_DIR, f"multi_svm_seed{seed}.joblib"))
    X, Y = load_svm_arrays("test", cfg)
    X = saved["scaler"].transform(X)
    pred = np.column_stack([clf.predict(X) for clf in saved["models"]])
    return float((pred == Y.astype(int)).all(axis=1).mean())


def exact_match_prior(cfg):
    """Le prior ne prédit jamais rien (fréquences < 0,5) : correct seulement pour un clip sans aucun label."""
    train = DesedDataset(split="train", config=cfg)
    test = DesedDataset(split="test", config=cfg)
    p = np.stack([train.labels[str(f)] for f in train.files]).mean(axis=0)
    Y = np.stack([test.labels[str(f)] for f in test.files]).astype(int)
    pred = (p >= 0.5).astype(int)
    return float((Y == pred).all(axis=1).mean())


def main():
    cfg = read_config()
    device = cfg["device"]
    df = pd.read_csv(CSV)
    shutil.copy(CSV, "results_before_accuracy.csv")

    loaders, accs = {}, []
    for _, r in df.iterrows():
        family, name = r["family"], r["model_name"]
        try:
            if family == "prior":
                acc = exact_match_prior(cfg)
            elif family == "svm":
                acc = exact_match_svm(int(r["seed"]), cfg)
            else:
                width, seed = float(r["width"]), int(r["seed"])
                if family not in loaders:
                    ds = DATASET_FOR_KIND[family](split="test", config=cfg)
                    loaders[family] = make_loader(ds, 32, shuffle=False, config=cfg)
                model = build_model(family, width, cfg["n_classes"]).to(device)
                weights = os.path.join(SAVE_DIR, checkpoint_name(family, width, seed) + ".pth")
                model.load_state_dict(torch.load(weights, map_location=device))
                acc = exact_match_nn(model, loaders[family], device)
        except FileNotFoundError as e:
            print(f"[manquant] {name} : {e}")
            acc = np.nan
        print(f"{name}: exact-match = {acc:.4f}")
        accs.append(acc)

    df["test_accuracy"] = accs
    df.to_csv(CSV, index=False)
    print(f"\ncolonne test_accuracy ajoutée à {CSV}")


if __name__ == "__main__":
    main()