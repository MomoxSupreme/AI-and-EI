import os

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

from Models.models import build_model, checkpoint_name
from Utils.config_reader import read_config
from Utils.datasets import DATASET_FOR_KIND, load_svm_arrays, make_loader
from Utils.metrics import evaluate_loader

SAVE_DIR = "./Models/Saved_Models"


def test_model(kind, width=1.0, model=None, test_loader=None, config=None, batch_size=32, weights_path=None):
    """
    Renvoie (F1 macro, mAP macro) sur le split test.
    `model` et `test_loader` sont optionnels : sinon, poids sauvegardés et loader par défaut.
    """
    config = config if config is not None else read_config()
    device = config["device"]

    if test_loader is None:
        dataset = DATASET_FOR_KIND[kind](split="test", config=config)
        test_loader = make_loader(dataset, batch_size, shuffle=False, config=config)

    if model is None:
        if weights_path is None:
            weights_path = os.path.join(SAVE_DIR, checkpoint_name(kind, width, config["seed"]) + ".pth")
        model = build_model(kind, width, config["n_classes"]).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        model = model.to(device)

    _, f1, mean_ap = evaluate_loader(model, test_loader, None, device)
    return f1, mean_ap


def test_svm(model=None, test_data=None, config=None, model_path=None):
    """
    model     : (svm_list, scaler), tel que renvoyé par train_svm ; sinon chargé depuis model_path.
    test_data : (X_test, Y_test) déjà extraits ; sinon les MFCC sont extraits ici (et comptés dans l'énergie).
    Renvoie (F1 macro, mAP macro).
    """
    config = config if config is not None else read_config()

    if model is None:
        model_path = model_path or os.path.join(SAVE_DIR, f"multi_svm_seed{config['seed']}.joblib")
        saved = joblib.load(model_path)
        svm_list, scaler = saved["models"], saved["scaler"]
    else:
        svm_list, scaler = model

    if test_data is None:
        X_test, Y_test = load_svm_arrays("test", config)
    else:
        X_test, Y_test = test_data

    X_test_n = scaler.transform(X_test)
    f1s, aps = [], []
    for i, clf in enumerate(svm_list):
        y = Y_test[:, i].astype(int)
        f1s.append(f1_score(y, clf.predict(X_test_n), zero_division=0))
        aps.append(average_precision_score(y, clf.decision_function(X_test_n)))
    return float(np.mean(f1s)), float(np.mean(aps))


if __name__ == "__main__":
    cfg = read_config()
    for kind in ("m5", "cnn2d", "crnn"):
        f1, mean_ap = test_model(kind, 1.0, config=cfg)
        print(f"{kind} (w1.0) | Test F1: {f1:.4f} | Test mAP: {mean_ap:.4f}")
    f1, mean_ap = test_svm(config=cfg)
    print(f"SVM | Test F1: {f1:.4f} | Test mAP: {mean_ap:.4f}")