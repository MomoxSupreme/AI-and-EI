import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score

from Models.models import CRNNModel, M5model
from Utils.config_reader import read_config
from Utils.datasets import DesedDataset, DesedMelDataset, load_svm_arrays, make_loader
from Utils.metrics import evaluate_loader


def _test_nn(model, test_loader, config, build_model, dataset_cls, weights_path, batch_size):
    """F1 macro sur le split test. `model` et `test_loader` sont optionnels (sinon : poids sauvegardés + loader par défaut)."""
    config = config if config is not None else read_config()
    device = config["device"]

    if test_loader is None:
        test_loader = make_loader(dataset_cls(split="test", config=config), batch_size, shuffle=False, config=config)

    if model is None:
        model = build_model().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        model = model.to(device)

    _, f1, _ = evaluate_loader(model, test_loader, None, device)
    return f1


def test_m5(model=None, test_loader=None, config=None, batch_size=32,
            weights_path="./Models/Saved_Models/m5_best.pth"):
    cfg = config if config is not None else read_config()
    return _test_nn(model, test_loader, cfg, lambda: M5model(cfg["n_classes"]),
                    DesedDataset, weights_path, batch_size)


def test_crnn(model=None, test_loader=None, config=None, batch_size=32,
              weights_path="./Models/Saved_Models/cnn_best.pth"):
    cfg = config if config is not None else read_config()
    return _test_nn(model, test_loader, cfg, CRNNModel,
                    DesedMelDataset, weights_path, batch_size)


def test_svm(model=None, test_data=None, config=None,
             model_path="./Models/Saved_Models/multi_svm.joblib"):
    """
    model     : (svm_list, scaler), tel que renvoyé par train_svm ; sinon chargé depuis model_path.
    test_data : (X_test, Y_test) déjà extraits ; sinon les MFCC sont extraits ici (et comptés dans l'énergie).
    """
    config = config if config is not None else read_config()

    if model is None:
        saved = joblib.load(model_path)
        svm_list, scaler = saved["models"], saved["scaler"]
    else:
        svm_list, scaler = model

    if test_data is None:
        X_test, Y_test = load_svm_arrays("test", config)
    else:
        X_test, Y_test = test_data

    X_test_n = scaler.transform(X_test)
    scores = [
        f1_score(Y_test[:, i].astype(int), svm_list[i].predict(X_test_n), zero_division=0)
        for i in range(len(svm_list))
    ]
    return float(np.mean(scores))


if __name__ == "__main__":
    cfg = read_config()
    print(f"M5 Test F1: {test_m5(config=cfg):.4f}")
    print(f"CRNN Test F1: {test_crnn(config=cfg):.4f}")
    print(f"SVM Test F1: {test_svm(config=cfg):.4f}")