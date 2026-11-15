import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from Models.models import CRNNModel, M5model
from Utils.config_reader import read_config
from Utils.datasets import DesedDataset, DesedMelDataset, MicroDataLoader


def test_m5(model=None, test_loader=None, config=None, batch_size=32, weights_path="./Models/Saved_Models/m5_best.pth"):
    config = config if config is not None else read_config()
    device = config["device"]

    if test_loader is None:
        test_dataset = DesedDataset(
            split="eval",
            annotations_file=config["test_metadata_path"],
            sounds_dir=config["raw_test_datas_path"],
            config=config
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model is None:
        model = M5model(config["n_classes"]).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        model = model.to(device)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            logits = model(x_test)
            y_pred = torch.round(torch.sigmoid(logits))
            all_preds.append(y_pred.cpu())
            all_targets.append(y_test.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return f1_score(all_targets, all_preds, average="macro")


def test_crnn(model=None, test_loader=None, config=None, batch_size=32, weights_path="./Models/Saved_Models/cnn_best.pth"):
    config = config if config is not None else read_config()
    device = config["device"]

    if test_loader is None:
        test_dataset = DesedMelDataset(
            split="eval",
            annotations_file=config["test_metadata_path"],
            sounds_dir=config["raw_test_datas_path"],
            config=config
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model is None:
        model = CRNNModel().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        model = model.to(device)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            logits = model(x_test)
            y_pred = torch.round(torch.sigmoid(logits))
            all_preds.append(y_pred.cpu())
            all_targets.append(y_test.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return f1_score(all_targets, all_preds, average="macro")


def test_svm(model=None, test_data=None, config=None, model_path="./Models/Saved_Models/multi_svm.joblib"):
    config = config if config is not None else read_config()

    if model is None:
        saved = joblib.load(model_path)
        svm_list = saved["models"]
        scaler = saved["scaler"]
    else:
        svm_list, scaler = model

    if test_data is None:
        test_dataset = MicroDataLoader(
            split="eval",
            csv_path=config["test_metadata_path"],
            raw_data_path=config["raw_test_datas_path"],
            config=config
        )
        test_loader = DataLoader(test_dataset)
        X_test, Y_test_all = [], []
        for x, y in test_loader:
            X_test.append(x)
            Y_test_all.append(y)
        X_test = np.array(X_test).squeeze()
        Y_test_all = np.array(Y_test_all).squeeze()
    else:
        X_test, Y_test_all = test_data

    X_test_n = scaler.transform(X_test)
    n_classes = len(svm_list)
    svm_f1_scores = []

    for i in range(n_classes):
        Y_pred = svm_list[i].predict(X_test_n)
        svm_f1_scores.append(f1_score(Y_test_all[:, i], Y_pred))

    return float(np.mean(svm_f1_scores))


if __name__ == "__main__":
    cfg = read_config()
    print(f"M5 Test F1: {test_m5(config=cfg):.4f}")
    print(f"CRNN Test F1: {test_crnn(config=cfg):.4f}")
    print(f"SVM Test F1: {test_svm(config=cfg):.4f}")