import os

import joblib
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Models.models import CRNNModel, M5model
from Utils.config_reader import read_config
from Utils.datasets import DesedDataset, DesedMelDataset, load_svm_arrays, make_loader
from Utils.metrics import evaluate_loader
from Utils.scheduler import EarlyStopping
from Utils.visualize import plot_history, print_progress_bar


def _train_nn(name, model, dataset_cls, config, batch_size, save_dir, save_name):
    """
    Boucle d'entraînement commune au M5 et au CRNN.
    """
    device = config["device"]
    epochs = config["epochs"]

    train_dataset = dataset_cls(split="train", config=config)
    val_dataset = dataset_cls(split="val", config=config)
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, config=config)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False, config=config)

    model = model.to(device)
    loss_fn = config["loss"]()
    optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.get("scheduler_factor", 0.5),
        patience=config.get("scheduler_patience", 3),
    )
    early_stopping = EarlyStopping(patience=config.get("early_stopping_patience", 6))

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_map": []}

    print(f"\n--- Entraînement {name} ({len(train_dataset)} clips train / {len(val_dataset)} clips val) ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            print_progress_bar(
                step,
                len(train_loader),
                prefix=f"Epoch {epoch + 1}/{epochs}",
                suffix=f"| Batch Loss: {loss.item():.4f}",
            )

        train_loss = running_loss / len(train_dataset)
        val_loss, val_f1, val_map = evaluate_loader(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_map"].append(val_map)

        print(
            f"   -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | Val mAP: {val_map:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"Early stopping à l'epoch {epoch + 1} (meilleure Val Loss : {early_stopping.best_loss:.4f})")
            break

    early_stopping.load_best_weights(model)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Modèle {name} sauvegardé dans : {save_path}")

    plot_history(history["train_loss"], history["val_loss"], history["val_f1"], name)
    return model, history


def train_m5(config=None, batch_size=32, save_dir="./Models/Saved_Models"):
    config = config if config is not None else read_config()
    torch.manual_seed(config["seed"])
    model = M5model(config["n_classes"])
    return _train_nn("M5", model, DesedDataset, config, batch_size, save_dir, "m5_best.pth")


def train_cnn(config=None, batch_size=32, save_dir="./Models/Saved_Models"):
    config = config if config is not None else read_config()
    torch.manual_seed(config["seed"])
    model = CRNNModel()
    return _train_nn("CRNN", model, DesedMelDataset, config, batch_size, save_dir, "cnn_best.pth")


def train_svm(config=None, save_dir="./Models/Saved_Models"):
    """
    Entraîne un SVM par classe (GridSearchCV en validation croisée sur le train).
    Renvoie ((svm_list, scaler), history) ; history["val_f1"] contient le F1 moyen en validation croisée.
    Aucune donnée du split test n'est lue ici : l'extraction MFCC du test n'est donc pas comptée
    dans l'énergie d'entraînement.
    """
    config = config if config is not None else read_config()
    n_classes = config["n_classes"]

    print("\n--- Extraction MFCC & Entraînement Multi-SVM ---")
    X_train, Y_train_all = load_svm_arrays(
        "train", config,
        progress=lambda step, total: print_progress_bar(step, total, prefix="MFCC Train"),
    )
    Y_train_all = Y_train_all.astype(int)

    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)

    p_grid = {"C": [0.1, 1.0, 10.0, 100.0], "gamma": [0.001, 0.01, 0.1]}

    svm_list, best_scores, best_params = [], [], []
    for i in range(n_classes):
        clf = GridSearchCV(
            estimator=SVC(class_weight="balanced"),
            param_grid=p_grid,
            scoring="f1",
        )
        clf.fit(X_train_n, Y_train_all[:, i])

        svm_list.append(clf)
        best_scores.append(clf.best_score_)
        best_params.append(clf.best_params_)

        print_progress_bar(
            i + 1,
            n_classes,
            prefix="GridSearch SVM",
            suffix=f"| Classe {i + 1}/{n_classes} F1 (CV): {clf.best_score_:.4f}",
        )

    cv_f1 = float(np.mean(best_scores))
    print(f"F1 macro en validation croisée : {cv_f1:.4f}")
    print("F1 (CV) par classe :", best_scores)
    print("Meilleurs paramètres :", best_params)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "multi_svm.joblib")
    joblib.dump({"models": svm_list, "scaler": scaler}, save_path)
    print(f"Modèle Multi-SVM sauvegardé dans : {save_path}")

    return (svm_list, scaler), {"val_f1": [cv_f1]}


if __name__ == "__main__":
    cfg = read_config()
    # train_m5(cfg)
    # train_cnn(cfg)
    # train_svm(cfg)