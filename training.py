import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from Utils.config_reader import read_config
from Utils.datasets import DesedDataset, DesedMelDataset, MicroDataLoader
from Models.models import CRNNModel, M5model
from Utils.scheduler import EarlyStopping
from Utils.visualize import *





def train_m5(config=None, batch_size=32, epochs=10, save_dir="./Models/Saved_Models"):
    config = config if config is not None else read_config()
    torch.manual_seed(config["seed"])
    device = config["device"]
    learning_rate = config["learning_rate"]
    n_classes = config["n_classes"]

    train_dataset = DesedDataset(split="train", config=config)
    test_dataset = DesedDataset(split="test", config=config)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = M5model(n_classes).to(device)
    loss_fn = config["loss"]()
    optimizer = config["optimizer"](model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.get("scheduler_factor", 0.5),
        patience=config.get("scheduler_patience", 3),
    )
    early_stopping = EarlyStopping(patience=config.get("early_stopping_patience", 6))

    train_losses_history = []
    val_losses_history = []
    val_f1_history = []

    print("\n--- Entraînement M5 ---")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for step, (x, y_target) in enumerate(train_dataloader, start=1):
            x, y_target = x.to(device), y_target.to(device)

            optimizer.zero_grad()
            x_logits = model(x)
            loss = loss_fn(x_logits, y_target)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            print_progress_bar(
                step,
                len(train_dataloader),
                prefix=f"Epoch {epoch+1}/{epochs}",
                suffix=f"| Batch Loss: {loss.item():.4f}",
            )

        avg_train_loss = running_train_loss / len(train_dataloader)
        train_losses_history.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.inference_mode():
            for x_test, y_test in test_dataloader:
                x_test, y_test = x_test.to(device), y_test.to(device)

                test_logits = model(x_test)
                test_loss = loss_fn(test_logits, y_test)
                running_val_loss += test_loss.item()

                y_pred = torch.round(torch.sigmoid(test_logits))
                all_preds.append(y_pred.cpu())
                all_targets.append(y_test.cpu())

        avg_val_loss = running_val_loss / len(test_dataloader)
        val_losses_history.append(avg_val_loss)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        epoch_f1 = f1_score(all_targets, all_preds, average="macro")
        val_f1_history.append(epoch_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"   -> Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {epoch_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print(f"Early stopping à l'epoch {epoch+1} (meilleure Val Loss : {early_stopping.best_loss:.4f})")
            break

    early_stopping.load_best_weights(model)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "m5_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Modèle M5 sauvegardé dans : {save_path}")

    plot_history(train_losses_history, val_losses_history, val_f1_history, "M5")
    return model, {"train_loss": train_losses_history, "val_loss": val_losses_history, "val_f1": val_f1_history}


def train_cnn(config=None, batch_size=32, epochs=10, save_dir="./Models/Saved_Models"):
    config = config if config is not None else read_config()
    torch.manual_seed(config["seed"])
    device = config["device"]
    learning_rate = config["learning_rate"]

    train_dataset = DesedMelDataset(split="train", config=config)
    test_dataset = DesedMelDataset(split="test", config=config)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNNModel().to(device)
    optimizer = config["optimizer"](model.parameters(), lr=learning_rate)
    loss_fn = config["loss"]()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.get("scheduler_factor", 0.5),
        patience=config.get("scheduler_patience", 3),
    )
    early_stopping = EarlyStopping(patience=config.get("early_stopping_patience", 6))

    train_losses_history = []
    val_losses_history = []
    val_f1_history = []

    print("\n--- Entraînement CRNN ---")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for step, (X_train, Y_train) in enumerate(train_dataloader, start=1):
            X_train, Y_train = X_train.to(device), Y_train.to(device)

            optimizer.zero_grad()
            X_logits = model(X_train)
            loss = loss_fn(X_logits, Y_train)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            print_progress_bar(
                step,
                len(train_dataloader),
                prefix=f"Epoch {epoch+1}/{epochs}",
                suffix=f"| Batch Loss: {loss.item():.4f}",
            )

        avg_train_loss = running_train_loss / len(train_dataloader)
        train_losses_history.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.inference_mode():
            for X_test, Y_test in test_dataloader:
                X_test, Y_test = X_test.to(device), Y_test.to(device)

                Y_logits = model(X_test)
                val_loss = loss_fn(Y_logits, Y_test)
                running_val_loss += val_loss.item()

                Y_pred = torch.round(torch.sigmoid(Y_logits))
                all_preds.append(Y_pred.cpu())
                all_targets.append(Y_test.cpu())

        avg_val_loss = running_val_loss / len(test_dataloader)
        val_losses_history.append(avg_val_loss)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        epoch_f1 = f1_score(all_targets, all_preds, average="macro")
        val_f1_history.append(epoch_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"   -> Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {epoch_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print(f"Early stopping à l'epoch {epoch+1} (meilleure Val Loss : {early_stopping.best_loss:.4f})")
            break

    early_stopping.load_best_weights(model)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cnn_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Modèle CRNN sauvegardé dans : {save_path}")

    plot_history(train_losses_history, val_losses_history, val_f1_history, "CRNN")
    return model, {"train_loss": train_losses_history, "val_loss": val_losses_history, "val_f1": val_f1_history}


def constructOutput(svm_list, input_data):
    output = np.zeros(len(svm_list))
    for i, clf in enumerate(svm_list):
        output[i] = clf.predict(input_data)
    return output


def train_svm(config=None, save_dir="./Models/Saved_Models"):
    config = config if config is not None else read_config()
    n_classes = config["n_classes"]

    train_dataset = MicroDataLoader(split="train", config=config)
    test_dataset = MicroDataLoader(split="test", config=config)

    train_dataloader = DataLoader(train_dataset)
    test_dataloader = DataLoader(test_dataset)

    X_train, Y_train_all = [], []
    X_test, Y_test_all = [], []

    print("\n--- Extraction MFCC & Entraînement Multi-SVM ---")
    for step, (x, y) in enumerate(train_dataloader, start=1):
        X_train.append(x)
        Y_train_all.append(y)
        print_progress_bar(step, len(train_dataloader), prefix="MFCC Train")

    for step, (x, y) in enumerate(test_dataloader, start=1):
        X_test.append(x)
        Y_test_all.append(y)
        print_progress_bar(step, len(test_dataloader), prefix="MFCC Test ")

    X_train = np.array(X_train).squeeze()
    X_test = np.array(X_test).squeeze()
    Y_train_all = np.array(Y_train_all).squeeze()
    Y_test_all = np.array(Y_test_all).squeeze()

    p_grid = {"C": [0.1, 1.0, 10.0, 100.0], "gamma": [0.001, 0.01, 0.1]}

    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)

    svm_lists = []
    svm_f1_scores = []
    precision_scores = []
    recall_scores = []
    best_params = []
    best_scores = []

    for i in range(n_classes):
        Y_train = Y_train_all[:, i]
        Y_test = Y_test_all[:, i]

        clf = GridSearchCV(
            estimator=SVC(class_weight="balanced"),
            param_grid=p_grid,
            scoring="f1",
        )
        clf.fit(X_train_n, Y_train)

        best_scores.append(clf.best_score_)
        best_params.append(clf.best_params_)
        svm_lists.append(clf)

        Y_pred = clf.predict(X_test_n)

        f1_cls = f1_score(Y_test, Y_pred)
        svm_f1_scores.append(f1_cls)
        precision_scores.append(precision_score(Y_test, Y_pred))
        recall_scores.append(recall_score(Y_test, Y_pred))

        print_progress_bar(
            i + 1,
            n_classes,
            prefix="GridSearch SVM",
            suffix=f"| Classe {i+1}/{n_classes} F1: {f1_cls:.4f}",
        )

    print(f"F1 Macro global : {np.mean(svm_f1_scores):.4f}")
    print("Scores F1 par classe :", svm_f1_scores)
    print("Précisions par classe :", precision_scores)
    print("Recall par classe :", recall_scores)
    print("Meilleurs paramètres :", best_params)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "multi_svm.joblib")
    joblib.dump({"models": svm_lists, "scaler": scaler}, save_path)
    print(f"Modèle Multi-SVM sauvegardé dans : {save_path}")

    return svm_lists, scaler


if __name__ == "__main__":
    cfg = read_config()
    # train_m5(cfg)
    # train_cnn(cfg)
    # train_svm(cfg)