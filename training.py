"""
fichier qui regroupe l'entraînement des différents modèles
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from config_reader import read_config
from datasets import DesedDataset, DesedMelDataset, MicroDataLoader
from models import CNNModel, M5model


#entraînement du M5

def train_m5(config=None, batch_size=32, epochs=10):
    config = config if config is not None else read_config()
    device = config["device"]
    learning_rate = config["learning_rate"]
    n_classes = config["n_classes"]

    train_dataset = DesedDataset(split="train", config=config)
    test_dataset = DesedDataset(split="test", config=config)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = M5model(n_classes)
    loss_fn = config["loss"]()
    optimizer = config["optimizer"](model.parameters(), lr=learning_rate)

    model.to(device)

    train_losses_history = []
    test_losses_history = []
    test_acc_history = []

    for epoch in range(epochs):
        model.train()

        for data in train_dataloader:
            x = data[0].to(device)
            y_target = data[1].to(device)

            x_logits = model(x)
            loss = loss_fn(x_logits, y_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses_history.append(loss.item())

        model.eval()
        for data in test_dataloader:
            with torch.inference_mode():
                test_logits = data[0].to(device)
                y_test = data[1].to(device)

                test_logits = model(test_logits)
                test_loss = loss_fn(test_logits, y_test)
                test_losses_history.append(test_loss.item())

                y_pred = torch.round(torch.sigmoid(test_logits))
                test_acc = f1_score(y_test.to("cpu"), y_pred.to("cpu"), average="macro")
                test_acc_history.append(test_acc)

        progress_percent = (epoch + 1) / epochs * 100
        progress_view = ["▮"] * int(progress_percent) + ["▯"] * (100 - int(progress_percent))
        print("loss : ", train_losses_history[-1])
        print(f"Epoch {epoch+1}/{epochs} [{''.join(progress_view)}] ")

    plt.figure(figsize=(10, 5))
    plt.plot(test_acc_history, label="Loss par batch", color="orangered")
    plt.xlabel("Iterations (Batches)")
    plt.ylabel("Valeur de la Loss")
    plt.title("Évolution de la Loss au cours de l'entraînement")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(test_losses_history)

    f1_mean = 0
    n = 0
    model.eval()
    with torch.inference_mode():
        for data in test_dataloader:
            n += 1
            x = data[0].to(device)
            y = data[1].to(device)

            y_logits = model(x)
            y_pred = torch.round(torch.sigmoid(y_logits))

            f1_mean += f1_score(y.to("cpu"), y_pred.to("cpu"), average="macro")

    f1_mean = f1_mean / n
    print("la f1_mean est :", f1_mean)
    return model


#entraînement du crnn

def train_cnn(config=None, batch_size=32, epochs=1):
    config = config if config is not None else read_config()
    device = config["device"]
    learning_rate = config["learning_rate"]

    train_dataset = DesedMelDataset(split="train", config=config)
    test_dataset = DesedMelDataset(split="test", config=config)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = CNNModel()

    optimizer = config["optimizer"](model.parameters(), lr=learning_rate)
    loss_fn = config["loss"]()

    model.to(device)

    for epoch in range(epochs):
        model.train()

        for batch in train_dataloader:
            X_train = batch[0].to(device)
            Y_train = batch[1].to(device)

            X_logits = model(X_train)
            loss = loss_fn(X_logits, Y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        mean_f1_score = 0

        with torch.inference_mode():
            for batch in test_dataloader:
                X_test = batch[0].to(device)
                Y_test = batch[1].to(device)

                Y_logits = model(X_test)
                loss = loss_fn(Y_logits, Y_train)

                Y_pred = torch.round(torch.sigmoid(Y_logits))
                Y_pred = Y_pred.to("cpu")
                Y_test.to = Y_test.to("cpu")

                mean_f1_score += f1_score(Y_test, Y_pred, average="macro")

    mean_f1_score = mean_f1_score / len(test_dataloader)
    print(mean_f1_score)
    return model


#entraînement des svm

def constructOutput(svm_list, input_data):
    output = np.zeros(10)
    for i, clf in enumerate(svm_list):
        Y_pred = clf.predict(input_data)
        output[i] = Y_pred
    return output


def train_svm(config=None):
    config = config if config is not None else read_config()
    n_classes = config["n_classes"]

    train_dataset = MicroDataLoader(split="train", config=config)
    test_dataset = MicroDataLoader(split="test", config=config)

    train_dataloader = DataLoader(train_dataset)
    test_dataloader = DataLoader(test_dataset)

    X_train, Y_train_all = [], []
    X_test, Y_test_all = [], []

    for data in train_dataloader:
        X_train.append(data[0])
        Y_train_all.append(data[1])

    for data in test_dataloader:
        X_test.append(data[0])
        Y_test_all.append(data[1])

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
    verif_sum = []
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

        svm_f1_scores.append(f1_score(Y_test, Y_pred))
        verif_sum.append(Y_pred.sum().item())
        precision_scores.append(precision_score(Y_test, Y_pred))
        recall_scores.append(recall_score(Y_test, Y_pred))

    print("scores f1 :", svm_f1_scores)
    print("précisions par classe :", precision_scores)
    print("recall par classe", recall_scores)
    print("meilleurs paramètres par classe", best_params)
    print("meilleurs scores pour les meilleurs paramètres par classe", best_scores)
    print(verif_sum)

    X_unique_test, Y_unique_test = test_dataset[47]
    X_unique_test = X_unique_test.reshape(1, -1)
    X_unique_test_n = scaler.transform(X_unique_test)

    test = constructOutput(svm_lists, X_unique_test_n)
    print(test)
    print(Y_unique_test)

    return svm_lists, scaler


if __name__ == "__main__":
    cfg = read_config()

    # train_m5(cfg)
    # train_cnn(cfg)
    # train_svm(cfg)