"""
this file contains all the needed from scratch models needed for the evaluations
there are : a 1D cnn on the waveform, a svm (to test a classical AI without nn), a 2D crnn on the spectrogram.

A transformer based nn is also planned to be implemented
"""

#imports
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from torch import nn

#block functions
def conv1d_block(c_in, c_out, k=3, pool=4):
    layers = [
        nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
    ]
    if pool:
        layers.append(nn.MaxPool1d(pool))
    return nn.Sequential(*layers)

def conv2d_block(c_in, c_out, k=3):
    layers = [
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
    ]
    return nn.Sequential(*layers)

#1D conv model (M5 model)

class M5model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=64, stride=4, padding=30, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            conv1d_block(64, 64),
            conv1d_block(64, 128),
            conv1d_block(128, 256),
            conv1d_block(256, 256, pool=0),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)
        x = self.features(x)
        x = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.fc(self.dropout(x))

#CRNN model

class CRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            conv2d_block(64, 128),
            conv2d_block(128, 256),
            conv2d_block(256, 512),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 10)
        self.rnn = nn.LSTM(
            input_size=4096,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            dropout=0.5,
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x, _ = torch.max(x, dim=1)
        return self.fc(self.dropout(x))

#SVM model

class MultiSVMModel:
    """
    Encapsule les 10 classifieurs SVM (un par événement) avec leur GridSearchCV.
    """
    def __init__(self, num_classes=10, param_grid=None):
        self.num_classes = num_classes
        self.param_grid = param_grid or {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": [0.001, 0.01, 0.1],
        }
        self.classifiers = [
            GridSearchCV(
                estimator=SVC(class_weight="balanced"),
                param_grid=self.param_grid,
                scoring="f1",
            )
            for _ in range(self.num_classes)
        ]

    def fit(self, X, Y):
        for i in range(self.num_classes):
            self.classifiers[i].fit(X, Y[:, i])
        return self

    def predict(self, X):
        # Gère aussi bien un seul échantillon (1D/2D) qu'un batch complet
        X_in = np.atleast_2d(X)
        preds = np.column_stack([clf.predict(X_in) for clf in self.classifiers])
        return preds[0] if X.ndim == 1 else preds

