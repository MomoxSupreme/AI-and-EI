"""
Modèles faits maison pour l'évaluation :
 - M5model       : CNN 1D sur waveform
 - CNN2DModel    : CNN 2D sur log-mel, sans récurrence
 - CRNNModel     : même CNN 2D + GRU bidirectionnel (esprit baseline DCASE, version légère)
 - MultiSVMModel : un SVM par classe (non réseau)

Chaque réseau a un paramètre `width` (multiplicateur du nombre de canaux) : on obtient ainsi
plusieurs points F1 / paramètres / FLOPs / énergie pour une même architecture.
"""

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from torch import nn


def _c(base, width, minimum=8):
    """Nombre de canaux = base * width, arrondi, avec un plancher."""
    return max(minimum, int(round(base * width)))


def model_name(kind, width):
    return f"{kind}_w{width}"


def checkpoint_name(kind, width, seed):
    return f"{model_name(kind, width)}_seed{seed}"


# ----------------------------------------------------------------------------
# blocs
# ----------------------------------------------------------------------------
def conv1d_block(c_in, c_out, k=3, pool=4):
    layers = [
        nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
    ]
    if pool:
        layers.append(nn.MaxPool1d(pool))
    return nn.Sequential(*layers)


def conv2d_block(c_in, c_out, pool=(2, 2), k=3):
    """pool = (pooling sur les mels, pooling sur le temps)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=pool, stride=pool),
    )


def _pool_time(x):
    """(B, T, D) -> (B, 2D) : moyenne + max sur le temps."""
    return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)


# ----------------------------------------------------------------------------
# M5 : CNN 1D sur waveform
# ----------------------------------------------------------------------------
class M5model(nn.Module):
    """width=1.0 correspond à la version d'origine (64, 64, 128, 256, 256 canaux)."""

    def __init__(self, num_classes, width=1.0):
        super().__init__()
        c1, c2, c3, c4 = (_c(b, width) for b in (64, 64, 128, 256))
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=64, stride=4, padding=30, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(4),
            conv1d_block(c1, c2),
            conv1d_block(c2, c3),
            conv1d_block(c3, c4),
            conv1d_block(c4, c4, pool=0),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * c4, num_classes)

    def forward(self, x):
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)
        x = self.features(x)
        x = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.fc(self.dropout(x))


# ----------------------------------------------------------------------------
# CNN 2D sur log-mel, et CRNN
# ----------------------------------------------------------------------------
class _MelCNN(nn.Module):
    """
    (B, 1, mel, temps) -> (B, T', C).
    Pooling du temps seulement dans les 2 premiers blocs (626 -> 156 frames), pooling des mels dans les 5 blocs
    (128 -> 4), puis moyenne sur les mels restants. Un BatchNorm2d(1) en entrée normalise l'échelle des dB,
    ce qui accélère nettement l'apprentissage.
    """

    BASE_CHANNELS = (16, 32, 64, 128, 128)
    POOLS = ((2, 2), (2, 2), (2, 1), (2, 1), (2, 1))

    def __init__(self, width):
        super().__init__()
        channels = [_c(b, width) for b in self.BASE_CHANNELS]
        layers = [nn.BatchNorm2d(1)]
        c_in = 1
        for c_out, pool in zip(channels, self.POOLS):
            layers.append(conv2d_block(c_in, c_out, pool))
            c_in = c_out
        self.blocks = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x):
        x = self.blocks(x)          # (B, C, mel', T')
        x = x.mean(dim=2)           # (B, C, T')
        return x.permute(0, 2, 1)   # (B, T', C)


class CNN2DModel(nn.Module):
    """CNN 2D seul : pas de récurrence, sert à isoler l'apport du GRU."""

    def __init__(self, num_classes=10, width=1.0):
        super().__init__()
        self.backbone = _MelCNN(width)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * self.backbone.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)                       # (B, T', C)
        return self.fc(self.dropout(_pool_time(x)))


class CRNNModel(nn.Module):
    """CNN 2D + GRU bidirectionnel + pooling moyenne/max sur le temps."""

    def __init__(self, num_classes=10, width=1.0):
        super().__init__()
        self.backbone = _MelCNN(width)
        hidden = _c(64, width)
        self.rnn = nn.GRU(self.backbone.out_channels, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(4 * hidden, num_classes)

    def forward(self, x):
        x = self.backbone(x)       # (B, T', C)
        x, _ = self.rnn(x)         # (B, T', 2*hidden)
        return self.fc(self.dropout(_pool_time(x)))


MODEL_KINDS = ("m5", "cnn2d", "crnn")


def build_model(kind, width=1.0, num_classes=10):
    if kind == "m5":
        return M5model(num_classes, width)
    if kind == "cnn2d":
        return CNN2DModel(num_classes, width)
    if kind == "crnn":
        return CRNNModel(num_classes, width)
    raise ValueError(f"modèle inconnu : {kind!r} (attendu : {MODEL_KINDS})")


# ----------------------------------------------------------------------------
# SVM
# ----------------------------------------------------------------------------
class MultiSVMModel:
    """
    Encapsule les 10 classifieurs SVM (un par événement) avec leur GridSearchCV.
    (Non utilisée par le pipeline actuel : train_svm construit directement ses GridSearchCV.)
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
        X_in = np.atleast_2d(X)
        preds = np.column_stack([clf.predict(X_in) for clf in self.classifiers])
        return preds[0] if X.ndim == 1 else preds