import time
import numpy as np
import torch
from codecarbon import EmissionsTracker
from torch.utils.flop_counter import FlopCounterMode


def get_pytorch_stats(model, sample_input):
    model_cpu = model.to("cpu").eval()
    x = sample_input.unsqueeze(0).to("cpu")  # ajout dimension batch (1, ...)

    num_params = sum(p.numel() for p in model_cpu.parameters())

    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            model_cpu(x)
        flops = flop_counter.get_total_flops()

    return num_params, flops


def get_svm_stats(svm_list, n_features=40):
    # Nombre total de vecteurs de support sur les 10 SVM + les 10 coefficients alpha + 10 biais
    total_sv = sum(clf.best_estimator_.support_vectors_.shape[0] for clf in svm_list)
    num_params = total_sv * n_features + total_sv + len(svm_list)

    # Inférence RBF : calcul de distance euclidienne (2 * d ops par SV) + noyau exp + somme pondérée
    flops = total_sv * (2 * n_features + 5)
    return num_params, flops