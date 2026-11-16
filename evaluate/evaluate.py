import pandas as pd
import torch
from codecarbon import EmissionsTracker

from test import test_crnn, test_m5, test_svm
from training import train_cnn, train_m5, train_svm
from Utils.config_reader import read_config
from Utils.datasets import MEL_HOP_LENGTH, N_MELS, DesedDataset, DesedMelDataset, make_loader
from Utils.stats import get_pytorch_stats, get_svm_stats

"""

"""


def measure_training(model_name, train_function):
    """Mesure l'énergie et les émissions d'une fonction d'entraînement."""
    tracker = EmissionsTracker(
        project_name=f"{model_name}_train",
        save_to_file=False,
        log_level="error",
    )

    tracker.start()
    model, history = train_function()
    tracker.stop()

    energy_data = tracker.final_emissions_data
    energy_consumption = energy_data.energy_consumed
    emissions = energy_data.emissions

    print(f"{model_name} (Train):")
    print(f"Energy consumption: {energy_consumption:.6f} kWh")
    print(f"Emissions: {emissions:.6f} kg CO2eq")

    return model, history, energy_consumption, emissions


def measure_testing(model_name, model, test_function, test_data, n_samples, config):
    """Mesure l'énergie par passage (clip) et le F1 macro en test."""
    tracker = EmissionsTracker(
        project_name=f"{model_name}_test",
        save_to_file=False,
        log_level="error",
    )

    tracker.start()
    test_f1 = test_function(model, test_data, config=config)
    tracker.stop()

    total_test_energy = tracker.final_emissions_data.energy_consumed
    energy_per_passage = total_test_energy / n_samples if n_samples > 0 else 0.0

    print(f"{model_name} (Test):")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Energy per passage: {energy_per_passage:.9f} kWh")

    return test_f1, energy_per_passage


def main():
    cfg = read_config()

    n_test = len(DesedDataset(split="test", config=cfg))

    n_audio_samples = int(cfg["target_rate"]) * int(cfg["audio_seconds"]) 
    n_frames = n_audio_samples // MEL_HOP_LENGTH + 1                   

    models = {
        "M5": {
            "train": lambda: train_m5(cfg),
            "test": test_m5,
            # loader construit ici : l'itération (donc le chargement audio) a lieu dans la mesure
            "test_data": make_loader(DesedDataset(split="test", config=cfg), 32, False, cfg),
            "sample_input": torch.randn(1, n_audio_samples),
        },
        "CRNN": {
            "train": lambda: train_cnn(cfg),
            "test": test_crnn,
            "test_data": make_loader(DesedMelDataset(split="test", config=cfg), 32, False, cfg),
            "sample_input": torch.randn(1, N_MELS, n_frames),
        },
        "SVM": {
            "train": lambda: train_svm(cfg),
            "test": test_svm,
            "test_data": None,  # test_svm extrait lui-même les MFCC, dans la mesure d'énergie
            "sample_input": None,
        },
    }

    measures = {
        "model_name": [],
        "total_params": [],
        "trainable_params": [],
        "flops_per_passage": [],
        "training_consumption_kwh": [],
        "training_emissions_kg": [],
        "best_val_f1": [],  
        "test_f1": [],
        "test_consumption_kwh_per_passage": [],
    }

    for model_name, model_cfg in models.items():
        model, history, train_energy, train_emissions = measure_training(model_name, model_cfg["train"])

        best_val_f1 = max(history["val_f1"]) if history.get("val_f1") else 0.0

        if model_name == "SVM":
            svm_list, _scaler = model
            total_params, flops = get_svm_stats(svm_list)
            trainable_params = total_params
        else:
            total_params, flops = get_pytorch_stats(model, model_cfg["sample_input"])
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        test_f1, test_energy_per_passage = measure_testing(
            model_name, model, model_cfg["test"], model_cfg["test_data"], n_test, cfg
        )

        measures["model_name"].append(model_name)
        measures["total_params"].append(total_params)
        measures["trainable_params"].append(trainable_params)
        measures["flops_per_passage"].append(flops)
        measures["training_consumption_kwh"].append(train_energy)
        measures["training_emissions_kg"].append(train_emissions)
        measures["best_val_f1"].append(best_val_f1)
        measures["test_f1"].append(test_f1)
        measures["test_consumption_kwh_per_passage"].append(test_energy_per_passage)

    pd.DataFrame(measures).to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()