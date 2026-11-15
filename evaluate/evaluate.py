import pandas as pd
import torch
from training import *
from Utils.config_reader import read_config
from Utils.stats import get_pytorch_stats, get_svm_stats
from codecarbon import EmissionsTracker


"""
this file will construct the csv file containing the datas usefull for the evaluations
we use codecarbon to measure environmental impact (notably energy consumption that has also an economic impact)
and we use native pytorch functions to get the params and flops
"""


def measure_training(model_name, train_function):
    """
    Measure the energy consumption and emissions of a training function.
    """
    tracker = EmissionsTracker(
        project_name=f"{model_name}_train",
        save_to_file=False,
        log_level="error"
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


def measure_testing(model_name, model, test_function, test_loader):
    """
    Measure the energy consumption per passage and F1 score during inference/testing.
    """
    tracker = EmissionsTracker(
        project_name=f"{model_name}_test",
        save_to_file=False,
        log_level="error"
    )

    tracker.start()
    test_f1 = test_function(model, test_loader)
    tracker.stop()

    energy_data = tracker.final_emissions_data
    total_test_energy = energy_data.energy_consumed

    # Calcul de la consommation moyenne par échantillon (passage)
    num_samples = len(test_loader.dataset) if hasattr(test_loader, "dataset") else len(test_loader)
    energy_per_passage = total_test_energy / num_samples if num_samples > 0 else 0.0

    print(f"{model_name} (Test):")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Energy per passage: {energy_per_passage:.9f} kWh")

    return test_f1, energy_per_passage


def main():
    models = {
        "M5": {
            "train": train_m5,
            "test": test_M5,
            "test_data": test_loader_m5,     
            "sample_input": sample_input_m5 
        },
        "CRNN": {
            "train": train_cnn,
            "test": test_CRNN,
            "test_data": test_loader_crnn,     
            "sample_input": sample_input_crnn  
        },
        "SVM": {
            "train": train_svm,
            "test": test_SVM,
            "test_data": X_test_svm,           # Données de test (array/tuple) pour SVM
            "sample_input": None
        }
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
        "test_consumption_kwh_per_passage": []
    }

    for model_name, cfg in models.items():
        model, history, train_energy, train_emissions = measure_training(
            model_name,
            cfg["train"]
        )

        best_val_f1 = max(history["val_f1"]) if history.get("val_f1") else 0.0

        if model_name == "SVM":
            total_params, flops = get_svm_stats(model)
            trainable_params = total_params
        else:
            total_params, flops = get_pytorch_stats(model, cfg["sample_input"])
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        test_f1, test_energy_per_passage = measure_testing(
            model_name,
            model,
            cfg["test"],
            cfg["test_data"]
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

    measures_df = pd.DataFrame(measures)
    measures_df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()