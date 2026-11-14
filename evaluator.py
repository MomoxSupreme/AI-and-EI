"""Train and evaluate every configured model while recording energy metrics."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
from codecarbon import EmissionsTracker
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from config import MODELS_CED, MODELS_WHISPER, N_CLASSES, SEED, TARGET_RATE
from dataset import CachedEmbeddingDataset
from models import CedAudioClassifier, WhisperAudioClassifier
from test import EvaluationDataset, test

RESULTS_FILE = "evaluator_results.csv"


class BestValidationAccuracy(Callback):
    """Keep the highest epoch-level exact-match validation accuracy."""

    def __init__(self) -> None:
        self.best = 0.0

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        value = trainer.callback_metrics.get("val_exact_match")
        if value is not None:
            self.best = max(self.best, float(value.detach().cpu()))


def _energy_consumed(tracker: EmissionsTracker) -> float:
    """Return CodeCarbon's measured energy in kWh."""
    data = getattr(tracker, "final_emissions_data", None)
    energy = getattr(data, "energy_consumed", None)
    if energy is None:
        energy = getattr(tracker, "_total_energy", None)
    if energy is None:
        raise RuntimeError("CodeCarbon did not expose energy_consumed.")
    return float(energy)


def _tracked_fit(trainer: L.Trainer, model: L.LightningModule, train_loader: DataLoader,
                 valid_loader: DataLoader, accuracy_callback: BestValidationAccuracy) -> tuple[float, float]:
    tracker = EmissionsTracker(tracking_mode="process", log_level="warning")
    tracker.start()
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    finally:
        tracker.stop()
    return _energy_consumed(tracker), accuracy_callback.best


def evaluate_model(model_name: str, model_class: type[L.LightningModule]) -> dict[str, Any]:
    model = model_class(n_classes=N_CLASSES, model_name=model_name)
    embeddings_file = Path("embeddings") / f"{model_name.split('/')[-1]}_multilabel_embeddings.pt"
    embeddings_file.parent.mkdir(exist_ok=True)
    dataset = CachedEmbeddingDataset(
        target_len=10 * TARGET_RATE,
        fs=TARGET_RATE,
        batch_size=8,
        num_frames=1,
        preprocess_function=model.preprocess_function,
        backbone=model,
        cache_path=str(embeddings_file),
    )
    train_size = int(0.8 * len(dataset))
    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8)
    best_accuracy = BestValidationAccuracy()
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join("models", model_name),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-model",
    )
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val_loss", patience=10, min_delta=0.001, mode="min"),
            best_accuracy,
        ],
        logger=TensorBoardLogger("lightning_logs", name=model_name.split("/")[-1]),
    )
    training_consumption, best_val_accuracy = _tracked_fit(
        trainer, model, train_loader, valid_loader, best_accuracy
    )
    if checkpoint.best_model_path:
        model = model_class.load_from_checkpoint(checkpoint.best_model_path)

    eval_dataset = EvaluationDataset(model.preprocess_function)
    eval_loader = DataLoader(eval_dataset, batch_size=8)
    tracker = EmissionsTracker(tracking_mode="process", log_level="warning")
    tracker.start()
    try:
        _, test_accuracy = test(eval_loader, model)
    finally:
        tracker.stop()
    test_consumption = _energy_consumed(tracker)
    return {
        "model_name": model_name,
        "training_consumption_kwh": training_consumption,
        "best_val_accuracy": best_val_accuracy,
        "test_consumption_kwh_per_passage": test_consumption / len(eval_dataset),
        "test_accuracy": test_accuracy,
    }


def main(model_names: list[str] | None = None) -> None:
    specs = {name: CedAudioClassifier for name in MODELS_CED}
    specs.update({name: WhisperAudioClassifier for name in MODELS_WHISPER})
    selected = model_names or list(specs)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "model_name", "training_consumption_kwh", "best_val_accuracy",
            "test_consumption_kwh_per_passage", "test_accuracy",
        ])
        writer.writeheader()
        for model_name in selected:
            if model_name not in specs:
                raise ValueError(f"Unknown model: {model_name}")
            writer.writerow(evaluate_model(model_name, specs[model_name]))
            output.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Subset of configured model names.")
    main(parser.parse_args().models)
