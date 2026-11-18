"""Train and evaluate every configured model while recording energy metrics."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
from tqdm import tqdm
from codecarbon import EmissionsTracker
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from dataset import CachedEmbeddingDataset
from models import CedAudioClassifier, WhisperAudioClassifier
from test import EvaluationDataset, multilabel_f1, test
from torch.utils.flop_counter import FlopCounterMode
from collections.abc import Mapping
from audio_RAG import CLASS_NAMES, top_similar_classes
from config import *


class BestValidationAccuracy(Callback):
    """Keep the highest epoch-level validation exact-match accuracy and F1."""

    def __init__(self) -> None:
        self.best = 0.0
        self.best_f1 = 0.0

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        value = trainer.callback_metrics.get("val_exact_match")
        if value is not None:
            self.best = max(self.best, float(value.detach().cpu()))
        f1 = trainer.callback_metrics.get("val_f1")
        if f1 is not None:
            self.best_f1 = max(self.best_f1, float(f1.detach().cpu()))
            
            
            
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
                 valid_loader: DataLoader, accuracy_callback: BestValidationAccuracy) -> tuple[float, float,float]:
    tracker = EmissionsTracker(tracking_mode="process", log_level="warning")
    tracker.start()
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    finally:
        tracker.stop()
    return _energy_consumed(tracker), accuracy_callback.best, accuracy_callback.best_f1

def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _count_flops(model: torch.nn.Module, eval_dataset: EvaluationDataset) -> int:
    """FLOPs of one inference pass (encode + classifier head) for a single passage."""
    device = next(model.parameters()).device
    model.eval()
    # batch_size=1 and the DataLoader's collation, so shapes match what test() sees
    inputs, _ = next(iter(DataLoader(eval_dataset, batch_size=1)))
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        if isinstance(inputs, Mapping):
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            output = model(model.encode(**inputs))
        else:
            output = model(inputs.to(device))
    return counter.get_total_flops()

def _count_encode_flops(model: torch.nn.Module, eval_dataset: EvaluationDataset) -> int:
    """FLOPs of one encoding pass (backbone only, no classification head)."""
    device = next(model.parameters()).device
    model.eval()
    inputs, _ = eval_dataset[0]
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model.encode(**inputs)
    return counter.get_total_flops()


def evaluate_rag(model_name: str, model_class: type[L.LightningModule],encoding_consumption: float = 0.0) -> dict[str, Any]:
    """Retrieval-based classification: 'training' = encoding the train set,
    'test' = encoding each eval passage and ranking classes by similarity."""
    model = model_class(n_classes=N_CLASSES, model_name=model_name)
    embeddings_file = Path("embeddings") / f"{model_name.split('/')[-1]}_rag_embeddings.pt"
    embeddings_file.parent.mkdir(exist_ok=True)
    # Force re-encoding, otherwise the cache would be loaded and the energy would be ~0.
    embeddings_file.unlink(missing_ok=True)

    # "Training" phase: encode the whole train dataset (no head, no gradient steps).
    tracker = EmissionsTracker(tracking_mode="process", log_level="warning")
    tracker.start()
    try:
        dataset = CachedEmbeddingDataset(
            target_len=10 * TARGET_RATE,
            fs=TARGET_RATE,
            batch_size=8,
            num_frames=1,
            preprocess_function=model.preprocess_function,
            backbone=model,
            cache_path=str(embeddings_file),
        )
    finally:
        tracker.stop()
    training_consumption = _energy_consumed(tracker)

    train_embeddings = dataset.embeddings.to(DEVICE)
    train_labels = dataset.labels.to(DEVICE)

    eval_dataset = EvaluationDataset(model.preprocess_function)
    model.to(DEVICE)
    model.eval()

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    flops_per_passage = _count_encode_flops(model, eval_dataset)

    # "Test" phase: encode each eval passage and rank classes by similarity.
    correct = 0
    all_predictions, all_targets = [], []
    tracker = EmissionsTracker(tracking_mode="process", log_level="warning")
    tracker.start()
    try:
        with torch.inference_mode():
            for index in tqdm(range(len(eval_dataset)), desc="RAG testing"):
                inputs, target = eval_dataset[index]
                inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
                query = model.encode(**inputs).squeeze(0)
                true_ids = set(target.nonzero(as_tuple=True)[0].tolist())
                matches = top_similar_classes(
                    train_embeddings, train_labels, query, number_of_classes=len(true_ids)
                )
                predicted_ids = {CLASS_NAMES.index(match["class"]) for match in matches}
                correct += int(predicted_ids == true_ids)
                prediction = torch.zeros(N_CLASSES, dtype=torch.int)
                prediction[list(predicted_ids)] = 1
                all_predictions.append(prediction)
                all_targets.append(target.int())
    finally:
        tracker.stop()
    test_consumption = _energy_consumed(tracker)
    test_f1 = multilabel_f1(torch.stack(all_predictions), torch.stack(all_targets))
    return {
        "model_name": f"{model_name} (RAG)",
        "head": "none",
        "total_params": backbone_params,
        "trainable_params": 0,
        "flops_per_passage": flops_per_passage,
        "training_consumption_kwh": training_consumption,
        "encoding_consumption_kwh": encoding_consumption,           # train-set vectorization
        "total_training_consumption_kwh": training_consumption + encoding_consumption,
        "best_val_accuracy": "",
        "best_val_f1": "",
        "test_consumption_kwh_per_passage": test_consumption / len(eval_dataset),
        "test_accuracy": correct / len(eval_dataset),
        "test_f1": test_f1,
    }
def evaluate_model(model_name, model_class, head_type="linear",
                   encoding_consumption: float = 0.0) -> dict[str, Any]:
    L.seed_everything(SEED)  # same head initialisation on every run
    model = model_class(n_classes=N_CLASSES, model_name=model_name, head_type=head_type)
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
        dirpath=os.path.join("models", model_name, head_type),   # one folder per head
        monitor="val_loss", mode="min", save_top_k=1, filename="best-model",
    )
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_DELTA, mode="min"),
            best_accuracy,
        ],
        logger=TensorBoardLogger("lightning_logs", name=f"{model_name.split('/')[-1]}_{head_type}"),
    )
    training_consumption, best_val_accuracy, best_val_f1 = _tracked_fit(
        trainer, model, train_loader, valid_loader, best_accuracy
    )
    if checkpoint.best_model_path:
        model = model_class.load_from_checkpoint(checkpoint.best_model_path, map_location=DEVICE)

    eval_dataset = EvaluationDataset(model.preprocess_function)
    eval_loader = DataLoader(eval_dataset, batch_size=8)
    eval_dataset = EvaluationDataset(model.preprocess_function)
    eval_loader = DataLoader(eval_dataset, batch_size=8)

    total_params, trainable_params = _count_parameters(model)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    flops_per_passage = _count_flops(model, eval_dataset)

    tracker = EmissionsTracker(tracking_mode="process", log_level="warning")
    tracker = EmissionsTracker(tracking_mode="process", log_level="warning")
    tracker.start()
    try:
        _, test_accuracy, test_f1 = test(eval_loader, model)
    finally:
        tracker.stop()
    test_consumption = _energy_consumed(tracker)
    return {
        "model_name": model_name,
        "head": head_type,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops_per_passage": flops_per_passage,
        "training_consumption_kwh": training_consumption,           # head training only
        "encoding_consumption_kwh": encoding_consumption,           # train-set vectorization
        "total_training_consumption_kwh": training_consumption + encoding_consumption,
        "best_val_accuracy": best_val_accuracy,
        "best_val_f1": best_val_f1,
        "test_consumption_kwh_per_passage": test_consumption / len(eval_dataset),
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
    }


def main(model_names: list[str] | None = None, heads: list[str] | None = None) -> None:
    heads = heads or list(HEAD_TYPES)
    specs = {name: CedAudioClassifier for name in MODELS_CED}
    specs.update({name: WhisperAudioClassifier for name in MODELS_WHISPER})
    selected = model_names or list(specs)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "model_name", "head", "total_params", "trainable_params", "flops_per_passage",
            "training_consumption_kwh", "encoding_consumption_kwh", "total_training_consumption_kwh", "best_val_accuracy", "best_val_f1",
            "test_consumption_kwh_per_passage", "test_accuracy", "test_f1",
        ])
        writer.writeheader()
        for model_name in selected:
            rag_row = evaluate_rag(model_name, specs[model_name])
            encoding = rag_row["training_consumption_kwh"]
            rag_row["encoding_consumption_kwh"] = encoding
            rag_row["total_training_consumption_kwh"] = encoding
            for head in heads:
                writer.writerow(evaluate_model(model_name, specs[model_name], head, encoding))
                output.flush()
            writer.writerow(rag_row)
            output.flush()
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Subset of configured model names.")
    parser.add_argument("--heads", nargs="*", choices=HEAD_TYPES, default=list(HEAD_TYPES),
                        help="Classification heads to train and evaluate.")
    args = parser.parse_args()
    main(args.models, args.heads)
