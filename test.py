"""Evaluation helpers for the multilabel audio classification task."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections.abc import Mapping
from config import N_CLASSES


from torchmetrics.functional.classification import multilabel_f1_score

# Must match the averaging used for val_f1 in train.py ("macro" or "micro").
F1_AVERAGE = "macro"


def multilabel_f1(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """F1 over multi-hot tensors of shape (n_items, N_CLASSES)."""
    return multilabel_f1_score(
        predictions.int(), targets.int(), num_labels=N_CLASSES, average=F1_AVERAGE
    ).item()

class EvaluationDataset(Dataset):
    """Load labeled evaluation passages from the DESED public eval split."""

    def __init__(
        self,
        preprocess_function: Callable[[str], dict[str, torch.Tensor]],
        metadata_path: str = "data/DESED_public_eval/metadata/eval/public_weak.tsv",
        audio_dir: str = "data/DESED_public_eval/audio/eval",
    ) -> None:
        self.preprocess_function = preprocess_function
        self.audio_dir = audio_dir
        self.classes = OrderedDict(
            {
                "Alarm_bell_ringing": 0,
                "Blender": 1,
                "Cat": 2,
                "Dishes": 3,
                "Dog": 4,
                "Electric_shaver_toothbrush": 5,
                "Frying": 6,
                "Running_water": 7,
                "Speech": 8,
                "Vacuum_cleaner": 9,
            }
        )
        labels: dict[str, set[int]] = {}
        with open(metadata_path, encoding="utf-8") as metadata:
            next(metadata)
            for line in metadata:
                filename, event_label = line.rstrip("\n").split("\t")
                if event_label in self.classes:
                    labels.setdefault(filename, set()).add(self.classes[event_label])
        self.samples = sorted(labels.items())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        filename, class_ids = self.samples[index]
        inputs = self.preprocess_function(os.path.join(self.audio_dir, filename))
        target = torch.zeros(N_CLASSES, dtype=torch.float32)
        target[list(class_ids)] = 1.0
        return inputs, target


def test(
    data_loader: DataLoader,
    model: torch.nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    device: torch.device | None = None,
) -> tuple[float, float, float]:
    """Evaluate a multilabel model and return ``(loss, exact_match_accuracy)``."""
    criterion = criterion or torch.nn.functional.binary_cross_entropy_with_logits
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    all_predictions, all_targets = [], []
    with torch.inference_mode():
        for inputs, target in tqdm(data_loader, total=len(data_loader), desc="Testing"):
            target = target.to(device)
            if isinstance(inputs, Mapping):
                inputs = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                embeddings = model.encode(**inputs)
                output = model(embeddings)
            else:
                output = model(inputs.to(device))
            logits = getattr(output, "logits", output)
            total_loss += criterion(logits, target).item() * target.size(0)
            predictions = logits.sigmoid() >= 0.5
            total_correct += (predictions == target.bool()).all(dim=1).sum().item()
            total_items += target.size(0)
            all_predictions.append(predictions.cpu())
            all_targets.append(target.bool().cpu())

    if total_items == 0:
        raise ValueError("The evaluation dataset is empty.")
    f1 = multilabel_f1(torch.cat(all_predictions), torch.cat(all_targets))
    return total_loss / total_items, total_correct / total_items, f1
