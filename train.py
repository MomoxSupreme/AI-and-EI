"""Training is implemented by :class:`models.CedAudioClassifier`.

The Lightning Trainer is configured and started in ``main.py``.
"""


import torch
import torch.nn as nn

def step(self, batch, stage: str):
    embeddings, labels = batch
    outputs = self(embeddings)
    loss = nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
    predictions = outputs.logits.sigmoid() >= 0.5
    exact_match = (predictions == labels.bool()).all(dim=1).float().mean()
    true_positives = (predictions & labels.bool()).sum().float()
    predicted_positives = predictions.sum().float()
    actual_positives = labels.sum()
    precision = true_positives / predicted_positives.clamp_min(1)
    recall = true_positives / actual_positives.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    self.log(
        f"{stage}_loss",
        loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=labels.size(0),
    )
    self.log(
        f"{stage}_exact_match",
        exact_match,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=labels.size(0),
    )
    self.log(f"{stage}_f1", f1, on_step=False, on_epoch=True, prog_bar=True,
             batch_size=labels.size(0))
    return loss


def train_step(lightning_module, batch):
        return step(lightning_module, batch, "train")


def valid_step(lightning_module, batch):
        return step(lightning_module, batch, "val")