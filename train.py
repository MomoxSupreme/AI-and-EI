"""Training is implemented by :class:`models.CedAudioClassifier`.

The Lightning Trainer is configured and started in ``main.py``.
"""


import torch.nn as nn

def step(self, batch, stage: str):
    embeddings, labels = batch
    outputs = self(embeddings)
    loss = nn.functional.cross_entropy(outputs.logits, labels)
    accuracy = (outputs.logits.argmax(dim=1) == labels).float().mean()
    self.log(
        f"{stage}_loss",
        loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=labels.size(0),
    )
    self.log(
        f"{stage}_accuracy",
        accuracy,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=labels.size(0),
    )
    return loss


def train_step(lightning_module, batch):
        return step(lightning_module, batch, "train")


def valid_step(lightning_module, batch):
        return step(lightning_module, batch, "val")