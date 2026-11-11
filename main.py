import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from config import TARGET_RATE, MODELS_CED, N_CLASSES
from dataset import CachedEmbeddingDataset
from models import CedAudioClassifier
import os


def main():
    model = CedAudioClassifier(n_classes=N_CLASSES, model_name=MODELS_CED[0])
    embeddings_path = os.path.join("embeddings", f"{MODELS_CED[0]}_embeddings.pt")
    dataset = CachedEmbeddingDataset(
        target_len=10 * TARGET_RATE,
        fs=TARGET_RATE,
        batch_size=8,
        num_frames=1,
        preprocess_function=model.preprocess_function,
        backbone=model,
        cache_path="models/ced_embeddings.pt",
    )

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(2026),
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    checkpoint = ModelCheckpoint(
        dirpath="models",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-model",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.001,
        mode="min",
        verbose=True,
    )

    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint, early_stopping],
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )


if __name__ == "__main__":
    main()
