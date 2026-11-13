import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from config import TARGET_RATE, MODELS_CED, N_CLASSES, SEED, MODELS_WHISPER
from dataset import CachedEmbeddingDataset
from models import CedAudioClassifier, WhisperAudioClassifier
import os


def main():
    model_name = MODELS_CED[0]
    model = CedAudioClassifier(n_classes=N_CLASSES, model_name=model_name)
    embeddings_path = "embeddings"
    embeddings_file = os.path.join(
        "embeddings", f"{model_name.split('/')[-1]}_multilabel_embeddings.pt"
    )
    if not os.path.exists(embeddings_path):
        os.mkdir(embeddings_path)
    dataset = CachedEmbeddingDataset(
        target_len=10 * TARGET_RATE,
        fs=TARGET_RATE,
        batch_size=8,
        num_frames=1,
        preprocess_function=model.preprocess_function,
        backbone=model,
        cache_path=embeddings_file,
    )

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join("models", model_name),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-model",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.001,
        mode="min",
        verbose=True,
    )
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=model_name.split("/")[-1],
    )

    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint, early_stopping],
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )


if __name__ == "__main__":
    main()
