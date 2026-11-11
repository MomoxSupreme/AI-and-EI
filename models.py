import torch
import torch.nn as nn
import torchaudio
import lightning.pytorch as L
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from config import *
from train import train_step, valid_step


class CedAudioClassifier(L.LightningModule):
    """Combines the CED feature extractor and classification model into one module.

    The pretrained encoder is frozen; only a new classification head with
    `n_classes` outputs is trainable.
    """

    def __init__(
        self,
        n_classes: int,
        model_name: str = MODEL_NAME,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.learning_rate = learning_rate

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Replace the 527-class AudioSet head with a fresh n_classes head
        in_features = self.model.outputlayer[1].in_features  # 768
        self.model.outputlayer = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-5),
            nn.Linear(in_features, n_classes),
        )
        self.model.outputlayer.apply(self.model._init_weights)

        # Keep config consistent for save / reload / inference
        self.model.config.num_labels = n_classes
        self.model.config.id2label = {i: f"class_{i}" for i in range(n_classes)}
        self.model.config.label2id = {v: k for k, v in self.model.config.id2label.items()}

        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze everything, then unfreeze only the classification head."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.outputlayer.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------ #
    # Audio preprocessing
    # ------------------------------------------------------------------ #
    def prepare_audio(self, audio_path: str) -> dict:
        """Load, resample to 16 kHz, and feature-extract a single audio file."""
        audio, sampling_rate = torchaudio.load(audio_path)
        if sampling_rate != TARGET_RATE:
            audio = torchaudio.transforms.Resample(
                orig_freq=sampling_rate, new_freq=TARGET_RATE
            )(audio)
        if audio.ndim != 2:
            raise ValueError(f"Expected audio with shape [channels, samples], got {audio.shape}")
        audio = audio.mean(dim=0)
        target_samples = 10 * TARGET_RATE
        if audio.shape[-1] < target_samples:
            audio = nn.functional.pad(audio, (0, target_samples - audio.shape[-1]))
        else:
            audio = audio[:target_samples]
        return self.feature_extractor(audio, sampling_rate=TARGET_RATE, return_tensors="pt")

    def preprocess_function(self, audio_path):
        return self.prepare_audio(audio_path)

    # ------------------------------------------------------------------ #
    # nn.Module API
    # ------------------------------------------------------------------ #
    def forward(self, input_values, labels=None, **kwargs):
        if input_values.ndim == 4 and input_values.shape[1] == 1:
            input_values = input_values.squeeze(1)
        return self.model(input_values=input_values, labels=labels, **kwargs)


    def training_step(self, batch, batch_idx):
        return train_step(self, batch)

    def validation_step(self, batch, batch_idx):
        valid_step(self, batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(), lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict(self, audio_path: str) -> int:
        """Return the predicted class id for one audio file."""
        inputs = self.prepare_audio(audio_path)
        with torch.no_grad():
            logits = self(**inputs).logits
        return logits.squeeze().argmax().item()

    # ------------------------------------------------------------------ #
    # Training helpers
    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizer(self, lr: float = 1e-3):
        """Optimizer over the trainable head only."""
        return torch.optim.AdamW(self.trainable_parameters(), lr=lr)

    def save_pretrained(self, out_dir: str):
        self.model.save_pretrained(out_dir)
        self.feature_extractor.save_pretrained(out_dir)

    @classmethod
    def from_pretrained(cls, out_dir: str, model_name: str = MODEL_NAME):
        """Reload a fine-tuned head on top of the (still frozen) encoder."""
        classifier = cls.__new__(cls)
        L.LightningModule.__init__(classifier)
        classifier.feature_extractor = AutoFeatureExtractor.from_pretrained(out_dir)
        classifier.model = AutoModelForAudioClassification.from_pretrained(out_dir)
        classifier.n_classes = classifier.model.config.num_labels
        classifier.learning_rate = 1e-3
        classifier._freeze_backbone()
        return classifier
