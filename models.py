import torch
import torch.nn as nn
import torchaudio
import lightning.pytorch as L
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from config import *
from train import train_step, valid_step
from transformers import WhisperModel, WhisperFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput

class BaseModel(L.LightningModule):

    @staticmethod
    def _pool_embeddings(outputs):
        embeddings = getattr(outputs, "last_hidden_state", None)
        if embeddings is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states:
                embeddings = hidden_states[-1]
        if embeddings is None:
            embeddings = getattr(outputs, "logits", None)
        if embeddings is None:
            raise RuntimeError(
                "The backbone did not return hidden states. "
                "Enable hidden-state outputs on the backbone before caching embeddings."
            )
        return embeddings.mean(dim=1) if embeddings.ndim == 3 else embeddings

    def _freeze_backbone(self) -> None:
        """Freeze everything, then unfreeze only the classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.outputlayer.parameters():
            param.requires_grad = True

    @classmethod
    def from_pretrained(cls, out_dir: str, model_name: str = MODEL_NAME):
        """Reload a fine-tuned head on top of the (still frozen) encoder."""
        classifier = cls.__new__(cls)
        L.LightningModule.__init__(classifier)
        classifier.feature_extractor = AutoFeatureExtractor.from_pretrained(out_dir)
        classifier.model = AutoModelForAudioClassification.from_pretrained(out_dir)
        classifier.backbone = classifier.model.encoder
        classifier.outputlayer = classifier.model.outputlayer
        classifier.n_classes = classifier.model.config.num_labels
        classifier.learning_rate = LEARNING_RATE
        classifier._freeze_backbone()
        return classifier


    def training_step(self, batch, batch_idx):
        return train_step(self, batch)

    def validation_step(self, batch, batch_idx):
        valid_step(self, batch)

    def on_train_epoch_start(self):
        learning_rate = self.optimizers().param_groups[0]["lr"]
        self.log(
            "learning_rate",
            learning_rate,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = OPTIMIZER(
            self.trainable_parameters(), lr=LEARNING_RATE
        )
        scheduler = SCHEDULER(
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

    def predict(self, audio_path: str) -> list[int]:
        """Return all class ids whose predicted probability reaches 0.5."""
        inputs = self.prepare_audio(audio_path)
        with torch.no_grad():
            embeddings = self.encode(**inputs)
            logits = self(embeddings).logits
        return torch.where(logits.squeeze().sigmoid() >= 0.5)[0].tolist()

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
    def encode(self, input_values, **kwargs):
        if input_values.ndim == 4 and input_values.shape[1] == 1:
            input_values = input_values.squeeze(1)
        with torch.no_grad():
            outputs = self.backbone(input_values=input_values)
        return self._pool_embeddings(outputs)

    def forward(self, embeddings, **kwargs):
        logits = self.outputlayer(embeddings)
        return SequenceClassifierOutput(logits=logits)

class CedAudioClassifier(BaseModel):
    """Combines the CED feature extractor and classification model into one module.

    The pretrained encoder is frozen; only a new classification head with
    `n_classes` outputs is trainable.
    """

    def __init__(
        self,
        n_classes: int,
        model_name: str = MODEL_NAME,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes

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
        self.backbone = self.model.encoder
        self.outputlayer = self.model.outputlayer

        # Keep config consistent for save / reload / inference
        self.model.config.num_labels = n_classes
        self.model.config.id2label = {i: f"class_{i}" for i in range(n_classes)}
        self.model.config.label2id = {v: k for k, v in self.model.config.id2label.items()}

        self._freeze_backbone()
    # ------------------------------------------------------------------ #
    # Audio preprocessing
    # ------------------------------------------------------------------ #
    

    


class WhisperAudioClassifier(BaseModel):
    def __init__(
        self,
        n_classes: int,
        model_name: str = "openai/whisper-tiny",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_name
        )

        # Load Whisper and extract only its encoder
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder

        hidden_size = whisper.config.d_model  # 384

        # Classification head
        self.outputlayer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, n_classes),
        )

        self.outputlayer.apply(whisper._init_weights)

        del whisper

        self.backbone = self.encoder
        self._freeze_backbone()

    def encode(self, input_features, **kwargs):
        if input_features.ndim == 4 and input_features.shape[1] == 1:
            input_features = input_features.squeeze(1)
        with torch.no_grad():
            outputs = self.backbone(input_features=input_features)
        return self._pool_embeddings(outputs)
