import os

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

from Utils.config_reader import read_config

CLASSES = [
    "Alarm_bell_ringing", "Blender", "Cat", "Dishes", "Dog",
    "Electric_shaver_toothbrush", "Frying", "Running_water",
    "Speech", "Vacuum_cleaner",
]

# paramètres du mel-spectrogramme (importés aussi par evaluate.py pour les sample_input)
N_FFT = 2048
MEL_HOP_LENGTH = 256
N_MELS = 128


def _resolve(split, config):
    if split in ("train", "val"):
        return config["train_metadata_path"], config["raw_train_datas_path"]
    if split in ("test", "eval"):
        return config["test_metadata_path"], config["raw_test_datas_path"]
    raise ValueError(f"split inconnu : {split!r} (attendu : train, val, test)")


def _select_files(files, split, config):
    files = np.array(sorted(files))
    if split in ("test", "eval"):
        return files
    rng = np.random.RandomState(int(config["seed"]))
    perm = rng.permutation(len(files))
    n_val = int(len(files) * float(config.get("val_ratio", 0.2)))
    chosen = perm[:n_val] if split == "val" else perm[n_val:]
    return files[np.sort(chosen)]


class _BaseDesed(Dataset):
    """Logique commune : lecture du tsv, labels multi-hot précalculés, chargement du waveform."""

    classes = CLASSES

    def __init__(self, split="train", annotations_file=None, sounds_dir=None,
                 transform=None, config=None):
        self.config = config if config is not None else read_config()
        default_annotations, default_dir = _resolve(split, self.config)

        annotations = pd.read_csv(annotations_file or default_annotations, sep="\t")
        self.sounds_dir = sounds_dir or default_dir
        self.transform = transform
        self.files = _select_files(annotations["filename"].unique(), split, self.config)
        self.labels = self._build_labels(annotations)
        self._resamplers = {}

    def _build_labels(self, annotations):
        index = {c: i for i, c in enumerate(self.classes)}
        labels = {}
        for filename, events in annotations.groupby("filename")["event_label"]:
            v = np.zeros(len(self.classes), dtype=np.float32)
            for event in events:
                if event in index:
                    v[index[event]] = 1.0
            labels[filename] = v
        return labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = str(self.files[idx])
        x = self.extract(os.path.join(self.sounds_dir, filename))
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.from_numpy(self.labels[filename])

    def extract(self, path):
        raise NotImplementedError

    def _load_waveform(self, path):
        """Waveform mono, rééchantillonné, de longueur fixe : (1, target_rate * audio_seconds)."""
        target_sr = int(self.config["target_rate"])
        target_len = target_sr * int(self.config["audio_seconds"])

        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True)

        if sr != target_sr:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(orig_freq=sr, new_freq=target_sr)
            wav = self._resamplers[sr](wav)

        n = wav.shape[1]
        if n > target_len:
            wav = wav[:, :target_len]
        elif n < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - n))
        return wav


# dataset pour le modèle sur waveform (M5) -> (1, 160000)
class DesedDataset(_BaseDesed):

    def extract(self, path):
        return self._load_waveform(path)


# dataset pour le mel-spectrogramme (CRNN) -> (1, 128, 626)
class DesedMelDataset(_BaseDesed):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mel = T.MelSpectrogram(
            sample_rate=int(self.config["target_rate"]),
            n_fft=N_FFT,
            hop_length=MEL_HOP_LENGTH,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=N_MELS,
            mel_scale="htk",
        )
        self.to_db = T.AmplitudeToDB()

    def extract(self, path):
        return self.to_db(self.mel(self._load_waveform(path)))


# dataset pour les SVM -> vecteur de 40 valeurs (20 moyennes + 20 écarts-types de MFCC)
class MicroDataLoader(_BaseDesed):

    def extract(self, path, n_mfcc=20):
        sr = int(self.config["target_rate"])
        y, sr = librosa.load(path, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def make_loader(dataset, batch_size=32, shuffle=False, config=None):
    config = config if config is not None else {}
    num_workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=str(config.get("device", "cpu")).startswith("cuda"),
    )


def load_svm_arrays(split, config=None, progress=None):
    """Extrait les MFCC d'un split et renvoie (X, Y) en numpy. `progress(step, total)` est optionnel."""
    dataset = MicroDataLoader(split=split, config=config)
    X, Y = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X.append(x)
        Y.append(y.numpy())
        if progress is not None:
            progress(i + 1, len(dataset))
    return np.stack(X), np.stack(Y)