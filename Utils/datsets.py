import os
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
from config_reader import read_config


#datasset pour le modèle sur waveform

class DesedDataset(Dataset):

    classes = [
        "Alarm_bell_ringing", "Blender", "Cat", "Dishes", "Dog",
        "Electric_shaver_toothbrush", "Frying", "Running_water",
        "Speech", "Vacuum_cleaner"
    ]

    def __init__(self, annotations_file, sounds_dir,
                 classes=classes, transform=None, config=None):

        self.config = config if config is not None else read_config()
        self.annotations = pd.read_csv(
            annotations_file, sep="\t"
        )

        self.sounds_dir = sounds_dir
        self.classes = classes
        self.transform = transform

        self.files = self.annotations["filename"].unique()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        snd_path = os.path.join(
            self.sounds_dir, filename
        )

        sound = self.load_audio(
            snd_path,
            target_sr=self.config["target_rate"],
            duration=self.config["audio_seconds"]
        )

        annotations = self.annotations[
            self.annotations["filename"] == filename
        ]

        label = torch.zeros(len(self.classes))

        for event in annotations["event_label"]:
            if event in self.classes:
                label[self.classes.index(event)] = 1.0

        return sound, label

    def load_audio(self, path, target_sr=16000, duration=10):
        waveform, sr = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=target_sr
            )
            waveform = resampler(waveform)

        target_length = target_sr * duration

        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, target_length - waveform.shape[1])
            )

        return waveform


#dataset pour le melspectrogramme

class DesedMelDataset(Dataset):

    classes = [
        "Alarm_bell_ringing", "Blender", "Cat", "Dishes", "Dog",
        "Electric_shaver_toothbrush", "Frying", "Running_water",
        "Speech", "Vacuum_cleaner"
    ]

    def __init__(self, annotations_file, sounds_dir,
                 classes=classes, transform=None, config=None):

        self.config = config if config is not None else read_config()
        self.annotations = pd.read_csv(
            annotations_file, sep="\t"
        )

        self.sounds_dir = sounds_dir
        self.classes = classes
        self.transform = transform

        self.files = self.annotations["filename"].unique()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        snd_path = os.path.join(
            self.sounds_dir, filename
        )

        melspec = self.load_audio(snd_path)

        annotations = self.annotations[
            self.annotations["filename"] == filename
        ]

        label = torch.zeros(len(self.classes))

        for event in annotations["event_label"]:
            if event in self.classes:
                label[self.classes.index(event)] = 1.0

        return melspec, label

    def load_audio(self, path):
        target_sr = self.config["target_rate"]
        duration = self.config["audio_seconds"]

        SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(path)

        SPEECH_WAVEFORM = F.resample(
            SPEECH_WAVEFORM,
            orig_freq=SAMPLE_RATE,
            new_freq=target_sr,
        )

        target_length = target_sr * duration

        if SPEECH_WAVEFORM.shape[1] > target_length:
            SPEECH_WAVEFORM = SPEECH_WAVEFORM[:, :target_length]
        elif SPEECH_WAVEFORM.shape[1] < target_length:
            SPEECH_WAVEFORM = torch.nn.functional.pad(
                SPEECH_WAVEFORM,
                (0, target_length - SPEECH_WAVEFORM.shape[1])
            )

        n_fft = 2048
        win_length = None
        hop_length = 256
        n_mels = 128

        mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )

        SPEECH_WAVEFORM = torch.mean(SPEECH_WAVEFORM, dim=0, keepdim=True)

        melspec = mel_spectrogram(SPEECH_WAVEFORM)

        melspec_db = T.AmplitudeToDB()(melspec)

        return melspec_db


#dataset pour les svms

class MicroDataLoader(Dataset):
    """
    petit "dataloader" pour avoir les données sur ce truc non nn

    doit renvoyer un path vers un audio (train ou data selon)
    """

    def __init__(self, csv_path, raw_data_path, config=None):

        self.config = config if config is not None else read_config()
        self.raw_data_path = raw_data_path

        self.annotations = pd.read_csv(
            csv_path, sep="\t"
        )

        self.files = self.annotations["filename"].unique()

        self.classes = self.annotations["event_label"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sound_path = os.path.join(self.raw_data_path, self.files[idx])

        labels = np.zeros(10)

        filename = self.files[idx]

        annotations = self.annotations[
            self.annotations["filename"] == filename
        ]

        events = sorted(annotations["event_label"].unique())

        label = [1.0 if c in events else 0.0 for c in sorted(self.classes.unique())]

        label = np.array(label)

        return self.extract_mfcc(sound_path, sr=self.config["target_rate"]), label

    def extract_mfcc(self, audio_path, sr=16000, n_mfcc=20):
        """
        fonction d'extractions de caractéristiques qui renvoie un vecteur de dimension 40 (20 moyennes + 20 variances)
        """

        # chargement de l'audio
        y, sr = librosa.load(
            audio_path,
            sr=sr,
            mono=True
        )

        # acquisition des features
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512
        )

        # moyenne et variance sur les matrices mfcc pour avoir le vecteur finale
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # les 20 moyennes et les 20 variances sont concaténées dans un gros vecteur
        features = np.concatenate([
            mfcc_mean,
            mfcc_std
        ])

        return features