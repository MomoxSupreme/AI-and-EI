
import os
import numpy as np
import random
from torchaudio.transforms import MelSpectrogram
import torchaudio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
# from prettytable import PrettyTable
from codecarbon import EmissionsTracker
# from thop import profile, clever_format
import pandas as pd
from config import DEVICE
from collections import OrderedDict
from utils import pad_audio, to_mono


class CachedEmbeddingDataset(Dataset):
    def __init__(
        self,
        target_len=None,
        fs=None,
        batch_size=None,
        num_frames=None,
        preprocess_function=None,
        backbone=None,
        cache_path=None,
    ):
        self.cache_path = cache_path
        if cache_path is not None and os.path.exists(cache_path):
            self.load_cache(cache_path)
            return

        data_path = "data"
        data_folder = os.path.join(data_path, "DESED_public_eval/audio/train/")
        metadata_file = os.path.join(data_folder, "..", "public_weak.tsv")
        self.data_folder = data_folder
        self.load_classes_labels()
        self.load_meta(metadata_file)
        self.batch_size = batch_size
        self.fs = fs
        self.target_len = target_len
        self.file_list = os.listdir(data_folder)
        self.num_frames = num_frames
        self.preprocess_function = preprocess_function

        if backbone is not None and cache_path is not None:
            self.cache_embeddings(backbone, cache_path)
            self.load_cache(cache_path)

    def __len__(self):
        if hasattr(self, "labels"):
            return len(self.labels)
        return len(self.file_list)

    def __getitem__(self, idx):
        if hasattr(self, "embeddings"):
            return self.embeddings[idx], self.labels[idx]

        filename = self.file_list[idx]
        file_path = os.path.join(self.data_folder, filename)

        X = self.preprocess_function(file_path)
        y = torch.zeros(len(self.classes_labels), dtype=torch.float32)
        y[self.metadata[filename]] = 1.0

        return X, y
    
    def cache_embeddings(self, backbone, cache_path):
        was_training = backbone.training
        backbone.to(DEVICE)
        backbone.eval()
        embeddings, labels = [], []
        with torch.no_grad():
            for idx in tqdm(range(len(self)), desc="Encoding dataset"):
                inputs, label = self[idx]
                inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
                embeddings.append(backbone.encode(**inputs).squeeze(0).cpu())
                labels.append(label)
        if was_training:
            backbone.train()

        data = {
            "embeddings": torch.stack(embeddings),
            "labels": torch.stack(labels),
        }
        torch.save(data, cache_path)

    def load_cache(self, cache_path):
        data = torch.load(cache_path, map_location=DEVICE)
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]

    def load_meta(self, meta_file):
        # Initialize an empty metadata dictionary
        self.metadata = {}
        class_counts = {class_name: 0 for class_name in self.classes_labels}

        # Read the metadata file (weak.tsv) while skipping the first line (headers)
        with open(meta_file, 'r') as file:
            next(file)  # Skip the first line (headers)
            for line in file:
                parts = line.strip().split()
                filename = parts[0]

                # Extract all event labels from the line
                event_labels = parts[1].split(',')

                # Map each event label to a number and store in a list
                for label in event_labels:
                    if label in class_counts:
                        class_counts[label] += 1

                encoded_event_labels = [
                    self.classes_labels[label]
                    for label in event_labels
                    if label in self.classes_labels
                ]
                current_labels = self.metadata.setdefault(filename, [])
                for encoded_label in encoded_event_labels:
                    if encoded_label not in current_labels:
                        current_labels.append(encoded_label)

        print("Dataset samples per class:")
        pprint(class_counts)


    def load_classes_labels(self):

        # Define the label-to-number mapping dictionary
        self.classes_labels = OrderedDict(
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

        # Create a data loader
        
     