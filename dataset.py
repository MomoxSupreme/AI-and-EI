
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
# from prettytable import PrettyTable
from codecarbon import EmissionsTracker
# from thop import profile, clever_format
import pandas as pd
from collections import OrderedDict
from utils import pad_audio, to_mono



class WeakDataset(Dataset):
    def __init__(self, target_len, fs, batch_size, num_frames, preprocess_function):
        data_path = 'data'     
        data_folder = os.path.join(data_path, 'DESED_public_eval/audio/train/')
        metadata_file = os.path.join(data_folder, "..", 'public_weak.tsv')
        self.data_folder = data_folder
        self.load_classes_labels()
        self.load_meta(metadata_file)
        self.batch_size = batch_size
        self.fs = fs
        self.target_len = 10 * fs
        self.file_list = os.listdir(data_folder)
        self.num_frames = num_frames
        self.preprocess_function = preprocess_function
        self.target_len = target_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_folder, filename)
    
        X = self.preprocess_function(file_path)
        y = self.metadata[filename][0]
        
        return X, y

    def load_meta(self, meta_file):
        # Initialize an empty metadata dictionary
        self.metadata = {}

        # Read the metadata file (weak.tsv) while skipping the first line (headers)
        with open(meta_file, 'r') as file:
            next(file)  # Skip the first line (headers)
            for line in file:
                parts = line.strip().split()
                filename = parts[0]

                # Extract all event labels from the line
                event_labels = parts[1].split(',')

                # Map each event label to a number and store in a list
                encoded_event_labels = [self.classes_labels.get(label, -1) for label in event_labels]

                # Add the filename and encoded event labels to the metadata dictionary
                self.metadata[filename] = encoded_event_labels


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
        
     